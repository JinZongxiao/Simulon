"""
core/integrator/integrator.py — Velocity-Verlet 积分器

支持三种系综：
  NVE  — 微正则（无热浴/压浴）
  NVT  — 正则（Langevin 热浴）
  NPT  — 等温等压（Langevin 热浴 + Berendsen 压浴，压浴由 BaseModel 调用）

用法示例：
    integrator = VerletIntegrator(
        molecular, dt=0.001,
        ensemble='NVT', temperature=(300, 300), gamma=0.01,
    )
    # NPT：BaseModel 构造时额外传入 barostat=BerendsenBarostat(...)
"""
import torch
import torch.nn as nn

from core.md_model import IntegratorInterface


class VerletIntegrator(IntegratorInterface, nn.Module):
    """
    速度-Verlet 积分器。

    Parameters
    ----------
    molecular : Molecular
        系统对象（含 coordinates、atom_velocities、atom_mass、device 等）。
    dt : float
        时间步长，单位 ps。
    ensemble : str | None
        'NVE'、'NVT'、'NPT' 或 None（等同 'NVE'）。
    temperature : tuple[float, float] | None
        (T_init, T_target)，单位 K。NVT/NPT 必须提供。
    gamma : float | None
        Langevin 阻尼系数（1/ps）。NVT/NPT 必须提供。
    """

    # 物理常数
    BOLTZMAN = 8.617333262e-5   # eV/K
    N_a      = 6.02214076e23    # mol⁻¹
    J_per_ev = 1.60218e-19      # J/eV

    def __init__(
        self,
        molecular,
        dt: float,
        ensemble: str | None = None,
        temperature: tuple | None = None,
        gamma: float | None = None,
        # legacy keyword kept for backward compat, ignored
        force_field=None,
    ):
        super().__init__()
        self.molecular = molecular
        self.dt = dt
        self.ensemble = (ensemble or 'NVE').upper()

        # 质量矩阵 [N, 3]，单位换算到 eV·ps²/Å²
        self.atom_mass = (
            molecular.atom_mass.unsqueeze(-1).expand_as(molecular.atom_velocities)
            * (10.0 / (self.N_a * self.J_per_ev))
        )

        # 预分配缓冲区
        self.new_coords = torch.empty_like(molecular.coordinates)
        self.vel_half   = torch.zeros_like(molecular.coordinates)

        # ── Langevin 热浴（NVT / NPT） ───────────────────────────────────────
        self.is_langevin_thermostat = False
        if self.ensemble in ('NVT', 'NPT'):
            if temperature is None or gamma is None:
                raise ValueError(
                    f"ensemble='{self.ensemble}' 需要同时提供 temperature 和 gamma 参数。"
                )
            T_init   = torch.tensor(float(temperature[0]), device=molecular.device)
            T_target = torch.tensor(float(temperature[1]), device=molecular.device)
            self.temperature = T_target
            self.gamma       = torch.tensor(float(gamma), device=molecular.device)
            # 以初始温度初始化速度
            molecular.set_maxwell_boltzmann_velocity(T_init)
            # 随机力幅值（Ornstein-Uhlenbeck 方案）
            exp_term = torch.exp(-2.0 * self.gamma * self.dt)
            self.random_force_factor = torch.sqrt(
                (self.BOLTZMAN * T_target) / self.atom_mass * (1.0 - exp_term)
            )
            self.is_langevin_thermostat = True

    # ─── Velocity-Verlet：第一半步 ───────────────────────────────────────────
    def first_half(self, forces_old: torch.Tensor):
        """
        r(t+dt) = r(t) + v(t)·dt + ½·a(t)·dt²
        v*(t+½dt) = v(t) + ½·a(t)·dt
        """
        accel = forces_old / self.atom_mass
        self.vel_half  = self.molecular.atom_velocities + 0.5 * accel * self.dt
        self.new_coords = self.molecular.coordinates + self.vel_half * self.dt
        self.molecular.update_coordinates(self.new_coords)

    # ─── Velocity-Verlet：第二半步 ───────────────────────────────────────────
    def second_half(self, forces_new: torch.Tensor) -> dict:
        """
        v(t+dt) = [v*(t+½dt) · λ_damp + ξ] + ½·a(t+dt)·dt
        """
        vel = self.vel_half

        # Langevin 阻尼 + 随机踢
        if self.is_langevin_thermostat:
            damp = torch.exp(-self.gamma * self.dt)
            xi   = torch.randn_like(self.random_force_factor) * self.random_force_factor
            vel  = vel * damp + xi

        accel_new = forces_new / self.atom_mass
        vel = vel + 0.5 * accel_new * self.dt
        self.molecular.update_velocities(vel)

        kin_energy = (0.5 * self.atom_mass * vel.pow(2)).sum()
        T = (2.0 / 3.0) * kin_energy / (self.molecular.atom_count * self.BOLTZMAN)
        return {
            'update_coordinates': self.new_coords,
            'kinetic_energy': kin_energy,
            'temperature': T,
        }

    # ─── PBC 折叠（供外部调用，坐标折回主盒子） ─────────────────────────────
    def apply_pbc(self, coordinates: torch.Tensor) -> torch.Tensor:
        """将坐标折回主盒子（支持 Box 对象和标量 box_length）。"""
        mol = self.molecular
        if hasattr(mol, 'box'):
            return mol.box.wrap_positions(coordinates)
        L = float(mol.box_length)
        return coordinates - torch.floor(coordinates / L) * L

    # ─── 单步 forward（向后兼容，不推荐在 BaseModel 中使用） ────────────────
    def forward(self, force: torch.Tensor) -> dict:
        self.first_half(force)
        # 若有外部 force_field 就重算（旧接口）
        self.second_half(force)
        kin_energy = (0.5 * self.atom_mass * self.molecular.atom_velocities.pow(2)).sum()
        T = (2.0 / 3.0) * kin_energy / (self.molecular.atom_count * self.BOLTZMAN)
        return {
            'update_coordinates': self.new_coords,
            'kinetic_energy': kin_energy,
            'temperature': T,
        }
