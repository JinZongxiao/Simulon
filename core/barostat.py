"""
core/barostat.py — Berendsen 等压调节器（NPT 集成）

压力单位全程使用 bar：
  1 eV/Å³ = 160 217.66 bar

参考：
  Berendsen et al., J. Chem. Phys. 81, 3684 (1984)
  标准等压缩放：μ = [1 - κ·(dt/τ_p)·(P₀ - P)]^(1/3)
"""
from __future__ import annotations
import torch


# 单位换算：eV/Å³ → bar
_EV_ANG3_TO_BAR: float = 160_217.66


class BerendsenBarostat:
    """
    各向同性 Berendsen 压力控制器。

    Parameters
    ----------
    molecular : reader.Molecular
        系统分子对象（持有 box、coordinates、graph_data.pos）。
    target_pressure : float
        目标压力，单位 bar。
    tau_p : float
        压力弛豫时间，单位 ps。
    compressibility : float
        等温压缩率，单位 bar⁻¹。水的默认值约 4.57e-5 bar⁻¹。
    mu_max : float
        单步最大缩放幅度（clamp），防止数值爆炸，默认 ±5%。
    """

    def __init__(
        self,
        molecular,
        target_pressure: float,
        tau_p: float,
        compressibility: float = 4.57e-5,
        mu_max: float = 0.05,
    ):
        self.molecular = molecular
        self.target_P = float(target_pressure)
        self.tau_p = float(tau_p)
        self.kappa = float(compressibility)
        self.mu_max = float(mu_max)

    # ─── 公共接口 ─────────────────────────────────────────────────────────────
    def step(
        self,
        dt: float,
        kinetic_energy: torch.Tensor,
        virial: torch.Tensor,
    ) -> float:
        """
        执行一步压力调节。

        Parameters
        ----------
        dt : float
            积分步长，单位 ps。
        kinetic_energy : Tensor
            当前动能（标量），单位 eV。
        virial : Tensor
            当前维里（scalar）= Σ_edges Fmag·r，单位 eV。
            注意：力场返回的 virial 只计算了上三角边，
            index_add_ 已对称，因此这里直接使用全体贡献。

        Returns
        -------
        float
            当前瞬时压力，单位 bar。
        """
        mol = self.molecular

        # ── 1. 计算瞬时压力 ──────────────────────────────────────────────────
        V = self._volume()                              # Å³
        # P_inst = (2KE + W) / (3V)  [eV/Å³] → bar
        # 力场的 virial = Σ r·F（标量，边的单向贡献已 ×2 via index_add_）
        # 标准维里定理：W = Σ_{i<j} r_ij·F_ij = virial（LJ/EAM 的实现已是全边和）
        P_ev_ang3 = (2.0 * kinetic_energy + virial) / (3.0 * V)
        P_inst = float(P_ev_ang3) * _EV_ANG3_TO_BAR   # bar

        # ── 2. Berendsen 缩放因子 μ ──────────────────────────────────────────
        # μ³ = 1 - κ·(dt/τ_p)·(P₀ - P)
        mu3 = 1.0 - self.kappa * (dt / self.tau_p) * (self.target_P - P_inst)
        # 防止 μ³ 为负（极端情况）
        mu3 = max(mu3, (1.0 - self.mu_max) ** 3)
        mu3 = min(mu3, (1.0 + self.mu_max) ** 3)
        mu = mu3 ** (1.0 / 3.0)

        # ── 3. 缩放盒子 ──────────────────────────────────────────────────────
        if hasattr(mol, 'box'):
            mol.box.scale(mu)
        # （backward compat：无 box 时不做盒子缩放，调用方负责）

        # ── 4. 缩放坐标 ──────────────────────────────────────────────────────
        mol.coordinates.data.mul_(mu)
        if hasattr(mol.graph_data, 'pos'):
            mol.graph_data.pos.data.mul_(mu)

        # ── 5. 强制邻居列表重建（重置位移累积量） ───────────────────────────
        mol.last_positions = mol.coordinates.clone()

        return P_inst

    # ─── 内部 ─────────────────────────────────────────────────────────────────
    def _volume(self) -> float:
        mol = self.molecular
        if hasattr(mol, 'box'):
            return mol.box.volume         # Å³（来自 Box.volume）
        # 向后兼容：纯标量 box_length
        L = float(mol.box_length)
        return L ** 3
