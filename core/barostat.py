"""
core/barostat.py — Barostat implementations

Pressure units are bar externally:
  1 eV/Å^3 = 160 217.66 bar

This module now provides:
  - BerendsenBarostat: legacy isotropic pressure control
  - AnisotropicNPTBarostat: diagonal extended-system anisotropic barostat
"""
from __future__ import annotations

import torch


_EV_ANG3_TO_BAR: float = 160_217.66
_BAR_TO_EV_ANG3: float = 1.0 / _EV_ANG3_TO_BAR
_KB_EV_K: float = 8.617333262e-5


class BerendsenBarostat:
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

    def step(
        self,
        dt: float,
        kinetic_energy: torch.Tensor,
        virial: torch.Tensor,
        kinetic_tensor: torch.Tensor | None = None,
        virial_tensor: torch.Tensor | None = None,
    ) -> float:
        mol = self.molecular
        V = self._volume()
        P_ev_ang3 = (2.0 * kinetic_energy + virial) / (3.0 * V)
        P_inst = float(P_ev_ang3) * _EV_ANG3_TO_BAR

        mu3 = 1.0 - self.kappa * (dt / self.tau_p) * (self.target_P - P_inst)
        mu3 = max(mu3, (1.0 - self.mu_max) ** 3)
        mu3 = min(mu3, (1.0 + self.mu_max) ** 3)
        mu = mu3 ** (1.0 / 3.0)

        if hasattr(mol, 'box'):
            mol.box.scale(mu)

        mol.coordinates.data.mul_(mu)
        if hasattr(mol.graph_data, 'pos'):
            mol.graph_data.pos.data.mul_(mu)
        mol.atom_velocities.data.mul_(mu)
        mol.last_positions = mol.coordinates.clone()
        return P_inst

    def _volume(self) -> float:
        mol = self.molecular
        if hasattr(mol, 'box'):
            return mol.box.volume
        L = float(mol.box_length)
        return L ** 3


class AnisotropicNPTBarostat:
    """
    Diagonal extended-system anisotropic barostat.

    This class evolves box strain rates eta_dot [1/ps] for each controlled axis:

      d eta_dot_i / dt = V * (P_int,i - P_ext,i) / W_i - gamma_p * eta_dot_i + noise_i / W_i

    and applies:

      H_new = diag(exp(dt * eta_dot)) @ H_old
      r_new = s @ H_new                      (fractional coordinates s kept fixed)
      v_new = exp(-dt * eta_dot) * v_old     (barostat coupling on peculiar velocity)

    It is diagonal-only by design, which is sufficient for tensile runs where
    the loading axis is controlled externally and lateral axes are pressure
    controlled.
    """

    def __init__(
        self,
        molecular,
        target_pressure_bar,
        temperature_k: float,
        tau_p: float = 0.2,
        gamma_p: float = 2.0,
        control_axes = (True, True, True),
        max_eta_dot: float = 0.05,
        stochastic: bool = False,
    ):
        self.molecular = molecular
        self.device = molecular.device
        self.target_pressure_bar = torch.as_tensor(
            target_pressure_bar, device=self.device, dtype=torch.float64
        )
        if self.target_pressure_bar.numel() == 1:
            self.target_pressure_bar = self.target_pressure_bar.repeat(3)
        if self.target_pressure_bar.numel() != 3:
            raise ValueError("target_pressure_bar must be scalar or length-3")
        self.control_axes = torch.tensor(control_axes, device=self.device, dtype=torch.bool)
        if self.control_axes.numel() != 3:
            raise ValueError("control_axes must have length 3")
        self.temperature_k = float(temperature_k)
        self.tau_p = float(tau_p)
        self.gamma_p = float(gamma_p)
        self.max_eta_dot = float(max_eta_dot)
        self.stochastic = bool(stochastic)

        dof = max(1, 3 * int(getattr(molecular, "atom_count", 1)))
        piston_mass = dof * _KB_EV_K * max(self.temperature_k, 1.0) * (self.tau_p ** 2)
        self.W = torch.full((3,), piston_mass, device=self.device, dtype=torch.float64)
        self.eta_dot = torch.zeros(3, device=self.device, dtype=torch.float64)

    def step(
        self,
        dt: float,
        kinetic_energy: torch.Tensor,
        virial: torch.Tensor,
        kinetic_tensor: torch.Tensor | None = None,
        virial_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kinetic_tensor is None or virial_tensor is None:
            raise ValueError("AnisotropicNPTBarostat requires kinetic_tensor and virial_tensor")

        mol = self.molecular
        dt = float(dt)
        volume = float(mol.box.volume)
        p_int_ev_ang3 = (kinetic_tensor.to(torch.float64) + virial_tensor.to(torch.float64)) / volume
        p_ext_ev_ang3 = self.target_pressure_bar * _BAR_TO_EV_ANG3

        axes = mol.box.H.to(torch.float64)
        axes = axes / torch.linalg.norm(axes, dim=1, keepdim=True).clamp_min(1e-12)
        p_axis_ev_ang3 = torch.einsum("ai,ij,aj->a", axes, p_int_ev_ang3, axes)

        delta_p = p_axis_ev_ang3 - p_ext_ev_ang3
        force_eta = volume * delta_p
        accel = force_eta / self.W

        if self.gamma_p > 0.0:
            accel = accel - self.gamma_p * self.eta_dot
        if self.stochastic:
            sigma_noise = torch.sqrt(2.0 * self.gamma_p * _KB_EV_K * self.temperature_k / self.W / max(dt, 1e-12))
            noise = torch.randn(3, device=self.device, dtype=torch.float64) * sigma_noise
            accel = accel + noise

        eta_dot_new = self.eta_dot + dt * accel
        eta_dot_new = torch.clamp(eta_dot_new, -self.max_eta_dot, self.max_eta_dot)
        eta_dot_new = torch.where(self.control_axes, eta_dot_new, torch.zeros_like(eta_dot_new))
        self.eta_dot = eta_dot_new

        scale = torch.exp(dt * self.eta_dot)
        scale = torch.where(self.control_axes, scale, torch.ones_like(scale))
        mu = torch.diag(scale)

        frac = mol.coordinates @ mol.box.H_inv.to(mol.coordinates)
        old_H = mol.box.H.detach().clone()
        mol.box.H = mu.to(old_H.dtype) @ old_H
        new_coords = frac @ mol.box.H.to(frac.dtype)
        new_coords = mol.box.wrap_positions(new_coords)
        mol.update_coordinates(new_coords)

        vel_scale = torch.exp(-dt * self.eta_dot).to(mol.atom_velocities.dtype)
        vel_scale = torch.where(self.control_axes, vel_scale, torch.ones_like(vel_scale))
        mol.atom_velocities = mol.atom_velocities * vel_scale.unsqueeze(0)

        new_lengths = mol.box.lengths.detach().clone()
        mol.box_length_cpu = float(new_lengths[0].item())
        if hasattr(mol, "box_length"):
            if torch.is_tensor(mol.box_length):
                if mol.box_length.ndim == 0:
                    mol.box_length = new_lengths[0].to(device=mol.device, dtype=mol.coordinates.dtype)
                else:
                    mol.box_length = new_lengths.to(device=mol.device, dtype=mol.coordinates.dtype)
            else:
                mol.box_length = float(new_lengths[0].item())

        mol.last_positions = mol.coordinates.clone()
        return (p_axis_ev_ang3 * _EV_ANG3_TO_BAR).to(mol.coordinates.dtype)
