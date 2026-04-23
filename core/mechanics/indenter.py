from __future__ import annotations

import torch
import torch.nn as nn


class SphericalIndenterForce(nn.Module):
    """
    Repulsive spherical indenter potential.

    For atoms inside the sphere, d < R:

      U_i = K / 3 * (R - d)^3
      F_i = K * (R - d)^2 * (r_i - c) / d

    K has units eV / A^3, so forces are eV / A.
    """

    def __init__(
        self,
        molecular,
        radius: float,
        stiffness: float,
        center,
    ):
        super().__init__()
        self.molecular = molecular
        self.device = molecular.device
        self.radius = float(radius)
        self.stiffness = float(stiffness)
        self.center = torch.as_tensor(center, device=self.device, dtype=molecular.coordinates.dtype)
        if self.center.numel() != 3:
            raise ValueError("center must have three components")
        self.force_on_indenter = torch.zeros(3, device=self.device, dtype=molecular.coordinates.dtype)
        self.last_energy = torch.zeros((), device=self.device, dtype=molecular.coordinates.dtype)
        self.contact_atoms = 0

    def set_center(self, center) -> None:
        self.center = torch.as_tensor(center, device=self.device, dtype=self.molecular.coordinates.dtype)

    def forward(self):
        coords = self.molecular.coordinates
        disp = coords - self.center.to(coords)
        dist = torch.linalg.norm(disp, dim=1).clamp_min(1e-12)
        penetration = self.radius - dist
        active = penetration > 0.0

        forces = torch.zeros_like(coords)
        energy = torch.zeros((), device=self.device, dtype=coords.dtype)
        virial_tensor = torch.zeros(3, 3, device=self.device, dtype=coords.dtype)

        if bool(active.any().item()):
            p = penetration[active]
            direction = disp[active] / dist[active].unsqueeze(1)
            force_vec = self.stiffness * p.pow(2).unsqueeze(1) * direction
            forces[active] = force_vec
            energy = (self.stiffness / 3.0) * p.pow(3).sum()
            virial_tensor = torch.einsum("ni,nj->ij", force_vec, coords[active])

        self.force_on_indenter = -forces.sum(dim=0)
        self.last_energy = energy.detach()
        self.contact_atoms = int(active.sum().item())
        return {
            "energy": energy,
            "forces": forces,
            "virial": torch.trace(virial_tensor),
            "virial_tensor": virial_tensor,
        }
