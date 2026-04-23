from __future__ import annotations

import torch


class UniaxialTensileLoader:
    """
    Apply homogeneous uniaxial tensile strain in lattice coordinates.

    Supports orthogonal and triclinic cells by scaling lattice vectors through
    the Box matrix H. Lateral response can be fixed or Poisson-like.
    """

    def __init__(
        self,
        molecular,
        axis: int = 0,
        strain_rate: float = 1.0e-3,
        lateral_mode: str = "fixed",
        poisson_ratio: float = 0.28,
        max_lateral_scale_step: float = 2.0e-3,
    ):
        self.molecular = molecular
        self.axis = int(axis)
        self.strain_rate = float(strain_rate)
        self.lateral_mode = str(lateral_mode).lower()
        self.poisson_ratio = float(poisson_ratio)
        self.max_lateral_scale_step = float(max_lateral_scale_step)
        if self.axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0/1/2, got {axis}")
        if not hasattr(molecular, "box"):
            raise NotImplementedError("UniaxialTensileLoader requires molecular.box")
        if self.lateral_mode not in ("fixed", "poisson"):
            raise ValueError(f"unsupported lateral_mode={lateral_mode}")

        self.initial_lengths = molecular.box.lengths.detach().clone().to(torch.float64)
        self.total_engineering_strain = 0.0

    def step(self, dt: float) -> float:
        dstrain = self.strain_rate * float(dt)
        scale_vec = torch.ones(3, device=self.molecular.device, dtype=torch.float64)
        scale_vec[self.axis] = 1.0 + dstrain

        lateral_axes = [i for i in range(3) if i != self.axis]
        if self.lateral_mode == "poisson":
            lat_scale = max(1.0 - self.poisson_ratio * dstrain, 1.0 - self.max_lateral_scale_step)
            for ax in lateral_axes:
                scale_vec[ax] = lat_scale

        old_H = self.molecular.box.H.detach().clone()
        frac = self.molecular.coordinates @ self.molecular.box.H_inv.to(self.molecular.coordinates).T
        new_H = torch.diag(scale_vec).to(old_H.dtype) @ old_H

        self.molecular.box.H = new_H
        coords = frac @ self.molecular.box.H.to(frac.dtype)
        coords = self.molecular.box.wrap_positions(coords)

        new_lengths = self.molecular.box.lengths.detach().clone()
        self.molecular.box_length_cpu = float(new_lengths[0].item())
        if hasattr(self.molecular, "box_length"):
            if torch.is_tensor(self.molecular.box_length):
                if self.molecular.box_length.ndim == 0:
                    self.molecular.box_length = new_lengths[0].to(
                        device=self.molecular.device, dtype=self.molecular.coordinates.dtype
                    )
                else:
                    self.molecular.box_length = new_lengths.to(
                        device=self.molecular.device, dtype=self.molecular.coordinates.dtype
                    )
            else:
                self.molecular.box_length = float(new_lengths[0].item())

        self.molecular.update_coordinates(coords)
        self.molecular.last_positions = self.molecular.coordinates.detach().clone()
        self.total_engineering_strain = (
            float(new_lengths[self.axis].item()) / float(self.initial_lengths[self.axis].item()) - 1.0
        )
        return self.total_engineering_strain

    def current_lengths(self) -> torch.Tensor:
        return self.molecular.box.lengths.detach().clone()
