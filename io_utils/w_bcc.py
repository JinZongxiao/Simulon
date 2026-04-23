from __future__ import annotations

from itertools import product
from pathlib import Path

import torch


def _orientation_matrix(label: str) -> torch.Tensor:
    label = str(label)
    if label == "100":
        mat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    elif label == "110":
        mat = [[1, 1, 0], [1, -1, 0], [0, 0, 1]]
    elif label == "111":
        mat = [[1, 1, 1], [1, -1, 0], [1, 1, -2]]
    else:
        raise ValueError(f"unsupported orientation {label}")
    return torch.tensor(mat, dtype=torch.float64)


def _unit_basis_positions(lattice_param: float, orientation: str) -> tuple[torch.Tensor, torch.Tensor]:
    a = float(lattice_param)
    H_conv = torch.eye(3, dtype=torch.float64) * a
    H_unit = _orientation_matrix(orientation) @ H_conv
    H_inv_t = torch.linalg.inv(H_unit).T

    basis_frac_conv = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        dtype=torch.float64,
    )
    grid_radius = int(_orientation_matrix(orientation).abs().sum().item()) + 2

    frac_positions = []
    seen = set()
    for n in product(range(-grid_radius, grid_radius + 1), repeat=3):
        shift = torch.tensor(n, dtype=torch.float64)
        for basis in basis_frac_conv:
            r_cart = (shift + basis) @ H_conv
            frac_new = r_cart @ H_inv_t
            if torch.all(frac_new >= -1e-8) and torch.all(frac_new < 1.0 - 1e-8):
                key = tuple(round(float(x), 8) for x in frac_new.tolist())
                if key not in seen:
                    seen.add(key)
                    frac_positions.append(frac_new)

    frac_tensor = torch.stack(frac_positions, dim=0)
    return H_unit, frac_tensor


def generate_oriented_bcc_w(
    lattice_param: float,
    orientation: str = "100",
    replicas: tuple[int, int, int] = (5, 5, 5),
) -> tuple[torch.Tensor, torch.Tensor]:
    H_unit, frac_unit = _unit_basis_positions(lattice_param, orientation)
    reps = torch.tensor(replicas, dtype=torch.int64)
    if reps.numel() != 3 or torch.any(reps <= 0):
        raise ValueError(f"replicas must be three positive integers, got {replicas}")

    H_super = torch.diag(reps.to(torch.float64)) @ H_unit
    coords = []
    for i, j, k in product(range(int(reps[0])), range(int(reps[1])), range(int(reps[2]))):
        cell_shift = torch.tensor([i, j, k], dtype=torch.float64)
        for frac in frac_unit:
            frac_super = (frac + cell_shift) / reps.to(torch.float64)
            coords.append(frac_super @ H_super)

    coords = torch.stack(coords, dim=0)
    return coords, H_super


def write_xyz(path: str | Path, coords: torch.Tensor, atom_type: str = "W", comment: str = "") -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comment = comment or f"{atom_type} generated structure"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{coords.shape[0]}\n")
        f.write(f"{comment}\n")
        for xyz in coords.tolist():
            f.write(f"{atom_type} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\n")
    return str(path)
