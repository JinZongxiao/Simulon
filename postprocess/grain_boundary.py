from __future__ import annotations

import csv
import json
from pathlib import Path

import torch


EV_PER_A2_TO_J_PER_M2 = 16.021766208


def grain_boundary_area_A2(box_vectors: torch.Tensor, in_plane_axes: tuple[int, int] = (0, 2)) -> float:
    h = box_vectors.to(torch.float64)
    area = torch.linalg.norm(torch.linalg.cross(h[in_plane_axes[0]], h[in_plane_axes[1]], dim=0))
    return float(area.item())


def grain_boundary_energy_j_m2(
    gb_energy_ev: float,
    atom_count: int,
    bulk_energy_per_atom_ev: float,
    area_A2: float,
    n_boundaries: int = 2,
) -> float:
    excess_ev = float(gb_energy_ev) - int(atom_count) * float(bulk_energy_per_atom_ev)
    denom = max(float(n_boundaries) * float(area_A2), 1e-12)
    return (excess_ev / denom) * EV_PER_A2_TO_J_PER_M2


def write_candidates_csv(path: str | Path, rows: list[dict]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "shift_x_frac",
        "shift_z_frac",
        "atom_count",
        "initial_energy_ev",
        "final_energy_ev",
        "energy_per_atom_ev",
        "energy_drop_ev",
        "final_max_force_ev_A",
        "final_mean_force_ev_A",
        "converged",
        "relaxed_structure",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return str(path)


def write_gb_energy_report(path: str | Path, report: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(path)
