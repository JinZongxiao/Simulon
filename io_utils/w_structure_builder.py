from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from io_utils.w_bcc import generate_oriented_bcc_w


SUPPORTED_KINDS = (
    "bulk",
    "surface",
    "vacancy",
    "interstitial",
    "substitution",
    "ods_w_precursor",
    "crack",
    "notch",
    "void",
    "bicrystal",
)


@dataclass
class WStructureBuildResult:
    coords: torch.Tensor
    atom_types: list[str]
    box_vectors: torch.Tensor
    summary: dict


def parse_replicas(value: str | Iterable[int]) -> tuple[int, int, int]:
    if isinstance(value, str):
        parts = [int(x.strip()) for x in value.split(",")]
    else:
        parts = [int(x) for x in value]
    if len(parts) != 3 or any(x <= 0 for x in parts):
        raise ValueError(f"replicas must be three positive integers, got {value}")
    return tuple(parts)


def parse_vector(value: str | None) -> tuple[float, float, float] | None:
    if value is None or value == "":
        return None
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"vector must have three comma-separated values, got {value}")
    return tuple(parts)


def parse_miller(value: str | Iterable[int]) -> tuple[int, int, int]:
    if isinstance(value, str):
        parts = [int(x.strip()) for x in value.split(",")]
    else:
        parts = [int(x) for x in value]
    if len(parts) != 3:
        raise ValueError(f"Miller index must have three integers, got {value}")
    return tuple(parts)


def write_xyz_with_types(path: str | Path, coords: torch.Tensor, atom_types: list[str], comment: str = "") -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if coords.shape[0] != len(atom_types):
        raise ValueError("coords and atom_types length mismatch")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{coords.shape[0]}\n")
        f.write((comment or "Simulon generated W structure") + "\n")
        for atom_type, xyz in zip(atom_types, coords.detach().cpu().tolist()):
            f.write(f"{atom_type} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}\n")
    return str(path)


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis.lower()]


def _box_lengths(box_vectors: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(box_vectors.to(torch.float64), dim=1)


def _center_from_box(box_vectors: torch.Tensor) -> torch.Tensor:
    return 0.5 * box_vectors.to(torch.float64).sum(dim=0)


def _axis_unit_vectors(box_vectors: torch.Tensor) -> torch.Tensor:
    axes = box_vectors.to(torch.float64)
    return axes / torch.linalg.norm(axes, dim=1, keepdim=True).clamp_min(1e-12)


def _project_to_box_axes(coords: torch.Tensor, box_vectors: torch.Tensor) -> torch.Tensor:
    return coords.to(torch.float64) @ _axis_unit_vectors(box_vectors).T


def _wrap_to_box(coords: torch.Tensor, box_vectors: torch.Tensor) -> torch.Tensor:
    h = box_vectors.to(torch.float64)
    h_inv = torch.linalg.inv(h)
    frac = coords.to(torch.float64) @ h_inv
    frac = frac - torch.floor(frac)
    return frac @ h


def _rotation_matrix(axis: torch.Tensor, angle_deg: float) -> torch.Tensor:
    axis = axis.to(torch.float64)
    axis = axis / torch.linalg.norm(axis).clamp_min(1e-12)
    ux, uy, uz = axis.tolist()
    angle = math.radians(float(angle_deg))
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return torch.tensor(
        [
            [c + ux * ux * one_c, ux * uy * one_c - uz * s, ux * uz * one_c + uy * s],
            [uy * ux * one_c + uz * s, c + uy * uy * one_c, uy * uz * one_c - ux * s],
            [uz * ux * one_c - uy * s, uz * uy * one_c + ux * s, c + uz * uz * one_c],
        ],
        dtype=torch.float64,
    )


def _composition(atom_types: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(atom_types).items()))


def _write_composition_csv(path: Path, atom_types: list[str]) -> str:
    comp = _composition(atom_types)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["element", "count", "fraction"])
        writer.writeheader()
        total = max(1, len(atom_types))
        for element, count in comp.items():
            writer.writerow({"element": element, "count": count, "fraction": count / total})
    return str(path)


def _pair_min_distance(coords: torch.Tensor) -> float | None:
    if coords.shape[0] < 2:
        return None
    # This is intended for build-time QA, not for million-atom production.
    if coords.shape[0] > 50_000:
        return None
    c = coords.to(torch.float64)
    best = float("inf")
    batch = 4096
    for start in range(0, c.shape[0], batch):
        block = c[start : start + batch]
        d = torch.cdist(block, c)
        row = torch.arange(block.shape[0], device=d.device)
        col = torch.arange(start, start + block.shape[0], device=d.device)
        d[row, col] = float("inf")
        best = min(best, float(d.min().item()))
    return best


def _choose_indices(count: int, n: int, rng: random.Random) -> list[int]:
    n = min(max(0, int(n)), count)
    if n == 0:
        return []
    return sorted(rng.sample(range(count), n))


def _vacancy_count(total_atoms: int, vacancy_count: int, vacancy_fraction: float) -> int:
    if vacancy_count > 0:
        return int(vacancy_count)
    if vacancy_fraction > 0.0:
        return max(1, int(round(total_atoms * vacancy_fraction)))
    return 1


def _apply_surface(
    coords: torch.Tensor,
    box_vectors: torch.Tensor,
    axis: str,
    vacuum_a: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    axis_idx = _axis_index(axis)
    new_box = box_vectors.detach().clone().to(torch.float64)
    length = torch.linalg.norm(new_box[axis_idx])
    if length <= 0:
        raise ValueError("surface axis length is zero")
    direction = new_box[axis_idx] / length
    new_box[axis_idx] = direction * (length + float(vacuum_a))
    shifted = coords.to(torch.float64) + direction.reshape(1, 3) * (0.5 * float(vacuum_a))
    return shifted, new_box, {"surface_axis": axis, "vacuum_A": float(vacuum_a)}


def _apply_vacancy(
    coords: torch.Tensor,
    atom_types: list[str],
    vacancy_count: int,
    vacancy_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, list[str], dict]:
    rng = random.Random(seed)
    n_remove = _vacancy_count(coords.shape[0], vacancy_count, vacancy_fraction)
    remove = set(_choose_indices(coords.shape[0], n_remove, rng))
    keep = [i for i in range(coords.shape[0]) if i not in remove]
    return coords[keep], [atom_types[i] for i in keep], {
        "vacancy_count": len(remove),
        "removed_indices": sorted(remove)[:100],
    }


def _apply_substitution(
    atom_types: list[str],
    element: str,
    substitution_count: int,
    substitution_fraction: float,
    seed: int,
) -> tuple[list[str], dict]:
    rng = random.Random(seed)
    if substitution_count > 0:
        n_sub = int(substitution_count)
    elif substitution_fraction > 0.0:
        n_sub = max(1, int(round(len(atom_types) * substitution_fraction)))
    else:
        n_sub = 1
    indices = _choose_indices(len(atom_types), n_sub, rng)
    new_types = list(atom_types)
    for idx in indices:
        new_types[idx] = element
    return new_types, {
        "substitution_element": element,
        "substitution_count": len(indices),
        "substituted_indices": indices[:100],
    }


def _candidate_interstitials(box_vectors: torch.Tensor, replicas: tuple[int, int, int]) -> torch.Tensor:
    reps = torch.tensor(replicas, dtype=torch.float64)
    candidates = []
    motifs = torch.tensor(
        [
            [0.25, 0.50, 0.00],
            [0.50, 0.25, 0.00],
            [0.00, 0.25, 0.50],
            [0.25, 0.00, 0.50],
            [0.50, 0.00, 0.25],
            [0.00, 0.50, 0.25],
        ],
        dtype=torch.float64,
    )
    for i in range(int(reps[0])):
        for j in range(int(reps[1])):
            for k in range(int(reps[2])):
                cell = torch.tensor([i, j, k], dtype=torch.float64)
                for motif in motifs:
                    candidates.append(((cell + motif) / reps) @ box_vectors.to(torch.float64))
    return torch.stack(candidates, dim=0)


def _apply_interstitial(
    coords: torch.Tensor,
    atom_types: list[str],
    box_vectors: torch.Tensor,
    replicas: tuple[int, int, int],
    count: int,
    element: str,
    seed: int,
) -> tuple[torch.Tensor, list[str], dict]:
    rng = random.Random(seed)
    n_add = max(1, int(count))
    candidates = _candidate_interstitials(box_vectors, replicas)
    # Prefer sites far from existing atoms.
    d = torch.cdist(candidates.to(torch.float64), coords.to(torch.float64))
    min_d = d.min(dim=1).values
    order = sorted(range(candidates.shape[0]), key=lambda i: (-float(min_d[i]), rng.random()))
    chosen = order[:n_add]
    added = candidates[chosen].to(coords.dtype)
    new_coords = torch.cat([coords, added], dim=0)
    new_types = list(atom_types) + [element] * len(chosen)
    return new_coords, new_types, {
        "interstitial_element": element,
        "interstitial_count": len(chosen),
        "interstitial_min_distance_before_A": float(min_d[chosen].min().item()) if chosen else None,
    }


def _apply_void(
    coords: torch.Tensor,
    atom_types: list[str],
    center: torch.Tensor,
    radius_a: float,
) -> tuple[torch.Tensor, list[str], dict]:
    d = torch.linalg.norm(coords.to(torch.float64) - center.reshape(1, 3), dim=1)
    keep_mask = d > float(radius_a)
    keep = keep_mask.nonzero(as_tuple=False).flatten().tolist()
    removed = int((~keep_mask).sum().item())
    return coords[keep], [atom_types[i] for i in keep], {
        "void_center_A": [float(x) for x in center.tolist()],
        "void_radius_A": float(radius_a),
        "removed_atoms": removed,
    }


def _parse_oxide_formula(formula: str) -> dict[str, int]:
    normalized = str(formula).strip().upper().replace(" ", "")
    supported = {
        "ABO": {"A": 1, "B": 1, "O": 1},
        "ABO2": {"A": 1, "B": 1, "O": 2},
        "ABO3": {"A": 1, "B": 1, "O": 3},
        "AB2O4": {"A": 1, "B": 2, "O": 4},
        "A2B2O7": {"A": 2, "B": 2, "O": 7},
    }
    if normalized not in supported:
        raise ValueError(
            f"unsupported oxide_formula={formula}; supported={sorted(supported)}"
        )
    return supported[normalized]


def _motif_for_formula(formula: str) -> list[tuple[str, tuple[float, float, float]]]:
    normalized = str(formula).strip().upper().replace(" ", "")
    if normalized == "ABO3":
        return [
            ("A", (0.0, 0.0, 0.0)),
            ("B", (0.5, 0.5, 0.5)),
            ("O", (0.5, 0.5, 0.0)),
            ("O", (0.5, 0.0, 0.5)),
            ("O", (0.0, 0.5, 0.5)),
        ]
    # Generic pseudo-oxide motif. This preserves stoichiometry but is not a
    # crystallographic oxide model; it is intended only as an ODS-W geometry seed.
    stoich = _parse_oxide_formula(normalized)
    fractions = [
        (0.00, 0.00, 0.00),
        (0.50, 0.50, 0.50),
        (0.50, 0.00, 0.00),
        (0.00, 0.50, 0.00),
        (0.00, 0.00, 0.50),
        (0.50, 0.50, 0.00),
        (0.50, 0.00, 0.50),
        (0.00, 0.50, 0.50),
        (0.25, 0.25, 0.25),
        (0.75, 0.75, 0.75),
        (0.25, 0.75, 0.25),
    ]
    motif: list[tuple[str, tuple[float, float, float]]] = []
    cursor = 0
    for label in ("A", "B", "O"):
        for _ in range(int(stoich[label])):
            motif.append((label, fractions[cursor % len(fractions)]))
            cursor += 1
    return motif


def _apply_ods_w_precursor(
    coords: torch.Tensor,
    atom_types: list[str],
    box_vectors: torch.Tensor,
    center: torch.Tensor,
    radius_a: float,
    a_element: str,
    b_element: str,
    oxide_formula: str,
    oxide_lattice_param_a: float,
    interface_clearance_a: float,
) -> tuple[torch.Tensor, list[str], dict]:
    if radius_a <= 0.0:
        raise ValueError("ods particle radius must be positive")
    if oxide_lattice_param_a <= 0.0:
        raise ValueError("oxide lattice parameter must be positive")
    center = center.to(torch.float64)
    d = torch.linalg.norm(coords.to(torch.float64) - center.reshape(1, 3), dim=1)
    remove_radius = float(radius_a) + max(0.0, float(interface_clearance_a))
    keep_mask = d > remove_radius
    keep = keep_mask.nonzero(as_tuple=False).flatten().tolist()
    removed_w = int((~keep_mask).sum().item())

    motif = _motif_for_formula(oxide_formula)
    half = int(math.ceil(float(radius_a) / float(oxide_lattice_param_a))) + 1
    inserted_coords: list[torch.Tensor] = []
    inserted_types: list[str] = []
    label_to_element = {"A": str(a_element), "B": str(b_element), "O": "O"}
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            for k in range(-half, half + 1):
                cell = torch.tensor([i, j, k], dtype=torch.float64) * float(oxide_lattice_param_a)
                for label, frac in motif:
                    offset = cell + torch.tensor(frac, dtype=torch.float64) * float(oxide_lattice_param_a)
                    pos = center + offset
                    if torch.linalg.norm(pos - center) <= float(radius_a):
                        inserted_coords.append(pos)
                        inserted_types.append(label_to_element[label])

    if not inserted_coords:
        raise ValueError("ODS particle contains no atoms; increase radius or reduce oxide lattice parameter")

    oxide_coords = torch.stack(inserted_coords, dim=0)
    new_coords = torch.cat([coords[keep].to(torch.float64), oxide_coords], dim=0)
    new_types = [atom_types[i] for i in keep] + inserted_types
    oxide_comp = _composition(inserted_types)
    operations = {
        "ods_precursor": True,
        "physics_ready": False,
        "requires_multielement_potential": True,
        "particle_model": "spherical_pseudo_oxide_geometry",
        "a_element": str(a_element),
        "b_element": str(b_element),
        "oxide_formula": str(oxide_formula).upper(),
        "oxide_stoichiometry_labels": _parse_oxide_formula(oxide_formula),
        "oxide_composition": oxide_comp,
        "oxide_atom_count": int(len(inserted_types)),
        "matrix_w_atoms_after_removal": int(len(keep)),
        "removed_w_atoms": removed_w,
        "particle_center_A": [float(x) for x in center.tolist()],
        "particle_radius_A": float(radius_a),
        "oxide_lattice_param_A": float(oxide_lattice_param_a),
        "interface_clearance_A": float(interface_clearance_a),
        "estimated_volume_fraction": float(
            (4.0 / 3.0) * math.pi * float(radius_a) ** 3
            / max(1e-12, abs(float(torch.linalg.det(box_vectors.to(torch.float64)).item())))
        ),
        "note": (
            "Geometry precursor only. Do not run physical ODS-W MD unless a "
            "validated W-A-B-O multi-element potential is supplied."
        ),
    }
    return new_coords, new_types, operations


def _apply_crack(
    coords: torch.Tensor,
    atom_types: list[str],
    box_vectors: torch.Tensor,
    half_length_a: float,
    opening_a: float,
    center: torch.Tensor | None,
    length_axis: str,
    opening_axis: str,
) -> tuple[torch.Tensor, list[str], dict]:
    center = center if center is not None else _center_from_box(box_vectors)
    length_idx = _axis_index(length_axis)
    opening_idx = _axis_index(opening_axis)
    if length_idx == opening_idx:
        raise ValueError("crack length axis and opening axis must differ")
    coord_axis = _project_to_box_axes(coords, box_vectors)
    center_axis = _project_to_box_axes(center.reshape(1, 3), box_vectors).reshape(3)
    rel = coord_axis - center_axis.reshape(1, 3)
    remove_mask = (
        (torch.abs(rel[:, length_idx]) <= float(half_length_a))
        & (torch.abs(rel[:, opening_idx]) <= 0.5 * float(opening_a))
    )
    keep = (~remove_mask).nonzero(as_tuple=False).flatten().tolist()
    removed = int(remove_mask.sum().item())
    return coords[keep], [atom_types[i] for i in keep], {
        "crack_center_A": [float(x) for x in center.tolist()],
        "crack_half_length_A": float(half_length_a),
        "crack_opening_A": float(opening_a),
        "crack_length_axis": length_axis,
        "crack_opening_axis": opening_axis,
        "removed_atoms": removed,
    }


def _apply_notch(
    coords: torch.Tensor,
    atom_types: list[str],
    box_vectors: torch.Tensor,
    radius_a: float,
    depth_a: float,
    surface_axis: str,
    surface_side: str,
) -> tuple[torch.Tensor, list[str], dict]:
    axis_idx = _axis_index(surface_axis)
    lengths = _box_lengths(box_vectors)
    unit_axes = _axis_unit_vectors(box_vectors)
    center_axis = _project_to_box_axes(_center_from_box(box_vectors).reshape(1, 3), box_vectors).reshape(3)
    surface_side = surface_side.lower()
    if surface_side not in ("min", "max"):
        raise ValueError("surface_side must be min or max")
    if surface_side == "min":
        center_axis[axis_idx] = 0.0
    else:
        center_axis[axis_idx] = float(lengths[axis_idx])
    coord_axis = _project_to_box_axes(coords, box_vectors)
    center = center_axis @ unit_axes
    if surface_side == "min":
        inside_depth = coord_axis[:, axis_idx] <= float(depth_a)
    else:
        inside_depth = coord_axis[:, axis_idx] >= float(lengths[axis_idx] - depth_a)
    radial_axes = [i for i in range(3) if i != axis_idx]
    rel = coord_axis[:, radial_axes] - center_axis[radial_axes].reshape(1, 2)
    radial = torch.linalg.norm(rel, dim=1)
    remove_mask = inside_depth & (radial <= float(radius_a))
    keep = (~remove_mask).nonzero(as_tuple=False).flatten().tolist()
    removed = int(remove_mask.sum().item())
    return coords[keep], [atom_types[i] for i in keep], {
        "notch_surface_axis": surface_axis,
        "notch_surface_side": surface_side,
        "notch_radius_A": float(radius_a),
        "notch_depth_A": float(depth_a),
        "notch_center_A": [float(x) for x in center.tolist()],
        "removed_atoms": removed,
    }


def _remove_close_pairs_near_axis_plane(
    coords: torch.Tensor,
    atom_types: list[str],
    box_vectors: torch.Tensor,
    plane_axis: int,
    plane_position: float,
    cutoff_a: float,
    search_width_a: float,
) -> tuple[torch.Tensor, list[str], int]:
    if cutoff_a <= 0.0:
        return coords, atom_types, 0
    coord_axis = _project_to_box_axes(coords, box_vectors)
    near = torch.abs(coord_axis[:, plane_axis] - float(plane_position)) <= float(search_width_a)
    near_idx = near.nonzero(as_tuple=False).flatten()
    if near_idx.numel() < 2:
        return coords, atom_types, 0
    near_coords = coords[near_idx].to(torch.float64)
    d = torch.cdist(near_coords, near_coords)
    pair_i, pair_j = torch.where(torch.triu(d < float(cutoff_a), diagonal=1))
    remove_local: set[int] = set()
    for i, j in zip(pair_i.detach().cpu().tolist(), pair_j.detach().cpu().tolist()):
        if i in remove_local or j in remove_local:
            continue
        # Remove the atom closer to the nominal GB plane.
        ai = float(abs(coord_axis[int(near_idx[i]), plane_axis] - plane_position))
        aj = float(abs(coord_axis[int(near_idx[j]), plane_axis] - plane_position))
        remove_local.add(i if ai <= aj else j)
    remove_global = {int(near_idx[i]) for i in remove_local}
    keep = [i for i in range(coords.shape[0]) if i not in remove_global]
    return coords[keep], [atom_types[i] for i in keep], len(remove_global)


def _sigma_001_stgb(h: int, k: int) -> int:
    raw = h * h + k * k
    if h % 2 != 0 and k % 2 != 0:
        return raw // 2
    return raw


def _enumerate_bcc_grain(
    lattice_param: float,
    rotation: torch.Tensor,
    box_vectors: torch.Tensor,
    y_min_frac: float,
    y_max_frac: float,
    margin_cells: int,
    shift: torch.Tensor | None = None,
) -> torch.Tensor:
    lengths = _box_lengths(box_vectors)
    max_extent = int(math.ceil(float(lengths.max().item()) / float(lattice_param))) + int(margin_cells)
    basis = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64)
    h_inv = torch.linalg.inv(box_vectors.to(torch.float64))
    shift = torch.zeros(3, dtype=torch.float64) if shift is None else shift.to(torch.float64)
    coords = []
    for i in range(-max_extent, max_extent + 1):
        for j in range(-max_extent, max_extent + 1):
            for k in range(-max_extent, max_extent + 1):
                cell = torch.tensor([i, j, k], dtype=torch.float64)
                for b in basis:
                    crystal = float(lattice_param) * (cell + b)
                    cart = crystal @ rotation.T + shift
                    frac = cart @ h_inv
                    if (
                        -1e-8 <= float(frac[0]) < 1.0 - 1e-8
                        and y_min_frac - 1e-8 <= float(frac[1]) < y_max_frac - 1e-8
                        and -1e-8 <= float(frac[2]) < 1.0 - 1e-8
                    ):
                        coords.append(cart)
    if not coords:
        return torch.empty((0, 3), dtype=torch.float64)
    return torch.stack(coords, dim=0)


def _deduplicate_by_fraction(coords: torch.Tensor, box_vectors: torch.Tensor, tol: float = 1e-7) -> torch.Tensor:
    if coords.shape[0] == 0:
        return coords
    h_inv = torch.linalg.inv(box_vectors.to(torch.float64))
    frac = coords.to(torch.float64) @ h_inv
    frac = frac - torch.floor(frac)
    seen = set()
    keep = []
    for idx, row in enumerate(frac.tolist()):
        key = tuple(int(round(x / tol)) for x in row)
        if key not in seen:
            seen.add(key)
            keep.append(idx)
    return (frac[keep] @ box_vectors.to(torch.float64))


def _build_001_symmetric_tilt_bicrystal(
    lattice_param: float,
    replicas: tuple[int, int, int],
    gb_plane: tuple[int, int, int],
    overlap_cutoff_a: float,
    gb_search_width_a: float,
    grain_b_translation_frac: tuple[float, float, float] | None = None,
) -> tuple[torch.Tensor, list[str], torch.Tensor, dict]:
    h, k, l = parse_miller(gb_plane)
    if l != 0 or h <= 0 or k <= 0:
        raise ValueError("current strict bicrystal builder supports only positive (h,k,0)[001] STGB")
    g = math.gcd(h, k)
    if g != 1:
        raise ValueError("gb_plane h and k must be coprime for primitive CSL construction")
    reps = parse_replicas(replicas)
    norm = math.sqrt(h * h + k * k)
    ex = torch.tensor([k / norm, -h / norm, 0.0], dtype=torch.float64)
    ey = torch.tensor([h / norm, k / norm, 0.0], dtype=torch.float64)
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

    # CSL-periodic cell: x is in the GB plane, y is the GB normal, z is [001].
    lx = float(lattice_param) * norm * int(reps[0])
    ly = float(lattice_param) * norm * int(reps[1])
    lz = float(lattice_param) * int(reps[2])
    box_vectors = torch.stack([lx * ex, ly * ey, lz * ez], dim=0)

    p_global = torch.stack([ex, ey, ez], dim=1)
    p_a = torch.stack([ex, ey, ez], dim=1)
    r_a = p_global @ p_a.T
    # The second grain must be a proper rotation, not a mirror reflection.
    # For a (h k 0)[001] symmetric tilt boundary, theta = 2 atan(k / h).
    theta = math.degrees(2.0 * math.atan(k / h))
    r_b = _rotation_matrix(ez, theta)

    margin = max(h, k, 2) + max(reps)
    grain_a = _enumerate_bcc_grain(
        lattice_param, r_a, box_vectors, y_min_frac=0.0, y_max_frac=0.5, margin_cells=margin
    )
    translation_frac = torch.tensor(
        grain_b_translation_frac or (0.0, 0.0, 0.0),
        dtype=torch.float64,
    )
    translation_cart = translation_frac @ box_vectors
    grain_b = _enumerate_bcc_grain(
        lattice_param, r_b, box_vectors, y_min_frac=0.5, y_max_frac=1.0, margin_cells=margin
    )
    if grain_b.shape[0] > 0:
        grain_b = grain_b + translation_cart.reshape(1, 3)
        grain_b = _wrap_to_box(grain_b, box_vectors)
    coords = torch.cat([grain_a, grain_b], dim=0)
    coords = _deduplicate_by_fraction(coords, box_vectors)
    atom_types = ["W"] * int(coords.shape[0])

    gb_position = 0.5 * ly
    coords, atom_types, overlap_removed_mid = _remove_close_pairs_near_axis_plane(
        coords,
        atom_types,
        box_vectors,
        plane_axis=1,
        plane_position=gb_position,
        cutoff_a=overlap_cutoff_a,
        search_width_a=max(float(gb_search_width_a), float(overlap_cutoff_a) * 2.0),
    )
    coords, atom_types, overlap_removed_periodic = _remove_close_pairs_near_axis_plane(
        coords,
        atom_types,
        box_vectors,
        plane_axis=1,
        plane_position=0.0,
        cutoff_a=overlap_cutoff_a,
        search_width_a=max(float(gb_search_width_a), float(overlap_cutoff_a) * 2.0),
    )

    sigma = _sigma_001_stgb(h, k)
    coord_axis = _project_to_box_axes(coords, box_vectors)
    grain_a_count = int((coord_axis[:, 1] < gb_position).sum().item())
    grain_b_count = int(coords.shape[0] - grain_a_count)
    operations = {
        "bicrystal_type": "csl_001_symmetric_tilt",
        "gb_plane_hkl": [int(h), int(k), 0],
        "tilt_axis_uvw": [0, 0, 1],
        "sigma": int(sigma),
        "misorientation_deg": float(theta),
        "gb_axis": "y",
        "gb_position_A": float(gb_position),
        "periodic_gb_position_A": 0.0,
        "overlap_cutoff_A": float(overlap_cutoff_a),
        "gb_search_width_A": float(gb_search_width_a),
        "overlap_removed_atoms": int(overlap_removed_mid + overlap_removed_periodic),
        "grain_b_translation_frac": [float(x) for x in translation_frac.tolist()],
        "grain_b_translation_A": [float(x) for x in translation_cart.tolist()],
        "grain_a_atoms": grain_a_count,
        "grain_b_atoms": grain_b_count,
        "csl_exact": True,
        "note": "CSL-periodic BCC [001] symmetric tilt grain-boundary seed; relax and search rigid-body translations before production physics.",
    }
    return coords, atom_types, box_vectors, operations


def build_w_structure(
    kind: str,
    orientation: str = "100",
    replicas: tuple[int, int, int] = (6, 6, 6),
    lattice_param: float = 3.1652,
    seed: int = 1234,
    surface_axis: str = "z",
    vacuum_a: float = 20.0,
    vacancy_count: int = 0,
    vacancy_fraction: float = 0.0,
    interstitial_count: int = 1,
    interstitial_element: str = "W",
    substitution_count: int = 0,
    substitution_fraction: float = 0.0,
    substitution_element: str = "Re",
    ods_a_element: str = "Zr",
    ods_b_element: str = "Y",
    ods_oxide_formula: str = "ABO3",
    ods_particle_radius_a: float = 8.0,
    ods_oxide_lattice_param_a: float = 4.5,
    ods_interface_clearance_a: float = 1.0,
    void_radius_a: float = 5.0,
    defect_center: tuple[float, float, float] | None = None,
    crack_half_length_a: float = 15.0,
    crack_opening_a: float = 2.0,
    crack_length_axis: str = "x",
    crack_opening_axis: str = "y",
    notch_radius_a: float = 6.0,
    notch_depth_a: float = 6.0,
    notch_surface_side: str = "min",
    gb_plane: tuple[int, int, int] = (3, 1, 0),
    gb_overlap_cutoff_a: float = 2.0,
    gb_search_width_a: float = 6.0,
    gb_translation_frac: tuple[float, float, float] | None = None,
) -> WStructureBuildResult:
    kind = kind.lower()
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"unsupported kind={kind}; supported={SUPPORTED_KINDS}")
    replicas = parse_replicas(replicas)
    if kind == "bicrystal":
        coords, atom_types, box_vectors, operations = _build_001_symmetric_tilt_bicrystal(
            lattice_param=lattice_param,
            replicas=replicas,
            gb_plane=gb_plane,
            overlap_cutoff_a=gb_overlap_cutoff_a,
            gb_search_width_a=gb_search_width_a,
            grain_b_translation_frac=gb_translation_frac,
        )
        initial_atoms = int(coords.shape[0] + operations["overlap_removed_atoms"])
    else:
        coords, box_vectors = generate_oriented_bcc_w(
            lattice_param=lattice_param,
            orientation=orientation,
            replicas=replicas,
        )
        coords = coords.to(torch.float64)
        box_vectors = box_vectors.to(torch.float64)
        atom_types = ["W"] * int(coords.shape[0])
        initial_atoms = int(coords.shape[0])
        operations: dict = {}

    center = torch.tensor(defect_center, dtype=torch.float64) if defect_center is not None else None
    if kind == "surface":
        coords, box_vectors, operations = _apply_surface(coords, box_vectors, surface_axis, vacuum_a)
    elif kind == "vacancy":
        coords, atom_types, operations = _apply_vacancy(
            coords, atom_types, vacancy_count, vacancy_fraction, seed
        )
    elif kind == "interstitial":
        coords, atom_types, operations = _apply_interstitial(
            coords, atom_types, box_vectors, replicas, interstitial_count, interstitial_element, seed
        )
    elif kind == "substitution":
        atom_types, operations = _apply_substitution(
            atom_types, substitution_element, substitution_count, substitution_fraction, seed
        )
    elif kind == "ods_w_precursor":
        coords, atom_types, operations = _apply_ods_w_precursor(
            coords,
            atom_types,
            box_vectors,
            center if center is not None else _center_from_box(box_vectors),
            ods_particle_radius_a,
            ods_a_element,
            ods_b_element,
            ods_oxide_formula,
            ods_oxide_lattice_param_a,
            ods_interface_clearance_a,
        )
    elif kind == "void":
        coords, atom_types, operations = _apply_void(
            coords, atom_types, center if center is not None else _center_from_box(box_vectors), void_radius_a
        )
    elif kind == "crack":
        coords, atom_types, operations = _apply_crack(
            coords,
            atom_types,
            box_vectors,
            crack_half_length_a,
            crack_opening_a,
            center,
            crack_length_axis,
            crack_opening_axis,
        )
    elif kind == "notch":
        coords, atom_types, operations = _apply_notch(
            coords,
            atom_types,
            box_vectors,
            notch_radius_a,
            notch_depth_a,
            surface_axis,
            notch_surface_side,
        )

    min_distance = _pair_min_distance(coords)
    lengths = _box_lengths(box_vectors)
    summary = {
        "builder": "WStructureBuilder",
        "kind": kind,
        "orientation": orientation,
        "replicas": list(replicas),
        "lattice_param_A": float(lattice_param),
        "seed": int(seed),
        "initial_atom_count": initial_atoms,
        "final_atom_count": int(coords.shape[0]),
        "atom_count_delta": int(coords.shape[0]) - initial_atoms,
        "composition": _composition(atom_types),
        "box_lengths_A": [float(x) for x in lengths.tolist()],
        "box_vectors_A": [[float(x) for x in row] for row in box_vectors.tolist()],
        "min_distance_A": min_distance,
        "operations": operations,
        "notes": [
            "Geometry builder only; physical validity still requires relaxation with a suitable potential.",
            "ODS-W precursor output is geometry-only unless a validated W-A-B-O multi-element potential is provided.",
            "Bicrystal output is an exact CSL-periodic [001] symmetric tilt seed for cubic BCC W.",
            "Production grain-boundary physics still requires rigid-body translation search and relaxation.",
            "Dislocation builders are intentionally deferred to a dedicated implementation.",
        ],
    }
    return WStructureBuildResult(coords=coords, atom_types=atom_types, box_vectors=box_vectors, summary=summary)


def write_preview_png(path: str | Path, coords: torch.Tensor, atom_types: list[str], title: str = "") -> str | None:
    path = Path(path)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    coords_cpu = coords.detach().cpu()
    fig, ax = plt.subplots(figsize=(6, 5))
    elements = sorted(set(atom_types))
    colors = {
        "W": "#4c566a",
        "Re": "#5e81ac",
        "O": "#bf616a",
        "Y": "#a3be8c",
        "Er": "#b48ead",
        "Zr": "#d08770",
        "Ti": "#ebcb8b",
        "Hf": "#88c0d0",
    }
    for element in elements:
        idx = [i for i, t in enumerate(atom_types) if t == element]
        pts = coords_cpu[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=5,
            alpha=0.75,
            label=element,
            c=colors.get(element, None),
            linewidths=0,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (A)")
    ax.set_ylabel("y (A)")
    ax.set_title(title or "Generated W structure")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _percentile(values: torch.Tensor, q: float) -> float | None:
    if values.numel() == 0:
        return None
    return float(torch.quantile(values.to(torch.float64), float(q)).item())


def _ods_interface_sanity(
    output_dir: Path,
    coords: torch.Tensor,
    atom_types: list[str],
    summary: dict,
) -> dict:
    if summary.get("kind") != "ods_w_precursor":
        return {"available": False, "reason": "not an ODS-W precursor structure"}
    if coords.shape[0] > 50_000:
        return {
            "available": False,
            "reason": "structure is too large for dense build-time interface distance analysis",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    w_idx = [i for i, t in enumerate(atom_types) if t == "W"]
    oxide_idx = [i for i, t in enumerate(atom_types) if t != "W"]
    if not w_idx or not oxide_idx:
        return {
            "available": False,
            "reason": "missing W matrix atoms or oxide precursor atoms",
        }

    w_coords = coords[w_idx].to(torch.float64)
    oxide_coords = coords[oxide_idx].to(torch.float64)
    oxide_types = [atom_types[i] for i in oxide_idx]
    d = torch.cdist(oxide_coords, w_coords)
    oxide_nearest_w = d.min(dim=1).values
    w_nearest_oxide = d.min(dim=0).values
    interface_cutoff = 3.5
    too_close_cutoff = 1.2
    caution_cutoff = 1.6
    rows = []
    by_element: dict[str, list[float]] = {}
    for element, dist in zip(oxide_types, oxide_nearest_w.detach().cpu().tolist()):
        by_element.setdefault(element, []).append(float(dist))
    for element in sorted(by_element):
        vals = torch.tensor(by_element[element], dtype=torch.float64)
        rows.append(
            {
                "element": element,
                "oxide_atom_count": int(vals.numel()),
                "min_nearest_w_A": float(vals.min().item()),
                "p05_nearest_w_A": _percentile(vals, 0.05),
                "p50_nearest_w_A": _percentile(vals, 0.50),
                "mean_nearest_w_A": float(vals.mean().item()),
                "max_nearest_w_A": float(vals.max().item()),
                "interface_atoms_within_3p5_A": int((vals <= interface_cutoff).sum().item()),
                "too_close_atoms_lt_1p2_A": int((vals < too_close_cutoff).sum().item()),
                "caution_atoms_lt_1p6_A": int((vals < caution_cutoff).sum().item()),
            }
        )

    csv_path = output_dir / "interface_sanity.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "element",
            "oxide_atom_count",
            "min_nearest_w_A",
            "p05_nearest_w_A",
            "p50_nearest_w_A",
            "mean_nearest_w_A",
            "max_nearest_w_A",
            "interface_atoms_within_3p5_A",
            "too_close_atoms_lt_1p2_A",
            "caution_atoms_lt_1p6_A",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output_dir / "interface_distance_hist.png"
    plot_written = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.5))
        for element in sorted(by_element):
            ax.hist(by_element[element], bins=24, alpha=0.55, label=f"{element}-to-W")
        ax.axvline(too_close_cutoff, color="#bf616a", linestyle="--", linewidth=1.2, label="too close 1.2 A")
        ax.axvline(caution_cutoff, color="#d08770", linestyle="--", linewidth=1.2, label="caution 1.6 A")
        ax.set_xlabel("nearest W distance (A)")
        ax.set_ylabel("oxide atom count")
        ax.set_title("ODS-W interface nearest-distance sanity")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        plot_written = str(plot_path)
    except Exception:
        plot_written = None

    too_close_atoms = int((oxide_nearest_w < too_close_cutoff).sum().item())
    caution_atoms = int((oxide_nearest_w < caution_cutoff).sum().item())
    interface_oxide_atoms = int((oxide_nearest_w <= interface_cutoff).sum().item())
    interface_w_atoms = int((w_nearest_oxide <= interface_cutoff).sum().item())
    pass_basic = too_close_atoms == 0 and interface_oxide_atoms > 0 and interface_w_atoms > 0
    notes = [
        "Distance checks are geometry-only; they do not validate energetics or thermodynamic stability.",
        "The current ODS-W precursor particle is a pseudo-oxide geometry seed, not a fully validated oxide crystal.",
    ]
    if too_close_atoms:
        notes.append("Some oxide atoms are closer than 1.2 A to W; increase interface clearance or adjust particle geometry.")
    elif caution_atoms:
        notes.append("Some oxide atoms are closer than 1.6 A to W; inspect the interface before using this geometry.")
    else:
        notes.append("No oxide-W contacts below 1.6 A were detected.")

    return {
        "available": True,
        "pass": bool(pass_basic),
        "csv": str(csv_path),
        "plot": plot_written,
        "interface_cutoff_A": interface_cutoff,
        "too_close_cutoff_A": too_close_cutoff,
        "caution_cutoff_A": caution_cutoff,
        "oxide_atom_count": int(len(oxide_idx)),
        "w_atom_count": int(len(w_idx)),
        "interface_oxide_atoms_within_3p5_A": interface_oxide_atoms,
        "interface_w_atoms_within_3p5_A": interface_w_atoms,
        "min_oxide_to_w_distance_A": float(oxide_nearest_w.min().item()),
        "mean_oxide_to_w_distance_A": float(oxide_nearest_w.mean().item()),
        "p05_oxide_to_w_distance_A": _percentile(oxide_nearest_w, 0.05),
        "p50_oxide_to_w_distance_A": _percentile(oxide_nearest_w, 0.50),
        "min_w_to_oxide_distance_A": float(w_nearest_oxide.min().item()),
        "mean_w_to_oxide_distance_A": float(w_nearest_oxide.mean().item()),
        "too_close_oxide_atoms_lt_1p2_A": too_close_atoms,
        "caution_oxide_atoms_lt_1p6_A": caution_atoms,
        "by_element": rows,
        "notes": notes,
    }


def _write_ods_report(path: Path, summary: dict) -> str | None:
    if summary.get("kind") != "ods_w_precursor":
        return None
    ops = summary.get("operations", {})
    sanity = summary.get("interface_sanity", {})
    comp = summary.get("composition", {})
    lines = [
        "# ODS-W Precursor Builder Report",
        "",
        "## 结构目的",
        "这个结构是 ODS-W 几何前驱：在 BCC W 基体中挖出一个球形区域，并填入 A-B-O 富集颗粒。",
        "它用于准备后续结构、界面和势函数工作，不代表已经完成物理弛豫或可用于直接 MD 结论。",
        "",
        "## 体系信息",
        f"- Kind: `{summary.get('kind')}`",
        f"- Orientation: `{summary.get('orientation')}`",
        f"- Replicas: `{summary.get('replicas')}`",
        f"- Lattice parameter: `{summary.get('lattice_param_A')}` A",
        f"- Final atom count: `{summary.get('final_atom_count')}`",
        f"- Composition: `{comp}`",
        f"- Box lengths: `{summary.get('box_lengths_A')}` A",
        f"- Global minimum distance: `{summary.get('min_distance_A')}` A",
        "",
        "## ODS 颗粒定义",
        f"- A element: `{ops.get('a_element')}`",
        f"- B element: `{ops.get('b_element')}`",
        f"- Oxide formula template: `{ops.get('oxide_formula')}`",
        f"- Particle model: `{ops.get('particle_model')}`",
        f"- Particle radius: `{ops.get('particle_radius_A')}` A",
        f"- Oxide pseudo lattice parameter: `{ops.get('oxide_lattice_param_A')}` A",
        f"- Interface clearance: `{ops.get('interface_clearance_A')}` A",
        f"- Removed W atoms: `{ops.get('removed_w_atoms')}`",
        f"- Inserted oxide atoms: `{ops.get('oxide_atom_count')}`",
        f"- Estimated oxide volume fraction: `{ops.get('estimated_volume_fraction')}`",
        "",
        "## 界面 sanity checks",
    ]
    if sanity.get("available"):
        lines += [
            f"- Sanity pass: `{sanity.get('pass')}`",
            f"- Min oxide-to-W distance: `{sanity.get('min_oxide_to_w_distance_A')}` A",
            f"- Mean oxide-to-W distance: `{sanity.get('mean_oxide_to_w_distance_A')}` A",
            f"- Oxide atoms within 3.5 A of W: `{sanity.get('interface_oxide_atoms_within_3p5_A')}`",
            f"- W atoms within 3.5 A of oxide: `{sanity.get('interface_w_atoms_within_3p5_A')}`",
            f"- Too-close oxide atoms (<1.2 A): `{sanity.get('too_close_oxide_atoms_lt_1p2_A')}`",
            f"- Caution oxide atoms (<1.6 A): `{sanity.get('caution_oxide_atoms_lt_1p6_A')}`",
            f"- Interface CSV: `{sanity.get('csv')}`",
            f"- Interface histogram: `{sanity.get('plot')}`",
        ]
        lines.append("")
        lines.append("Element-resolved nearest W distances:")
        for row in sanity.get("by_element", []):
            lines.append(
                f"- `{row['element']}`: count={row['oxide_atom_count']}, "
                f"min={row['min_nearest_w_A']:.4f} A, "
                f"median={row['p50_nearest_w_A']:.4f} A, "
                f"mean={row['mean_nearest_w_A']:.4f} A"
            )
    else:
        lines.append(f"- Interface sanity unavailable: `{sanity.get('reason')}`")
    lines += [
        "",
        "## 物理限制",
        "- 当前结构没有使用 W-A-B-O 多元素势函数弛豫。",
        "- `physics_ready=false` 是有意保守设置，表示不能直接用于 ODS-W 力学结论。",
        "- 若要做拉伸、压痕、裂纹或 DBTT，必须先找到并验证覆盖 W/A/B/O 的多元素势函数，或者训练 ML potential。",
        "- 当前 pseudo-oxide motif 主要保证几何、化学计量和界面距离可检查，不等价于真实氧化物晶体结构。",
        "",
        "## 建议下一步",
        "1. 检查 `preview.png` 和 `interface_distance_hist.png`，确认颗粒位置和界面距离没有明显异常。",
        "2. 根据目标材料确定真实氧化物结构和取向关系，例如 pyrochlore 或 perovskite。",
        "3. 建立 W/oxide interface validation set，用 DFT 或可靠多元素势函数弛豫。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_build_outputs(result: WStructureBuildResult, output_dir: str | Path, case_name: str | None = None) -> dict:
    output_dir = Path(output_dir)
    if case_name:
        output_dir = output_dir / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = output_dir / "structure.xyz"
    summary_path = output_dir / "summary.json"
    composition_path = output_dir / "composition.csv"
    preview_path = output_dir / "preview.png"
    report_path = output_dir / "report.md"

    write_xyz_with_types(
        xyz_path,
        result.coords,
        result.atom_types,
        comment=(
            f"Simulon WStructureBuilder kind={result.summary['kind']} "
            f"orientation={result.summary['orientation']} atoms={result.summary['final_atom_count']}"
        ),
    )
    composition_csv = _write_composition_csv(composition_path, result.atom_types)
    preview = write_preview_png(preview_path, result.coords, result.atom_types, title=f"W {result.summary['kind']}")
    summary = dict(result.summary)
    interface_sanity = _ods_interface_sanity(output_dir, result.coords, result.atom_types, summary)
    if interface_sanity.get("available"):
        summary["interface_sanity"] = interface_sanity
    elif summary.get("kind") == "ods_w_precursor":
        summary["interface_sanity"] = interface_sanity
    summary.update(
        {
            "output_dir": str(output_dir),
            "structure": str(xyz_path),
            "summary": str(summary_path),
            "composition_csv": composition_csv,
            "preview": preview,
        }
    )
    report = _write_ods_report(report_path, summary)
    if report:
        summary["report"] = report
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_eam_model_for_coords(
    structure_path: str | Path,
    box_vectors: torch.Tensor,
    eam_path: str | Path,
    skin_thickness: float,
):
    from core.force.eam_force_cu import EAMForceCUDAExt as EAMForce
    from core.integrator.integrator import VerletIntegrator
    from core.md_model import BaseModel, SumBackboneInterface
    from io_utils.eam_parser import EAMParser
    from io_utils.reader import AtomFileReader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = EAMParser(filepath=str(eam_path), device=device)
    mol = AtomFileReader(
        filename=str(structure_path),
        box_length=float(torch.linalg.norm(box_vectors[0]).item()),
        cutoff=parser.cutoff,
        device=device,
        skin_thickness=float(skin_thickness),
        is_mlp=True,
        box_vectors=box_vectors,
    )
    ff = EAMForce(parser, mol)
    sb = SumBackboneInterface([ff], mol)
    integ = VerletIntegrator(mol, dt=0.001, ensemble="NVE")
    return BaseModel(sb, integ, mol), mol, device


def relax_structure_steepest_descent(
    structure_path: str | Path,
    box_vectors: torch.Tensor,
    atom_types: list[str],
    eam_path: str | Path,
    output_dir: str | Path,
    max_steps: int = 500,
    step_size: float = 0.01,
    force_threshold: float = 0.05,
    print_interval: int = 50,
    max_backtracks: int = 10,
    skin_thickness: float = 1.0,
    method: str = "sd",
) -> dict:
    output_dir = Path(output_dir)
    model, mol, device = _build_eam_model_for_coords(structure_path, box_vectors, eam_path, skin_thickness)
    csv_path = output_dir / "relaxation.csv"
    relaxed_path = output_dir / "relaxed_structure.xyz"
    summary_path = output_dir / "relax_summary.json"

    rows = []
    converged = False
    stop_reason = "max_steps_reached"
    coords = mol.coordinates.detach().clone()

    def evaluate():
        out = model.sum_bone()
        forces = out["forces"]
        force_norm = torch.linalg.norm(forces, dim=1)
        return out, forces, float(force_norm.max().item()), float(force_norm.mean().item())

    method = str(method).lower()
    if method not in ("sd", "fire"):
        raise ValueError(f"unsupported relaxation method={method}; expected 'sd' or 'fire'")

    with torch.no_grad():
        out, forces, max_force, mean_force = evaluate()
        initial_energy = float(out["energy"].item())
        rows.append(
            {
                "step": 0,
                "energy_ev": initial_energy,
                "max_force_ev_A": max_force,
                "mean_force_ev_A": mean_force,
                "accepted_step_A": 0.0,
            }
        )
        current_energy = initial_energy

        if method == "sd":
            for step in range(1, int(max_steps) + 1):
                out, forces, max_force, mean_force = evaluate()
                if max_force <= float(force_threshold):
                    converged = True
                    stop_reason = "force_threshold_reached"
                    current_energy = float(out["energy"].item())
                    break

                direction = forces / torch.linalg.norm(forces, dim=1, keepdim=True).clamp_min(1e-12)
                coords0 = mol.coordinates.detach().clone()
                base_energy = float(out["energy"].item())
                trial_step = float(step_size)
                accepted = False
                accepted_energy = base_energy

                for _ in range(int(max_backtracks)):
                    trial = coords0 + trial_step * direction
                    mol.update_coordinates(trial)
                    trial_energy = float(model.sum_bone()["energy"].item())
                    if trial_energy < base_energy:
                        accepted = True
                        accepted_energy = trial_energy
                        break
                    trial_step *= 0.5

                if not accepted:
                    mol.update_coordinates(coords0)
                    stop_reason = "line_search_failed"
                    current_energy = base_energy
                    break

                current_energy = accepted_energy
                if step % max(1, int(print_interval)) == 0 or step == int(max_steps):
                    out_now, _, max_f_now, mean_f_now = evaluate()
                    rows.append(
                        {
                            "step": step,
                            "energy_ev": float(out_now["energy"].item()),
                            "max_force_ev_A": max_f_now,
                            "mean_force_ev_A": mean_f_now,
                            "accepted_step_A": trial_step,
                        }
                    )
        else:
            velocity = torch.zeros_like(mol.coordinates)
            dt_fire = float(step_size)
            dt_max = max(float(step_size) * 20.0, float(step_size))
            max_disp = max(float(step_size) * 2.0, 1e-6)
            alpha = 0.1
            alpha_start = 0.1
            f_inc = 1.1
            f_dec = 0.5
            f_alpha = 0.99
            n_min = 5
            n_positive = 0

            for step in range(1, int(max_steps) + 1):
                out, forces, max_force, mean_force = evaluate()
                current_energy = float(out["energy"].item())
                if max_force <= float(force_threshold):
                    converged = True
                    stop_reason = "force_threshold_reached"
                    break

                velocity = velocity + dt_fire * forces
                power = float((velocity * forces).sum().item())
                if power > 0.0:
                    n_positive += 1
                    v_norm = torch.linalg.norm(velocity)
                    f_norm = torch.linalg.norm(forces)
                    if v_norm > 0.0 and f_norm > 0.0:
                        velocity = (1.0 - alpha) * velocity + alpha * (v_norm / f_norm) * forces
                    if n_positive > n_min:
                        dt_fire = min(dt_fire * f_inc, dt_max)
                        alpha *= f_alpha
                else:
                    n_positive = 0
                    dt_fire *= f_dec
                    alpha = alpha_start
                    velocity.zero_()

                displacement = dt_fire * velocity
                disp_norm = torch.linalg.norm(displacement, dim=1, keepdim=True)
                scale = torch.clamp(max_disp / disp_norm.clamp_min(1e-12), max=1.0)
                displacement = displacement * scale
                mol.update_coordinates(mol.coordinates + displacement)

                if step % max(1, int(print_interval)) == 0 or step == int(max_steps):
                    out_now, _, max_f_now, mean_f_now = evaluate()
                    rows.append(
                        {
                            "step": step,
                            "energy_ev": float(out_now["energy"].item()),
                            "max_force_ev_A": max_f_now,
                            "mean_force_ev_A": mean_f_now,
                            "accepted_step_A": float(torch.linalg.norm(displacement, dim=1).max().item()),
                        }
                    )

        out_final, _, final_max_force, final_mean_force = evaluate()
        final_energy = float(out_final["energy"].item())
        coords = mol.coordinates.detach().cpu()

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "energy_ev", "max_force_ev_A", "mean_force_ev_A", "accepted_step_A"],
        )
        writer.writeheader()
        writer.writerows(rows)

    write_xyz_with_types(
        relaxed_path,
        coords,
        atom_types,
        comment=f"relaxed by Simulon WStructureBuilder fixed-box {method.upper()}, energy={final_energy:.8f} eV",
    )
    summary = {
        "relaxation_method": f"fixed_box_{method}",
        "eam": str(eam_path),
        "device": str(device),
        "max_steps": int(max_steps),
        "step_size_A": float(step_size),
        "force_threshold_ev_A": float(force_threshold),
        "converged": bool(converged),
        "stop_reason": stop_reason,
        "initial_energy_ev": initial_energy,
        "final_energy_ev": final_energy,
        "energy_drop_ev": initial_energy - final_energy,
        "final_max_force_ev_A": final_max_force,
        "final_mean_force_ev_A": final_mean_force,
        "relaxed_structure": str(relaxed_path),
        "relaxation_csv": str(csv_path),
        "fixed_box_vectors_A": [[float(x) for x in row] for row in box_vectors.tolist()],
        "notes": [
            "Fixed-box geometry relaxation after structure building.",
            "This does not replace production NVT/NPT relaxation or GB rigid-body translation search.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Relaxation: E {initial_energy:.6f} -> {final_energy:.6f} eV, "
        f"max|F|={final_max_force:.6f} eV/A, converged={converged}"
    )
    return summary
