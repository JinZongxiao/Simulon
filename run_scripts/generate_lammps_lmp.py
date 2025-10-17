#!/usr/bin/env python3
"""
Generate a LAMMPS data (.lmp) file for one or more elements at a target density.

Features:
- Input species as counts or ratios, e.g.:
  * --elements "Ar:100000"
  * --elements "Cu:5000,W:5000"
  * --elements "Cu:1,W:1" --total 10000  (ratios + total)
  * --elements "Ar" --total 10000       (single species)
- Density in g/cm^3
- Places atoms on a simple cubic grid or random positions within a cubic box sized from density
- Writes atom_style atomic/charge/molecular/full data file: Masses + Atoms sections

Example:
  python run_scripts/generate_lammps_lmp.py --elements "Ar:10000" --density 1.374 --style charge --placement random --output run_data/Ar_13A_lq.lmp
"""
from __future__ import annotations
import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

from core.element_info import element_info

NA = 6.02214076e23  # 1/mol


def parse_elements(spec: str, total: int | None) -> List[Tuple[str, int]]:
    """Parse element spec into list of (symbol, count).

    Accepts:
      - "Ar:10000"
      - "Cu:5000,W:5000"
      - "Cu:1,W:1" with --total 10000 (ratios)
      - "Ar" with --total 10000
    """
    tokens = [t.strip() for t in spec.replace(";", ",").split(",") if t.strip()]
    if not tokens:
        raise ValueError("--elements is empty")

    pairs: List[Tuple[str, float]] = []
    has_colon = any(":" in t for t in tokens)

    if has_colon:
        for t in tokens:
            if ":" in t:
                sym, num = t.split(":", 1)
                sym = sym.strip()
                try:
                    val = float(num)
                except Exception:
                    raise ValueError(f"Invalid number in token '{t}'")
                pairs.append((sym, val))
            else:
                pairs.append((t, 1.0))
        # decide counts vs ratios
        if total is None and all(abs(v - int(v)) < 1e-9 for _, v in pairs):
            # treat as absolute counts
            return [(s, int(v)) for s, v in pairs]
        # treat as ratios
        if total is None:
            raise ValueError("When using ratios in --elements, please provide --total")
        sum_ratio = sum(v for _, v in pairs)
        counts = [int(math.floor(v / sum_ratio * total)) for _, v in pairs]
        remainder = total - sum(counts)
        # distribute remainders to largest fractional parts
        fracs = [((v / sum_ratio * total) - c, i) for i, ((_, v), c) in enumerate(zip(pairs, counts))]
        fracs.sort(reverse=True)
        for k in range(remainder):
            counts[fracs[k][1]] += 1
        return [(pairs[i][0], counts[i]) for i in range(len(pairs))]
    else:
        # no colon: symbols only; require total
        if total is None:
            raise ValueError("Provide --total when element counts are not specified")
        m = len(tokens)
        base = total // m
        counts = [base] * m
        counts[0] += total - base * m  # put remainder to first
        return list(zip(tokens, counts))


def average_mass_amu(composition: List[Tuple[str, int]]) -> float:
    total_mass = 0.0
    for sym, cnt in composition:
        info = element_info.get(sym)
        if info is None or "mass" not in info:
            raise ValueError(f"Unknown element or missing mass: {sym}")
        total_mass += info["mass"] * cnt
    n_atoms = sum(cnt for _, cnt in composition)
    return total_mass / max(1, n_atoms)


def compute_box_length_A(composition: List[Tuple[str, int]], density_g_cm3: float) -> float:
    # mass in grams
    mass_g = 0.0
    for sym, cnt in composition:
        info = element_info.get(sym)
        if info is None or "mass" not in info:
            raise ValueError(f"Unknown element or missing mass: {sym}")
        mass_g += cnt * info["mass"] / NA
    volume_cm3 = mass_g / density_g_cm3
    volume_A3 = volume_cm3 * 1e24
    L = volume_A3 ** (1.0 / 3.0)
    return L


def build_simple_cubic_positions(n_atoms: int, box_len: float) -> List[Tuple[float, float, float]]:
    n_per_dim = max(1, math.ceil(n_atoms ** (1.0 / 3.0)))
    sx = sy = sz = n_per_dim
    while sx * sy * sz < n_atoms:
        # grow the smallest dimension
        if sx <= sy and sx <= sz:
            sx += 1
        elif sy <= sx and sy <= sz:
            sy += 1
        else:
            sz += 1
    spacing_x = box_len / sx
    spacing_y = box_len / sy
    spacing_z = box_len / sz
    coords: List[Tuple[float, float, float]] = []
    for ix in range(sx):
        x = (ix + 0.5) * spacing_x
        for iy in range(sy):
            y = (iy + 0.5) * spacing_y
            for iz in range(sz):
                z = (iz + 0.5) * spacing_z
                coords.append((x, y, z))
                if len(coords) >= n_atoms:
                    return coords
    return coords[:n_atoms]


def build_random_positions(n_atoms: int, box_len: float, seed: int | None = None) -> List[Tuple[float, float, float]]:
    rng = random.Random(seed)
    coords: List[Tuple[float, float, float]] = []
    for _ in range(n_atoms):
        x = rng.random() * box_len
        y = rng.random() * box_len
        z = rng.random() * box_len
        coords.append((x, y, z))
    return coords


def write_lammps_data_atomic(path: Path, composition: List[Tuple[str, int]], L: float, *, style: str = "charge", placement: str = "random", seed: int | None = None) -> None:
    # Map elements to type ids
    elements = [sym for sym, _ in composition]
    unique_syms = []
    for s in elements:
        if s not in unique_syms:
            unique_syms.append(s)
    type_map: Dict[str, int] = {s: i + 1 for i, s in enumerate(unique_syms)}

    # Flatten atom list with types preserving approximate proportions
    expanded: List[int] = []  # list of type ids
    for sym, cnt in composition:
        expanded.extend([type_map[sym]] * cnt)
    n_atoms = len(expanded)

    # Choose placement
    if placement == "random":
        coords = build_random_positions(n_atoms, L, seed)
    elif placement in ("sc", "simple-cubic", "grid"):
        coords = build_simple_cubic_positions(n_atoms, L)
    else:
        raise ValueError("placement must be one of: random, sc")

    # Masses
    masses_lines = ["Masses", ""]
    for sym in unique_syms:
        masses_lines.append(f"{type_map[sym]} {element_info[sym]['mass']:.8f} # {sym}")

    # Atoms section header
    atoms_header = f"Atoms # {style}" if style in {"atomic", "charge", "molecular", "full"} else "Atoms"
    atom_lines = [atoms_header, ""]

    # Generate atom lines according to style
    # atomic:    id type x y z
    # charge:    id type q x y z
    # molecular: id mol type x y z
    # full:      id mol type q x y z
    if style == "atomic":
        for i, (atype, (x, y, z)) in enumerate(zip(expanded, coords), start=1):
            atom_lines.append(f"{i} {atype} {x:.12f} {y:.12f} {z:.12f}")
    elif style == "charge":
        q = 0.0
        for i, (atype, (x, y, z)) in enumerate(zip(expanded, coords), start=1):
            atom_lines.append(f"{i} {atype} {q:.6f} {x:.12f} {y:.12f} {z:.12f}")
    elif style == "molecular":
        mol = 1
        for i, (atype, (x, y, z)) in enumerate(zip(expanded, coords), start=1):
            atom_lines.append(f"{i} {mol} {atype} {x:.12f} {y:.12f} {z:.12f}")
    elif style == "full":
        mol = 1
        q = 0.0
        for i, (atype, (x, y, z)) in enumerate(zip(expanded, coords), start=1):
            atom_lines.append(f"{i} {mol} {atype} {q:.6f} {x:.12f} {y:.12f} {z:.12f}")
    else:
        raise ValueError(f"Unsupported style: {style}")

    header = [
        "# LAMMPS data file generated by Simulon generator",
        f"{n_atoms} atoms",
        f"{len(unique_syms)} atom types",
        f"0.0 {L:.12f} xlo xhi",
        f"0.0 {L:.12f} ylo yhi",
        f"0.0 {L:.12f} zlo zhi",
        "",
    ]

    content = "\n".join(header + masses_lines + [""] + atom_lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Generate LAMMPS .lmp data file from density and elements")
    p.add_argument("--elements", required=True, help="Element spec, e.g. 'Ar:10000' or 'Cu:1,W:1' (with --total)")
    p.add_argument("--density", type=float, required=True, help="Target density in g/cm^3")
    p.add_argument("--total", type=int, default=None, help="Total atoms when using ratios or symbols without counts")
    p.add_argument("--style", type=str, default="charge", choices=["atomic", "charge", "molecular", "full"], help="LAMMPS atom_style to match Atoms section format")
    p.add_argument("--placement", type=str, default="random", choices=["random", "sc"], help="Coordinate placement: random or simple cubic (sc)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for --placement random")
    p.add_argument("--output", type=str, required=True, help="Output .lmp path")
    args = p.parse_args()

    composition = parse_elements(args.elements, args.total)
    if any(cnt <= 0 for _, cnt in composition):
        raise ValueError("All element counts must be positive")

    L = compute_box_length_A(composition, args.density)

    write_lammps_data_atomic(Path(args.output), composition, L, style=args.style, placement=args.placement, seed=args.seed)

    total_atoms = sum(c for _, c in composition)
    species_str = ", ".join(f"{s}:{c}" for s, c in composition)
    print(f"Wrote {total_atoms} atoms ({species_str}) at density {args.density} g/cm^3, box {L:.3f} Å, style={args.style}, placement={args.placement} -> {args.output}")


if __name__ == "__main__":
    main()
