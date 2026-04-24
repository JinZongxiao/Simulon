import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.force.eam_force_cu import EAMForceCUDAExt as EAMForce
from io_utils.eam_parser import EAMParser
from io_utils.reader import AtomFileReader
from io_utils.w_bcc import generate_oriented_bcc_w, write_xyz


_EV_ANG3_TO_BAR = 160_217.66
_ORIENTATIONS = ("100", "110", "111")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths() -> tuple[Path, Path]:
    root = _project_root()
    return (
        root / "run_data" / "W" / "WRe_YC2.eam.fs",
        root / "run_output" / "w_orientation_check",
    )


def _default_replicas(orientation: str) -> tuple[int, int, int]:
    return {
        "100": (4, 4, 3),
        "110": (4, 4, 3),
        "111": (3, 3, 2),
    }[orientation]


def _parse_replicas(value: str | None, orientation: str) -> tuple[int, int, int]:
    if not value:
        return _default_replicas(orientation)
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3 or any(x <= 0 for x in parts):
        raise ValueError(f"invalid replicas={value}")
    return tuple(parts)


def _format_diag(values: torch.Tensor) -> str:
    vals = [float(x) for x in values.detach().cpu().tolist()]
    return "[" + ", ".join(f"{x:.2f}" for x in vals) + "]"


def _project_to_lattice_axes(tensor: torch.Tensor, box) -> torch.Tensor:
    axes = box.H.to(device=tensor.device, dtype=tensor.dtype)
    axes = axes / torch.linalg.norm(axes, dim=1, keepdim=True).clamp_min(1e-12)
    return torch.einsum("ai,ij,aj->a", axes, tensor, axes)


def _build_parser() -> argparse.ArgumentParser:
    eam_default, out_default = _default_paths()
    p = argparse.ArgumentParser(
        description="Static EAM sanity check for generated oriented BCC W cells."
    )
    p.add_argument("--eam", default=str(eam_default))
    p.add_argument("--output-dir", default=str(out_default))
    p.add_argument("--orientation", choices=("100", "110", "111", "all"), default="all")
    p.add_argument("--lattice-param", type=float, default=3.1652)
    p.add_argument("--replicas", default=None, help="replicas as nx,ny,nz; default depends on orientation")
    p.add_argument("--skin-thickness", type=float, default=1.0)
    p.add_argument("--max-force-threshold", type=float, default=5.0, help="eV/Angstrom")
    p.add_argument("--stress-threshold-bar", type=float, default=50_000.0)
    return p


def check_orientation(args, orientation: str, parser: EAMParser, device: torch.device) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    orientation_dir = output_dir / f"orientation_{orientation}"
    orientation_dir.mkdir(parents=True, exist_ok=True)

    replicas = _parse_replicas(args.replicas, orientation)
    coords, box_vectors = generate_oriented_bcc_w(
        lattice_param=args.lattice_param,
        orientation=orientation,
        replicas=replicas,
    )
    structure_path = orientation_dir / f"W_{orientation}_check.xyz"
    write_xyz(
        structure_path,
        coords,
        atom_type="W",
        comment=f"W bcc orientation={orientation} replicas={replicas}",
    )

    mol = AtomFileReader(
        filename=str(structure_path),
        box_length=float(torch.norm(box_vectors[0]).item()),
        cutoff=parser.cutoff,
        device=device,
        skin_thickness=args.skin_thickness,
        is_mlp=True,
        box_vectors=box_vectors,
    )
    ff = EAMForce(parser, mol)
    with torch.no_grad():
        out = ff()

    forces = out["forces"]
    force_norm = torch.linalg.norm(forces, dim=1)
    virial_tensor = out["virial_tensor"].to(torch.float64)
    stress_tensor_bar = (virial_tensor / float(mol.box.volume)) * _EV_ANG3_TO_BAR
    stress_diag = _project_to_lattice_axes(stress_tensor_bar, mol.box)
    box_lengths = mol.box.lengths.detach().cpu()
    box_matrix = mol.box.H.detach().cpu()
    energy = float(out["energy"])
    natoms = int(mol.atom_count)
    max_force = float(force_norm.max())
    mean_force = float(force_norm.mean())
    energy_per_atom = energy / natoms
    max_abs_stress = float(torch.max(torch.abs(stress_diag)))

    passed = (
        energy_per_atom < 0.0
        and max_force < float(args.max_force_threshold)
        and max_abs_stress < float(args.stress_threshold_bar)
    )

    result = {
        "orientation": orientation,
        "replicas": list(replicas),
        "output_dir": str(orientation_dir),
        "structure": str(structure_path),
        "natoms": natoms,
        "energy_ev": energy,
        "energy_per_atom_ev": energy_per_atom,
        "max_force_ev_per_ang": max_force,
        "mean_force_ev_per_ang": mean_force,
        "stress_xx_bar": float(stress_diag[0]),
        "stress_yy_bar": float(stress_diag[1]),
        "stress_zz_bar": float(stress_diag[2]),
        "max_abs_stress_diag_bar": max_abs_stress,
        "virial_xx_ev": float(virial_tensor[0, 0]),
        "virial_yy_ev": float(virial_tensor[1, 1]),
        "virial_zz_ev": float(virial_tensor[2, 2]),
        "box_length_x": float(box_lengths[0]),
        "box_length_y": float(box_lengths[1]),
        "box_length_z": float(box_lengths[2]),
        "box_matrix": box_matrix.tolist(),
        "passed": passed,
    }

    print(
        f"{orientation}: atoms={natoms}, E/N={energy_per_atom:.6f} eV, "
        f"max|F|={max_force:.4f} eV/A, mean|F|={mean_force:.4f} eV/A, "
        f"stress_diag_bar={_format_diag(stress_diag)}, "
        f"box_lengths_A={_format_diag(box_lengths)}, pass={passed}, "
        f"dir={orientation_dir}"
    )
    return result


def write_outputs(results: list[dict], output_dir: Path) -> None:
    csv_path = output_dir / "orientation_check_all.csv"
    json_path = output_dir / "orientation_check_all.json"
    fields = [
        "orientation",
        "replicas",
        "natoms",
        "energy_ev",
        "energy_per_atom_ev",
        "max_force_ev_per_ang",
        "mean_force_ev_per_ang",
        "stress_xx_bar",
        "stress_yy_bar",
        "stress_zz_bar",
        "max_abs_stress_diag_bar",
        "box_length_x",
        "box_length_y",
        "box_length_z",
        "passed",
        "structure",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            csv_row = {key: row[key] for key in fields}
            csv_row["replicas"] = ",".join(str(x) for x in row["replicas"])
            writer.writerow(csv_row)

            orientation_dir = Path(row["output_dir"])
            orientation_csv = orientation_dir / "orientation_check.csv"
            orientation_json = orientation_dir / "orientation_check.json"
            with open(orientation_csv, "w", encoding="utf-8", newline="") as of:
                owriter = csv.DictWriter(of, fieldnames=fields)
                owriter.writeheader()
                owriter.writerow(csv_row)
            with open(orientation_json, "w", encoding="utf-8") as of:
                json.dump(row, of, indent=2)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"All results: {csv_path}")
    print(f"All summary: {json_path}")


def main() -> int:
    args = _build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = EAMParser(filepath=args.eam, device=device)
    orientations = _ORIENTATIONS if args.orientation == "all" else (args.orientation,)
    results = [
        check_orientation(args, orientation, parser, device)
        for orientation in orientations
    ]
    output_dir = Path(args.output_dir)
    write_outputs(results, output_dir)
    if all(row["passed"] for row in results):
        print("W orientation check PASS")
        return 0
    print("W orientation check FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
