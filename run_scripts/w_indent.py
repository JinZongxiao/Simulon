import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.force.eam_force_cu import EAMForceCUDAExt as EAMForce
from core.integrator.integrator import VerletIntegrator
from core.md_model import BaseModel, SumBackboneInterface
from core.mechanics import SphericalIndenterForce
from io_utils.eam_parser import EAMParser
from io_utils.reader import AtomFileReader
from io_utils.w_bcc import generate_oriented_bcc_w, write_xyz
from postprocess.indentation import plot_load_depth, summarize_load_depth


_EV_PER_ANG_TO_NN = 1.602176634


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths():
    root = _project_root()
    return (
        root / "run_data" / "W" / "WRe_YC2.eam.fs",
        root / "run_output" / "w_indent",
    )


def _default_replicas(orientation: str) -> tuple[int, int, int]:
    return {
        "100": (6, 6, 4),
        "110": (6, 6, 4),
        "111": (5, 5, 3),
    }[orientation]


def _parse_replicas(value: str | None, orientation: str) -> tuple[int, int, int]:
    if not value:
        return _default_replicas(orientation)
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3 or any(x <= 0 for x in parts):
        raise ValueError(f"invalid replicas={value}")
    return tuple(parts)


def _build_parser() -> argparse.ArgumentParser:
    eam_default, out_default = _default_paths()
    p = argparse.ArgumentParser(description="Run a minimal W nanoindentation simulation")
    p.add_argument("--eam", default=str(eam_default))
    p.add_argument("--output-dir", default=str(out_default))
    p.add_argument("--orientation", choices=("100", "110", "111"), default="100")
    p.add_argument("--lattice-param", type=float, default=3.2)
    p.add_argument("--replicas", default=None, help="slab replicas as nx,ny,nz")
    p.add_argument("--vacuum-A", type=float, default=24.0)
    p.add_argument("--bottom-thickness-A", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--indenter-radius-A", type=float, default=8.0)
    p.add_argument("--indenter-stiffness", type=float, default=5.0, help="eV/A^3")
    p.add_argument("--initial-depth-A", type=float, default=0.05)
    p.add_argument("--indent-rate-A-ps", type=float, default=0.05)
    p.add_argument("--print-interval", type=int, default=50)
    p.add_argument("--smoke", action="store_true")
    return p


def _axis_unit(box_h: torch.Tensor, axis: int, device, dtype) -> torch.Tensor:
    vec = box_h[axis].to(device=device, dtype=dtype)
    return vec / torch.linalg.norm(vec).clamp_min(1e-12)


def _make_slab_box(box_vectors: torch.Tensor, axis: int, vacuum_A: float) -> torch.Tensor:
    h = box_vectors.clone().to(torch.float64)
    length = torch.linalg.norm(h[axis]).item()
    if length <= 0.0:
        raise ValueError("invalid slab axis length")
    h[axis] = h[axis] * ((length + float(vacuum_A)) / length)
    return h


def run_w_indent(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / f"orientation_{args.orientation}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.steps = min(args.steps, 60)
        args.print_interval = min(args.print_interval, 10)
        args.indent_rate_A_ps = min(args.indent_rate_A_ps, 0.05)
        if args.replicas is None:
            args.replicas = "3,3,3"

    parser = EAMParser(filepath=args.eam, device=device)
    replicas = _parse_replicas(args.replicas, args.orientation)
    coords, base_h = generate_oriented_bcc_w(
        lattice_param=args.lattice_param,
        orientation=args.orientation,
        replicas=replicas,
    )
    box_vectors = _make_slab_box(base_h, axis=2, vacuum_A=args.vacuum_A)
    structure_path = output_dir / f"W_{args.orientation}_indent_slab.xyz"
    write_xyz(
        structure_path,
        coords,
        atom_type="W",
        comment=f"W indentation slab orientation={args.orientation} replicas={replicas}",
    )

    mol = AtomFileReader(
        filename=str(structure_path),
        box_length=float(torch.linalg.norm(box_vectors[0]).item()),
        cutoff=parser.cutoff,
        device=device,
        skin_thickness=1.0,
        is_mlp=True,
        box_vectors=box_vectors,
    )

    axis_unit = _axis_unit(mol.box.H, 2, device, mol.coordinates.dtype)
    projected = mol.coordinates @ axis_unit
    bottom = float(projected.min().item())
    top = float(projected.max().item())
    fixed_mask = projected <= bottom + float(args.bottom_thickness_A)
    fixed_positions = mol.coordinates[fixed_mask].detach().clone()

    lateral_center = 0.5 * mol.box.H[0].to(device=device, dtype=mol.coordinates.dtype)
    lateral_center = lateral_center + 0.5 * mol.box.H[1].to(device=device, dtype=mol.coordinates.dtype)

    def center_at_depth(depth_A: float) -> torch.Tensor:
        normal_position = top + float(args.indenter_radius_A) - float(depth_A)
        return lateral_center + normal_position * axis_unit

    eam_force = EAMForce(parser, mol)
    indenter = SphericalIndenterForce(
        mol,
        radius=args.indenter_radius_A,
        stiffness=args.indenter_stiffness,
        center=center_at_depth(args.initial_depth_A),
    )
    sb = SumBackboneInterface([eam_force, indenter], mol)
    integ = VerletIntegrator(
        mol,
        dt=args.dt,
        ensemble="NVT",
        temperature=(args.temperature, args.temperature),
        gamma=args.gamma,
    )
    model = BaseModel(sb, integ, mol)

    with torch.no_grad():
        mol.atom_velocities[fixed_mask] = 0.0
    csv_path = output_dir / "load_depth.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "depth_A",
                "load_nN",
                "indenter_force_axis_ev_per_A",
                "potential_energy_ev",
                "indenter_energy_ev",
                "kinetic_energy_ev",
                "total_energy_ev",
                "temperature_k",
                "contact_atoms",
                "max_force_ev_per_A",
            ]
        )

        for step in range(args.steps):
            depth = float(args.initial_depth_A) + (step + 1) * float(args.indent_rate_A_ps) * float(args.dt)
            indenter.set_center(center_at_depth(depth))
            out = model()

            with torch.no_grad():
                mol.coordinates[fixed_mask] = fixed_positions
                mol.atom_velocities[fixed_mask] = 0.0
                mol.update_coordinates(mol.coordinates)
                mol.last_positions = mol.coordinates.detach().clone()

            indenter_axis_force = float(torch.dot(indenter.force_on_indenter, axis_unit).item())
            load_nN = max(0.0, indenter_axis_force * _EV_PER_ANG_TO_NN)
            indenter_energy = float(indenter.last_energy.item())
            potential = float(out["energy"].item())
            kinetic = float(out["kinetic_energy"].item())
            total = potential + kinetic
            temp = float(out["temperature"].item())
            max_force = float(torch.linalg.norm(out["forces"], dim=1).max().item())

            writer.writerow(
                [
                    step + 1,
                    depth,
                    load_nN,
                    indenter_axis_force,
                    potential,
                    indenter_energy,
                    kinetic,
                    total,
                    temp,
                    int(indenter.contact_atoms),
                    max_force,
                ]
            )

            if (step + 1) % max(1, args.print_interval) == 0:
                print(
                    f"Step {step + 1}/{args.steps}: depth={depth:.4f} A, "
                    f"load={load_nN:.4f} nN, contact={indenter.contact_atoms}, "
                    f"Pot_E={potential:.4f} eV, T={temp:.2f} K"
                )

    summary = summarize_load_depth(csv_path)
    plot_path = output_dir / "load_depth.png"
    plot_load_depth(csv_path, plot_path)
    summary.update(
        {
            "structure": str(structure_path),
            "eam": str(args.eam),
            "orientation": str(args.orientation),
            "replicas": list(replicas),
            "steps": int(args.steps),
            "dt_ps": float(args.dt),
            "temperature_k": float(args.temperature),
            "indenter_radius_A": float(args.indenter_radius_A),
            "indenter_stiffness_ev_A3": float(args.indenter_stiffness),
            "indent_rate_A_ps": float(args.indent_rate_A_ps),
            "output_dir": str(output_dir),
            "csv": str(csv_path),
            "plot": str(plot_path),
            "device": str(device),
            "smoke": bool(args.smoke),
        }
    )
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"W indentation completed. Data: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Plot: {plot_path}")
    if args.smoke:
        print("SMOKE TEST PASS")
    return summary


def main():
    args = _build_parser().parse_args()
    run_w_indent(args)


if __name__ == "__main__":
    main()
