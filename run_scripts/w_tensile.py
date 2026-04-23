import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.md_model import BaseModel, SumBackboneInterface
from core.integrator.integrator import VerletIntegrator
from core.mechanics import UniaxialTensileLoader
from core.barostat import AnisotropicNPTBarostat
from core.force.eam_force_cu import EAMForceCUDAExt as EAMForce
from io_utils.eam_parser import EAMParser
from io_utils.reader import AtomFileReader
from io_utils.w_bcc import generate_oriented_bcc_w, write_xyz
from postprocess.stress_strain import plot_stress_strain, summarize_stress_strain


_EV_ANG3_TO_BAR = 160_217.66


def _kinetic_stress_tensor(model) -> torch.Tensor:
    vel = model.molecular.atom_velocities
    masses = model.Integrator.atom_mass[:, :1]
    return torch.einsum("ni,nj->ij", masses * vel, vel)


def _project_to_lattice_axes(tensor: torch.Tensor, box) -> torch.Tensor:
    axes = box.H.to(device=tensor.device, dtype=tensor.dtype)
    axes = axes / torch.linalg.norm(axes, dim=1, keepdim=True).clamp_min(1e-12)
    return torch.einsum("ai,ij,aj->a", axes, tensor, axes)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths():
    root = _project_root()
    return (
        root / "run_data" / "W" / "W250.xyz",
        root / "run_data" / "W" / "WRe_YC2.eam.fs",
        root / "run_output" / "w_tensile",
    )


def _default_replicas(orientation: str) -> tuple[int, int, int]:
    return {
        "100": (5, 5, 5),
        "110": (4, 4, 3),
        "111": (3, 3, 2),
    }[orientation]


def _build_parser() -> argparse.ArgumentParser:
    xyz_default, eam_default, out_default = _default_paths()
    p = argparse.ArgumentParser(description="Run a minimal W uniaxial tensile simulation")
    p.add_argument("--structure", default=str(xyz_default))
    p.add_argument("--eam", default=str(eam_default))
    p.add_argument("--output-dir", default=str(out_default))
    p.add_argument("--box-length", type=float, default=16.0)
    p.add_argument("--orientation", choices=("100", "110", "111", "custom"), default="100")
    p.add_argument("--lattice-param", type=float, default=3.2)
    p.add_argument("--replicas", default=None, help="supercell replicas as nx,ny,nz for generated structures")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--strain-rate", type=float, default=0.002)
    p.add_argument("--axis", choices=("x", "y", "z"), default="x")
    p.add_argument("--lateral-mode", choices=("fixed", "poisson", "stress-free"), default="fixed")
    p.add_argument("--poisson-ratio", type=float, default=0.28)
    p.add_argument("--target-lateral-stress-bar", type=float, default=0.0)
    p.add_argument("--barostat-tau", type=float, default=0.2)
    p.add_argument("--barostat-gamma", type=float, default=2.0)
    p.add_argument("--print-interval", type=int, default=20)
    p.add_argument("--traj-interval", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    return p


def _axis_to_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def _parse_replicas(value: str | None, orientation: str) -> tuple[int, int, int]:
    if not value:
        return _default_replicas(orientation)
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3 or any(x <= 0 for x in parts):
        raise ValueError(f"invalid replicas={value}")
    return tuple(parts)


def run_w_tensile(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_output_dir = Path(args.output_dir)
    if args.orientation == "custom":
        output_dir = base_output_dir / "orientation_custom"
    else:
        output_dir = base_output_dir / f"orientation_{args.orientation}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.steps = min(args.steps, 30)
        args.print_interval = min(args.print_interval, 10)
        args.traj_interval = 0
        args.strain_rate = min(args.strain_rate, 0.001)
        if args.orientation != "custom" and args.replicas is None:
            args.replicas = "2,2,2"

    parser = EAMParser(filepath=args.eam, device=device)
    structure_path = args.structure
    box_vectors = None
    box_length = args.box_length
    if args.orientation != "custom":
        replicas = _parse_replicas(args.replicas, args.orientation)
        coords, box_vectors = generate_oriented_bcc_w(
            lattice_param=args.lattice_param,
            orientation=args.orientation,
            replicas=replicas,
        )
        generated_xyz = output_dir / f"W_{args.orientation}_generated.xyz"
        structure_path = write_xyz(
            generated_xyz,
            coords,
            atom_type="W",
            comment=f"W bcc oriented {args.orientation} replicas={replicas}",
        )
        box_length = float(torch.norm(box_vectors[0]).item())

    mol = AtomFileReader(
        filename=structure_path,
        box_length=box_length,
        cutoff=parser.cutoff,
        device=device,
        skin_thickness=1.0,
        is_mlp=True,
        box_vectors=box_vectors,
    )
    ff = EAMForce(parser, mol)
    sb = SumBackboneInterface([ff], mol)
    ensemble = "NPT" if args.lateral_mode == "stress-free" else "NVT"
    integ = VerletIntegrator(
        mol,
        dt=args.dt,
        ensemble=ensemble,
        temperature=(args.temperature, args.temperature),
        gamma=args.gamma,
    )
    barostat = None
    if args.lateral_mode == "stress-free":
        control_axes = [True, True, True]
        control_axes[_axis_to_index(args.axis)] = False
        barostat = AnisotropicNPTBarostat(
            molecular=mol,
            target_pressure_bar=[
                args.target_lateral_stress_bar,
                args.target_lateral_stress_bar,
                args.target_lateral_stress_bar,
            ],
            temperature_k=args.temperature,
            tau_p=args.barostat_tau,
            gamma_p=args.barostat_gamma,
            control_axes=tuple(control_axes),
        )
    model = BaseModel(sb, integ, mol, barostat=barostat)
    loader = UniaxialTensileLoader(
        mol,
        axis=_axis_to_index(args.axis),
        strain_rate=args.strain_rate,
        lateral_mode=("fixed" if args.lateral_mode == "stress-free" else args.lateral_mode),
        poisson_ratio=args.poisson_ratio,
    )

    csv_path = output_dir / "stress_strain.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "strain",
                "stress_bar",
                "stress_xx_bar",
                "stress_yy_bar",
                "stress_zz_bar",
                "potential_energy_ev",
                "kinetic_energy_ev",
                "total_energy_ev",
                "temperature_k",
                "virial_ev",
                "virial_xx_ev",
                "virial_yy_ev",
                "virial_zz_ev",
                "box_length_x",
                "box_length_y",
                "box_length_z",
            ]
        )

        for step in range(args.steps):
            out = model()
            strain = loader.step(args.dt)
            lengths = loader.current_lengths()
            volume = float(mol.box.volume)
            kinetic_tensor = _kinetic_stress_tensor(model)
            virial_tensor = out["virial_tensor"].to(kinetic_tensor.dtype)
            sigma_tensor_bar = ((kinetic_tensor + virial_tensor) / volume) * _EV_ANG3_TO_BAR
            stress_axis_bar = _project_to_lattice_axes(sigma_tensor_bar, mol.box)
            stress_xx_bar = float(stress_axis_bar[0])
            stress_yy_bar = float(stress_axis_bar[1])
            stress_zz_bar = float(stress_axis_bar[2])
            pot = float(out["energy"])
            kin = float(out["kinetic_energy"])
            temp = float(out["temperature"])
            virial = float(out["virial"])
            total = pot + kin

            writer.writerow(
                [
                    step + 1,
                    strain,
                    stress_xx_bar,
                    stress_xx_bar,
                    stress_yy_bar,
                    stress_zz_bar,
                    pot,
                    kin,
                    total,
                    temp,
                    virial,
                    float(virial_tensor[0, 0]),
                    float(virial_tensor[1, 1]),
                    float(virial_tensor[2, 2]),
                    float(lengths[0]),
                    float(lengths[1]),
                    float(lengths[2]),
                ]
            )

            if (step + 1) % max(1, args.print_interval) == 0:
                print(
                    f"Step {step + 1}/{args.steps}: "
                    f"strain={strain:.6f}, sigma_xx={stress_xx_bar:.2f} bar, "
                    f"Pot_E={pot:.4f} eV, T={temp:.2f} K"
                )

    summary = summarize_stress_strain(csv_path)
    plot_path = output_dir / "stress_strain.png"
    plot_stress_strain(csv_path, plot_path)
    summary.update(
        {
            "structure": str(structure_path),
            "eam": str(args.eam),
            "orientation": str(args.orientation),
            "steps": int(args.steps),
            "dt_ps": float(args.dt),
            "temperature_k": float(args.temperature),
            "strain_rate_ps_inv": float(args.strain_rate),
            "lateral_mode": str(args.lateral_mode),
            "device": str(device),
            "smoke": bool(args.smoke),
            "plot": str(plot_path),
        }
    )
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"W tensile completed. Data: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Plot: {plot_path}")
    if args.smoke:
        print("SMOKE TEST PASS")
    return {"csv": str(csv_path), "summary": str(summary_path), **summary}


def main():
    args = _build_parser().parse_args()
    run_w_tensile(args)


if __name__ == "__main__":
    main()
