import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.barostat import AnisotropicNPTBarostat
from core.force.eam_force_cu import EAMForceCUDAExt as EAMForce
from core.integrator.integrator import VerletIntegrator
from core.md_model import BaseModel, SumBackboneInterface
from core.mechanics import UniaxialTensileLoader
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
    p = argparse.ArgumentParser(description="Run a W uniaxial tensile simulation")
    p.add_argument("--structure", default=str(xyz_default))
    p.add_argument("--eam", default=str(eam_default))
    p.add_argument("--output-dir", default=str(out_default))
    p.add_argument("--box-length", type=float, default=16.0)
    p.add_argument("--orientation", choices=("100", "110", "111", "custom"), default="100")
    p.add_argument("--lattice-param", type=float, default=3.2)
    p.add_argument("--replicas", default=None, help="supercell replicas as nx,ny,nz for generated structures")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--equil-steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--strain-rate", type=float, default=0.002)
    p.add_argument("--axis", choices=("x", "y", "z"), default="x")
    p.add_argument("--lateral-mode", choices=("fixed", "poisson", "stress-free"), default="stress-free")
    p.add_argument("--poisson-ratio", type=float, default=0.28)
    p.add_argument("--target-lateral-stress-bar", type=float, default=0.0)
    p.add_argument("--equil-target-pressure-bar", type=float, default=0.0)
    p.add_argument("--barostat-tau", type=float, default=0.2)
    p.add_argument("--barostat-gamma", type=float, default=2.0)
    p.add_argument("--barostat-compressibility-bar-inv", type=float, default=3.2e-6)
    p.add_argument("--barostat-pressure-tolerance-bar", type=float, default=25.0)
    p.add_argument("--max-lateral-box-ratio", type=float, default=2.0)
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


def _write_traj_frame(path: Path, coords: torch.Tensor, atom_types: list[str], comment: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{coords.shape[0]}\n")
        f.write(f"{comment}\n")
        coords_cpu = coords.detach().cpu().tolist()
        for atom_type, xyz in zip(atom_types, coords_cpu):
            f.write(f"{atom_type} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\n")


def _measure_state(model) -> dict:
    out = model.sum_bone()
    kinetic_tensor = _kinetic_stress_tensor(model)
    kinetic_energy = (0.5 * model.Integrator.atom_mass * model.molecular.atom_velocities.pow(2)).sum()
    temperature = (2.0 / 3.0) * kinetic_energy / (
        model.molecular.atom_count * model.Integrator.BOLTZMAN
    )
    return {
        "energy": out["energy"],
        "virial": out["virial"],
        "virial_tensor": out["virial_tensor"].to(kinetic_tensor.dtype),
        "kinetic_tensor": kinetic_tensor,
        "kinetic_energy": kinetic_energy,
        "temperature": temperature,
    }


def _make_loading_barostat(args, mol):
    if args.lateral_mode != "stress-free":
        return None
    control_axes = [True, True, True]
    control_axes[_axis_to_index(args.axis)] = False
    return AnisotropicNPTBarostat(
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
        compressibility_bar_inv=args.barostat_compressibility_bar_inv,
        pressure_tolerance_bar=args.barostat_pressure_tolerance_bar,
    )


def run_w_tensile(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / ("orientation_custom" if args.orientation == "custom" else f"orientation_{args.orientation}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.steps = min(args.steps, 30)
        args.equil_steps = min(args.equil_steps, 5)
        args.print_interval = min(args.print_interval, 10)
        args.strain_rate = min(args.strain_rate, 0.001)
        if args.orientation != "custom" and args.replicas is None:
            args.replicas = "2,2,2"

    parser = EAMParser(filepath=args.eam, device=device)
    structure_path = args.structure
    box_vectors = None
    box_length = args.box_length
    replicas = None
    if args.orientation != "custom":
        replicas = _parse_replicas(args.replicas, args.orientation)
        coords, box_vectors = generate_oriented_bcc_w(
            lattice_param=args.lattice_param,
            orientation=args.orientation,
            replicas=replicas,
        )
        box_length = float(torch.norm(box_vectors[0]).item())
        structure_path = write_xyz(
            output_dir / f"W_{args.orientation}_generated.xyz",
            coords,
            atom_type="W",
            comment=f"W bcc oriented {args.orientation} replicas={replicas}",
        )
    else:
        coords = None

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
    integ = VerletIntegrator(
        mol,
        dt=args.dt,
        ensemble="NPT",
        temperature=(args.temperature, args.temperature),
        gamma=args.gamma,
    )

    equil_barostat = None
    if int(args.equil_steps) > 0:
        equil_barostat = AnisotropicNPTBarostat(
            molecular=mol,
            target_pressure_bar=[
                args.equil_target_pressure_bar,
                args.equil_target_pressure_bar,
                args.equil_target_pressure_bar,
            ],
            temperature_k=args.temperature,
            tau_p=args.barostat_tau,
            gamma_p=args.barostat_gamma,
            control_axes=(True, True, True),
            compressibility_bar_inv=args.barostat_compressibility_bar_inv,
            pressure_tolerance_bar=args.barostat_pressure_tolerance_bar,
        )
    model = BaseModel(sb, integ, mol, barostat=equil_barostat)

    if int(args.equil_steps) > 0:
        for eq_step in range(int(args.equil_steps)):
            eq_out = model()
            if (eq_step + 1) % max(1, args.print_interval) == 0:
                print(f"Equil {eq_step + 1}/{args.equil_steps}: T={float(eq_out['temperature']):.2f} K")

    model.barostat = _make_loading_barostat(args, mol)
    loader = UniaxialTensileLoader(
        mol,
        axis=_axis_to_index(args.axis),
        strain_rate=args.strain_rate,
        lateral_mode=("fixed" if args.lateral_mode == "stress-free" else args.lateral_mode),
        poisson_ratio=args.poisson_ratio,
    )
    axis_idx = _axis_to_index(args.axis)
    lateral_axes = [i for i in range(3) if i != axis_idx]

    traj_path = output_dir / "trajectory.xyz"
    if int(args.traj_interval) > 0 and traj_path.exists():
        traj_path.unlink()

    state0 = _measure_state(model)
    sigma0_tensor_bar = (
        (state0["kinetic_tensor"] + state0["virial_tensor"]) / float(mol.box.volume)
    ) * _EV_ANG3_TO_BAR
    baseline_abs = _project_to_lattice_axes(sigma0_tensor_bar, mol.box)

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
                "stress_abs_bar",
                "stress_xx_abs_bar",
                "stress_yy_abs_bar",
                "stress_zz_abs_bar",
                "tension_bar",
                "tension_xx_bar",
                "tension_yy_bar",
                "tension_zz_bar",
                "tension_abs_bar",
                "tension_xx_abs_bar",
                "tension_yy_abs_bar",
                "tension_zz_abs_bar",
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

        lengths0 = loader.current_lengths()
        writer.writerow(
            [
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                float(baseline_abs[0]),
                float(baseline_abs[0]),
                float(baseline_abs[1]),
                float(baseline_abs[2]),
                0.0,
                0.0,
                0.0,
                0.0,
                float(baseline_abs[0]),
                float(baseline_abs[0]),
                float(baseline_abs[1]),
                float(baseline_abs[2]),
                float(state0["energy"]),
                float(state0["kinetic_energy"]),
                float(state0["energy"] + state0["kinetic_energy"]),
                float(state0["temperature"]),
                float(state0["virial"]),
                float(state0["virial_tensor"][0, 0]),
                float(state0["virial_tensor"][1, 1]),
                float(state0["virial_tensor"][2, 2]),
                float(lengths0[0]),
                float(lengths0[1]),
                float(lengths0[2]),
            ]
        )

        if int(args.traj_interval) > 0:
            _write_traj_frame(traj_path, mol.coordinates, mol.atom_types, "step=0 strain=0.000000")

        for step in range(args.steps):
            strain = loader.step(args.dt)
            out = model()
            lengths = loader.current_lengths()
            volume = float(mol.box.volume)
            kinetic_tensor = _kinetic_stress_tensor(model)
            virial_tensor = out["virial_tensor"].to(kinetic_tensor.dtype)
            sigma_tensor_bar = ((kinetic_tensor + virial_tensor) / volume) * _EV_ANG3_TO_BAR
            stress_axis_abs = _project_to_lattice_axes(sigma_tensor_bar, mol.box)
            stress_axis_bar = stress_axis_abs - baseline_abs.to(stress_axis_abs)
            tension_axis_abs = stress_axis_abs
            tension_axis_bar = stress_axis_bar

            pot = float(out["energy"])
            kin = float(out["kinetic_energy"])
            temp = float(out["temperature"])
            virial = float(out["virial"])
            total = pot + kin

            if args.lateral_mode == "stress-free" and args.max_lateral_box_ratio > 0.0:
                lateral_lengths = lengths[lateral_axes].to(torch.float64)
                lateral_ratios = lateral_lengths / lengths0[lateral_axes].to(torch.float64).clamp_min(1e-12)
                if bool(torch.any(lateral_ratios > float(args.max_lateral_box_ratio))):
                    raise RuntimeError(
                        "lateral box runaway detected: "
                        f"ratios={lateral_ratios.detach().cpu().tolist()} "
                        f"exceed max_lateral_box_ratio={args.max_lateral_box_ratio}"
                    )

            writer.writerow(
                [
                    step + 1,
                    strain,
                    float(stress_axis_bar[0]),
                    float(stress_axis_bar[0]),
                    float(stress_axis_bar[1]),
                    float(stress_axis_bar[2]),
                    float(stress_axis_abs[0]),
                    float(stress_axis_abs[0]),
                    float(stress_axis_abs[1]),
                    float(stress_axis_abs[2]),
                    float(tension_axis_bar[0]),
                    float(tension_axis_bar[0]),
                    float(tension_axis_bar[1]),
                    float(tension_axis_bar[2]),
                    float(tension_axis_abs[0]),
                    float(tension_axis_abs[0]),
                    float(tension_axis_abs[1]),
                    float(tension_axis_abs[2]),
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

            if int(args.traj_interval) > 0 and (step + 1) % max(1, int(args.traj_interval)) == 0:
                _write_traj_frame(
                    traj_path,
                    mol.coordinates,
                    mol.atom_types,
                    f"step={step + 1} strain={strain:.6f}",
                )

            if (step + 1) % max(1, args.print_interval) == 0:
                print(
                    f"Step {step + 1}/{args.steps}: "
                    f"strain={strain:.6f}, sigma_xx_tension={float(tension_axis_bar[0]):.2f} bar, "
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
            "replicas": list(replicas) if replicas is not None else None,
            "steps": int(args.steps),
            "equil_steps": int(args.equil_steps),
            "dt_ps": float(args.dt),
            "temperature_k": float(args.temperature),
            "strain_rate_ps_inv": float(args.strain_rate),
            "lateral_mode": str(args.lateral_mode),
            "equil_target_pressure_bar": float(args.equil_target_pressure_bar),
            "barostat_compressibility_bar_inv": float(args.barostat_compressibility_bar_inv),
            "barostat_pressure_tolerance_bar": float(args.barostat_pressure_tolerance_bar),
            "max_lateral_box_ratio": float(args.max_lateral_box_ratio),
            "initial_stress_xx_abs_bar": float(baseline_abs[0]),
            "initial_stress_yy_abs_bar": float(baseline_abs[1]),
            "initial_stress_zz_abs_bar": float(baseline_abs[2]),
            "traj": str(traj_path) if int(args.traj_interval) > 0 else None,
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
    if int(args.traj_interval) > 0:
        print(f"Trajectory: {traj_path}")
    if args.smoke:
        print("SMOKE TEST PASS")
    return {"csv": str(csv_path), "summary": str(summary_path), **summary}


def main():
    args = _build_parser().parse_args()
    run_w_tensile(args)


if __name__ == "__main__":
    main()
