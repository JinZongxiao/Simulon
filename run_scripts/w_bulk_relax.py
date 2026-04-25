import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.barostat import BerendsenBarostat
from core.force.eam_force_cu import EAMForceCUDAExt as EAMForce
from core.integrator.integrator import VerletIntegrator
from core.md_model import BaseModel, SumBackboneInterface
from io_utils.eam_parser import EAMParser
from io_utils.reader import AtomFileReader
from io_utils.w_bcc import generate_oriented_bcc_w, write_xyz


_EV_ANG3_TO_BAR = 1_602_176.6208


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths():
    root = _project_root()
    return (
        root / "run_data" / "W" / "W250.xyz",
        root / "run_data" / "W" / "WRe_YC2.eam.fs",
        root / "run_output" / "w_bulk_relax",
    )


def _default_replicas(orientation: str) -> tuple[int, int, int]:
    return {
        "100": (5, 5, 5),
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


def _kinetic_stress_tensor(model) -> torch.Tensor:
    vel = model.molecular.atom_velocities
    masses = model.Integrator.atom_mass[:, :1]
    return torch.einsum("ni,nj->ij", masses * vel, vel)


def _project_to_lattice_axes(tensor: torch.Tensor, box) -> torch.Tensor:
    axes = box.H.to(device=tensor.device, dtype=tensor.dtype)
    axes = axes / torch.linalg.norm(axes, dim=1, keepdim=True).clamp_min(1e-12)
    return torch.einsum("ai,ij,aj->a", axes, tensor, axes)


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


def _infer_bcc_cubic_cells_per_axis(atom_count: int) -> int | None:
    if atom_count <= 0 or atom_count % 2 != 0:
        return None
    cells = atom_count // 2
    n = round(cells ** (1.0 / 3.0))
    if n > 0 and n ** 3 == cells:
        return n
    return None


def _build_parser() -> argparse.ArgumentParser:
    xyz_default, eam_default, out_default = _default_paths()
    p = argparse.ArgumentParser(description="Relax a bulk W structure to near-zero pressure")
    p.add_argument("--structure", default=str(xyz_default))
    p.add_argument("--eam", default=str(eam_default))
    p.add_argument("--output-dir", default=str(out_default))
    p.add_argument("--box-length", type=float, default=16.0)
    p.add_argument("--orientation", choices=("100", "110", "111", "custom"), default="100")
    p.add_argument("--lattice-param", type=float, default=3.1652)
    p.add_argument("--replicas", default=None, help="supercell replicas as nx,ny,nz for generated structures")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--target-pressure-bar", type=float, default=0.0)
    p.add_argument("--barostat-tau", type=float, default=0.5)
    p.add_argument("--barostat-compressibility-bar-inv", type=float, default=3.2e-7)
    p.add_argument("--barostat-mu-max", type=float, default=0.01)
    p.add_argument("--print-interval", type=int, default=100)
    p.add_argument("--traj-interval", type=int, default=0)
    p.add_argument("--abort-temperature-k", type=float, default=5000.0)
    p.add_argument("--abort-pressure-bar", type=float, default=1.0e6)
    p.add_argument("--smoke", action="store_true")
    return p


def run_w_bulk_relax(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / ("orientation_custom" if args.orientation == "custom" else f"orientation_{args.orientation}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.steps = min(args.steps, 40)
        args.print_interval = min(args.print_interval, 10)
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
            comment=f"W bcc bulk orientation={args.orientation} replicas={replicas}",
        )

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
    barostat = BerendsenBarostat(
        molecular=mol,
        target_pressure=args.target_pressure_bar,
        tau_p=args.barostat_tau,
        compressibility=args.barostat_compressibility_bar_inv,
        mu_max=args.barostat_mu_max,
    )
    model = BaseModel(sb, integ, mol, barostat=barostat)

    traj_path = output_dir / "trajectory.xyz"
    if int(args.traj_interval) > 0 and traj_path.exists():
        traj_path.unlink()

    state0 = _measure_state(model)
    sigma0_tensor_bar = (
        (state0["kinetic_tensor"] + state0["virial_tensor"]) / float(mol.box.volume)
    ) * _EV_ANG3_TO_BAR
    stress0 = _project_to_lattice_axes(sigma0_tensor_bar, mol.box)
    lengths0 = mol.box.lengths.detach().clone()
    pressure0 = float(stress0.mean().item())

    csv_path = output_dir / "relaxation.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "pressure_bar",
                "stress_xx_bar",
                "stress_yy_bar",
                "stress_zz_bar",
                "potential_energy_ev",
                "kinetic_energy_ev",
                "total_energy_ev",
                "temperature_k",
                "box_length_x",
                "box_length_y",
                "box_length_z",
            ]
        )
        writer.writerow(
            [
                0,
                pressure0,
                float(stress0[0]),
                float(stress0[1]),
                float(stress0[2]),
                float(state0["energy"]),
                float(state0["kinetic_energy"]),
                float(state0["energy"] + state0["kinetic_energy"]),
                float(state0["temperature"]),
                float(lengths0[0]),
                float(lengths0[1]),
                float(lengths0[2]),
            ]
        )

        if int(args.traj_interval) > 0:
            _write_traj_frame(traj_path, mol.coordinates, mol.atom_types, "step=0")

        pressure_hist = [pressure0]
        temp_hist = [float(state0["temperature"])]
        stress_hist = [stress0]
        length_hist = [lengths0]

        for step in range(args.steps):
            out = model()
            kinetic_tensor = _kinetic_stress_tensor(model)
            virial_tensor = out["virial_tensor"].to(kinetic_tensor.dtype)
            sigma_tensor_bar = ((kinetic_tensor + virial_tensor) / float(mol.box.volume)) * _EV_ANG3_TO_BAR
            stress_axis_bar = _project_to_lattice_axes(sigma_tensor_bar, mol.box)
            pressure_bar = float(stress_axis_bar.mean().item())
            temp = float(out["temperature"])
            lengths = mol.box.lengths.detach().clone()
            pot = float(out["energy"])
            kin = float(out["kinetic_energy"])
            total = pot + kin

            if args.abort_temperature_k > 0.0 and temp > float(args.abort_temperature_k):
                raise RuntimeError(
                    f"temperature runaway detected: T={temp:.2f} K exceeds abort_temperature_k={args.abort_temperature_k}"
                )
            if args.abort_pressure_bar > 0.0 and max(abs(float(x)) for x in stress_axis_bar) > float(args.abort_pressure_bar):
                raise RuntimeError(
                    "pressure runaway detected: "
                    f"max|sigma|={max(abs(float(x)) for x in stress_axis_bar):.2f} bar "
                    f"exceeds abort_pressure_bar={args.abort_pressure_bar}"
                )

            writer.writerow(
                [
                    step + 1,
                    pressure_bar,
                    float(stress_axis_bar[0]),
                    float(stress_axis_bar[1]),
                    float(stress_axis_bar[2]),
                    pot,
                    kin,
                    total,
                    temp,
                    float(lengths[0]),
                    float(lengths[1]),
                    float(lengths[2]),
                ]
            )
            pressure_hist.append(pressure_bar)
            temp_hist.append(temp)
            stress_hist.append(stress_axis_bar.detach().cpu())
            length_hist.append(lengths.detach().cpu())

            if int(args.traj_interval) > 0 and (step + 1) % max(1, int(args.traj_interval)) == 0:
                _write_traj_frame(traj_path, mol.coordinates, mol.atom_types, f"step={step + 1}")

            if (step + 1) % max(1, args.print_interval) == 0:
                print(
                    f"Step {step + 1}/{args.steps}: "
                    f"P={pressure_bar:.2f} bar, "
                    f"sigma=[{float(stress_axis_bar[0]):.2f}, {float(stress_axis_bar[1]):.2f}, {float(stress_axis_bar[2]):.2f}] bar, "
                    f"T={temp:.2f} K"
                )

    relaxed_structure_path = output_dir / f"W_{args.orientation}_relaxed.xyz"
    write_xyz(
        relaxed_structure_path,
        mol.coordinates.detach().cpu(),
        atom_type="W",
        comment=(
            f"W bulk relaxed orientation={args.orientation} "
            f"Lx={float(mol.box.lengths[0].item()):.6f} "
            f"Ly={float(mol.box.lengths[1].item()):.6f} "
            f"Lz={float(mol.box.lengths[2].item()):.6f}"
        ),
    )

    final_stress = stress_hist[-1]
    final_lengths = length_hist[-1]
    recommended_box_length = float(final_lengths.to(torch.float64).mean().item())
    inferred_cells = _infer_bcc_cubic_cells_per_axis(mol.atom_count)
    summary = {
        "structure": str(structure_path),
        "relaxed_structure": str(relaxed_structure_path),
        "eam": str(args.eam),
        "orientation": str(args.orientation),
        "replicas": list(replicas) if replicas is not None else None,
        "atom_count": int(mol.atom_count),
        "steps": int(args.steps),
        "dt_ps": float(args.dt),
        "temperature_k": float(args.temperature),
        "target_pressure_bar": float(args.target_pressure_bar),
        "barostat_tau": float(args.barostat_tau),
        "barostat_compressibility_bar_inv": float(args.barostat_compressibility_bar_inv),
        "barostat_mu_max": float(args.barostat_mu_max),
        "initial_pressure_bar": pressure_hist[0],
        "final_pressure_bar": pressure_hist[-1],
        "max_abs_pressure_bar": max(abs(p) for p in pressure_hist),
        "initial_stress_xx_bar": float(stress_hist[0][0]),
        "initial_stress_yy_bar": float(stress_hist[0][1]),
        "initial_stress_zz_bar": float(stress_hist[0][2]),
        "final_stress_xx_bar": float(final_stress[0]),
        "final_stress_yy_bar": float(final_stress[1]),
        "final_stress_zz_bar": float(final_stress[2]),
        "initial_box_length_x": float(length_hist[0][0]),
        "initial_box_length_y": float(length_hist[0][1]),
        "initial_box_length_z": float(length_hist[0][2]),
        "final_box_length_x": float(final_lengths[0]),
        "final_box_length_y": float(final_lengths[1]),
        "final_box_length_z": float(final_lengths[2]),
        "recommended_box_length_A": recommended_box_length,
        "max_temperature_k": max(temp_hist),
        "traj": str(traj_path) if int(args.traj_interval) > 0 else None,
        "device": str(device),
        "smoke": bool(args.smoke),
        "csv": str(csv_path),
    }
    if inferred_cells is not None:
        summary["inferred_cubic_cells_per_axis"] = int(inferred_cells)
        summary["recommended_lattice_param_A"] = recommended_box_length / float(inferred_cells)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"W bulk relax completed. Data: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Relaxed structure: {relaxed_structure_path}")
    if int(args.traj_interval) > 0:
        print(f"Trajectory: {traj_path}")
    if args.smoke:
        print("SMOKE TEST PASS")
    return {"csv": str(csv_path), "summary": str(summary_path), **summary}


def main():
    args = _build_parser().parse_args()
    run_w_bulk_relax(args)


if __name__ == "__main__":
    main()
