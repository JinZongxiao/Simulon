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
from io_utils.eam_parser import EAMParser
from io_utils.reader import AtomFileReader
from io_utils.w_bcc import generate_oriented_bcc_w, write_xyz
from postprocess.crack import plot_crack, summarize_crack


_EV_ANG3_TO_BAR = 160_217.66


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths():
    root = _project_root()
    return (
        root / "run_data" / "W" / "W250.xyz",
        root / "run_data" / "W" / "WRe_YC2.eam.fs",
        root / "run_output" / "w_crack",
    )


def _default_replicas(orientation: str) -> tuple[int, int, int]:
    return {
        "100": (8, 8, 4),
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
    xyz_default, eam_default, out_default = _default_paths()
    p = argparse.ArgumentParser(description="Run a minimal W crack-opening simulation")
    p.add_argument("--structure", default=str(xyz_default))
    p.add_argument("--eam", default=str(eam_default))
    p.add_argument("--output-dir", default=str(out_default))
    p.add_argument("--box-length", type=float, default=16.0)
    p.add_argument("--orientation", choices=("100", "110", "111", "custom"), default="100")
    p.add_argument("--lattice-param", type=float, default=3.1652)
    p.add_argument("--replicas", default=None, help="supercell replicas as nx,ny,nz")
    p.add_argument("--vacuum-A", type=float, default=24.0)
    p.add_argument("--crack-half-length-A", type=float, default=8.0)
    p.add_argument("--crack-gap-A", type=float, default=1.2)
    p.add_argument("--grip-thickness-A", type=float, default=3.0)
    p.add_argument("--equil-steps", type=int, default=200)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--target-strain", type=float, default=0.02)
    p.add_argument("--opening-rate-A-ps", type=float, default=None)
    p.add_argument("--print-interval", type=int, default=50)
    p.add_argument("--traj-interval", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    return p


def _axis_unit(box_h: torch.Tensor, axis: int, device, dtype) -> torch.Tensor:
    vec = box_h[axis].to(device=device, dtype=dtype)
    return vec / torch.linalg.norm(vec).clamp_min(1e-12)


def _axis_unit_cpu(box_h: torch.Tensor, axis: int, dtype) -> torch.Tensor:
    vec = box_h[axis].to(dtype=dtype)
    return vec / torch.linalg.norm(vec).clamp_min(1e-12)


def _make_open_box(box_vectors: torch.Tensor, axis: int, vacuum_A: float) -> torch.Tensor:
    h = box_vectors.clone().to(torch.float64)
    length = torch.linalg.norm(h[axis]).item()
    h[axis] = h[axis] * ((length + float(vacuum_A)) / length)
    return h


def _read_xyz_coords(path: str | Path) -> torch.Tensor:
    coords = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not coords:
        raise ValueError(f"no coordinates found in {path}")
    return torch.tensor(coords, dtype=torch.float64)


def _kinetic_stress_tensor(model) -> torch.Tensor:
    vel = model.molecular.atom_velocities
    masses = model.Integrator.atom_mass[:, :1]
    return torch.einsum("ni,nj->ij", masses * vel, vel)


def _project_to_lattice_axes(tensor: torch.Tensor, box) -> torch.Tensor:
    axes = box.H.to(device=tensor.device, dtype=tensor.dtype)
    axes = axes / torch.linalg.norm(axes, dim=1, keepdim=True).clamp_min(1e-12)
    return torch.einsum("ai,ij,aj->a", axes, tensor, axes)


def _select_mouth_groups(vx, vy, x_center: float, y_center: float, crack_half_length: float):
    mouth_x = x_center + crack_half_length
    for xw in (0.8, 1.2, 1.8, 2.5):
        for yw in (0.8, 1.2, 1.8, 2.5):
            upper = (torch.abs(vx - mouth_x) <= xw) & (vy > y_center) & ((vy - y_center) <= yw)
            lower = (torch.abs(vx - mouth_x) <= xw) & (vy < y_center) & ((y_center - vy) <= yw)
            if bool(upper.any().item()) and bool(lower.any().item()):
                return upper, lower
    raise ValueError("failed to find crack-mouth atoms; increase replicas or crack size")


def _write_traj_frame(path: Path, coords: torch.Tensor, atom_types: list[str], comment: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{coords.shape[0]}\n")
        f.write(f"{comment}\n")
        coords_cpu = coords.detach().cpu().tolist()
        for atom_type, xyz in zip(atom_types, coords_cpu):
            f.write(f"{atom_type} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\n")


def run_w_crack(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / f"orientation_{args.orientation}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.steps = min(args.steps, 60)
        args.equil_steps = min(args.equil_steps, 10)
        args.target_strain = min(args.target_strain, 0.003)
        if args.opening_rate_A_ps is not None:
            args.opening_rate_A_ps = min(args.opening_rate_A_ps, 0.5)
        args.print_interval = min(args.print_interval, 10)
        if args.replicas is None:
            args.replicas = "4,4,3"
        args.crack_half_length_A = min(args.crack_half_length_A, 4.5)

    parser = EAMParser(filepath=args.eam, device=device)
    if args.orientation == "custom":
        replicas = None
        coords = _read_xyz_coords(args.structure)
        base_h = torch.diag(torch.tensor([args.box_length, args.box_length, args.box_length], dtype=torch.float64))
    else:
        replicas = _parse_replicas(args.replicas, args.orientation)
        coords, base_h = generate_oriented_bcc_w(
            lattice_param=args.lattice_param,
            orientation=args.orientation,
            replicas=replicas,
        )
    box_vectors = _make_open_box(base_h, axis=1, vacuum_A=args.vacuum_A)

    x_unit = _axis_unit_cpu(box_vectors, 0, coords.dtype)
    y_unit = _axis_unit_cpu(box_vectors, 1, coords.dtype)

    vx = coords @ x_unit
    vy = coords @ y_unit
    x_center = 0.5 * float(vx.min().item() + vx.max().item())
    y_center = 0.5 * float(vy.min().item() + vy.max().item())
    crack_plane_y = float(vy[torch.argmin(torch.abs(vy - y_center))].item())
    crack_mask = (torch.abs(vy - crack_plane_y) <= 0.5 * float(args.crack_gap_A)) & (
        torch.abs(vx - x_center) <= float(args.crack_half_length_A)
    )
    keep_mask = ~crack_mask
    if int(keep_mask.sum().item()) == coords.shape[0]:
        raise ValueError("crack geometry removed no atoms; adjust crack-gap-A or crack-half-length-A")
    coords = coords[keep_mask]

    structure_path = output_dir / f"W_{args.orientation}_crack.xyz"
    write_xyz(
        structure_path,
        coords,
        atom_type="W",
        comment=f"W crack orientation={args.orientation} replicas={replicas}",
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

    x_unit = _axis_unit(mol.box.H, 0, device, mol.coordinates.dtype)
    y_unit = _axis_unit(mol.box.H, 1, device, mol.coordinates.dtype)
    vx = mol.coordinates @ x_unit
    vy = mol.coordinates @ y_unit
    bottom = float(vy.min().item())
    top = float(vy.max().item())
    x_center = 0.5 * float(vx.min().item() + vx.max().item())
    y_center = crack_plane_y
    grip = float(args.grip_thickness_A)
    lower_grip = vy <= bottom + grip
    upper_grip = vy >= top - grip
    if not bool(lower_grip.any().item()) or not bool(upper_grip.any().item()):
        raise ValueError("failed to identify grip regions; increase replicas or reduce grip thickness")
    lower_ref = mol.coordinates[lower_grip].detach().clone()
    upper_ref = mol.coordinates[upper_grip].detach().clone()

    upper_mouth, lower_mouth = _select_mouth_groups(vx, vy, x_center, y_center, float(args.crack_half_length_A))
    gauge_length = max(1.0e-6, (top - bottom - 2.0 * grip))
    target_opening = float(args.target_strain) * gauge_length
    if args.opening_rate_A_ps is None:
        opening_rate = target_opening / (max(1, int(args.steps)) * float(args.dt))
    else:
        opening_rate = float(args.opening_rate_A_ps)

    ff = EAMForce(parser, mol)
    sb = SumBackboneInterface([ff], mol)
    integ = VerletIntegrator(
        mol,
        dt=args.dt,
        ensemble="NVT",
        temperature=(args.temperature, args.temperature),
        gamma=args.gamma,
    )
    model = BaseModel(sb, integ, mol)

    def apply_grips(total_opening: float) -> None:
        shift = 0.5 * float(total_opening) * y_unit
        with torch.no_grad():
            mol.coordinates[lower_grip] = lower_ref - shift
            mol.coordinates[upper_grip] = upper_ref + shift
            mol.atom_velocities[lower_grip] = 0.0
            mol.atom_velocities[upper_grip] = 0.0
            mol.update_coordinates(mol.coordinates)
            mol.last_positions = mol.coordinates.detach().clone()

    apply_grips(0.0)
    if args.equil_steps > 0:
        for eq_step in range(args.equil_steps):
            eq_out = model()
            apply_grips(0.0)
            if (eq_step + 1) % max(1, args.print_interval) == 0:
                print(f"Equil {eq_step + 1}/{args.equil_steps}: T={float(eq_out['temperature']):.2f} K")
    vy_zero = mol.coordinates @ y_unit
    initial_cmod = float((vy_zero[upper_mouth].mean() - vy_zero[lower_mouth].mean()).item())
    traj_path = output_dir / "trajectory.xyz"
    if int(args.traj_interval) > 0:
        if traj_path.exists():
            traj_path.unlink()
        _write_traj_frame(traj_path, mol.coordinates, mol.atom_types, "step=0 strain=0.000000 cmod=0.000000")

    csv_path = output_dir / "crack_response.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "applied_strain",
                "opening_A",
                "stress_bar",
                "cmod_A",
                "signed_cmod_A",
                "potential_energy_ev",
                "kinetic_energy_ev",
                "total_energy_ev",
                "temperature_k",
                "box_length_x",
                "box_length_y",
                "box_length_z",
            ]
        )

        for step in range(args.steps):
            opening = min(target_opening, (step + 1) * opening_rate * float(args.dt))
            apply_grips(opening)
            out = model()
            apply_grips(opening)

            kinetic_tensor = _kinetic_stress_tensor(model)
            virial_tensor = out["virial_tensor"].to(kinetic_tensor.dtype)
            sigma_tensor_bar = ((kinetic_tensor + virial_tensor) / float(mol.box.volume)) * _EV_ANG3_TO_BAR
            stress_axis = _project_to_lattice_axes(sigma_tensor_bar, mol.box)
            stress_bar = float(stress_axis[1])

            vy_now = mol.coordinates @ y_unit
            signed_cmod = float((vy_now[upper_mouth].mean() - vy_now[lower_mouth].mean()).item() - initial_cmod)
            cmod = abs(signed_cmod)
            total = float(out["energy"].item() + out["kinetic_energy"].item())
            lengths = mol.box.lengths.detach().cpu()

            writer.writerow(
                [
                    step + 1,
                    opening / gauge_length,
                    opening,
                    stress_bar,
                    cmod,
                    signed_cmod,
                    float(out["energy"].item()),
                    float(out["kinetic_energy"].item()),
                    total,
                    float(out["temperature"].item()),
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
                    f"step={step + 1} strain={opening / gauge_length:.6f} cmod={cmod:.6f}",
                )

            if (step + 1) % max(1, args.print_interval) == 0:
                print(
                    f"Step {step + 1}/{args.steps}: "
                    f"strain={opening / gauge_length:.6f}, stress={stress_bar:.2f} bar, "
                    f"CMOD={cmod:.4f} A, T={float(out['temperature']):.2f} K"
                )

    summary = summarize_crack(csv_path)
    plot_path = output_dir / "crack_response.png"
    plot_crack(csv_path, plot_path)
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
            "target_strain": float(args.target_strain),
            "opening_rate_A_ps": float(opening_rate),
            "crack_half_length_A": float(args.crack_half_length_A),
            "crack_gap_A": float(args.crack_gap_A),
            "grip_thickness_A": float(args.grip_thickness_A),
            "gauge_length_A": float(gauge_length),
            "output_dir": str(output_dir),
            "csv": str(csv_path),
            "plot": str(plot_path),
            "traj": str(traj_path) if int(args.traj_interval) > 0 else None,
            "device": str(device),
            "smoke": bool(args.smoke),
        }
    )
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"W crack completed. Data: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Plot: {plot_path}")
    if int(args.traj_interval) > 0:
        print(f"Trajectory: {traj_path}")
    if args.smoke:
        if float(summary.get("max_cmod_A", 0.0)) <= 0.0:
            raise AssertionError("crack smoke test requires positive CMOD response")
        print("SMOKE TEST PASS")
    return summary


def main():
    args = _build_parser().parse_args()
    run_w_crack(args)


if __name__ == "__main__":
    main()
