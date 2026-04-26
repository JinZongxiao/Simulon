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
from postprocess.indentation import plot_load_depth, summarize_load_depth, write_indentation_report


_EV_PER_ANG_TO_NN = 1.602176634


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths():
    root = _project_root()
    return (
        root / "run_data" / "W" / "W250.xyz",
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
    xyz_default, eam_default, out_default = _default_paths()
    p = argparse.ArgumentParser(description="Run a W nanoindentation simulation")
    p.add_argument("--structure", default=str(xyz_default))
    p.add_argument("--eam", default=str(eam_default))
    p.add_argument("--output-dir", default=str(out_default))
    p.add_argument("--box-length", type=float, default=16.0)
    p.add_argument("--orientation", choices=("100", "110", "111", "custom"), default="100")
    p.add_argument("--lattice-param", type=float, default=3.1652)
    p.add_argument("--replicas", default=None, help="slab replicas as nx,ny,nz")
    p.add_argument("--vacuum-A", type=float, default=24.0)
    p.add_argument("--bottom-thickness-A", "--bottom-fixed-thickness-A", dest="bottom_thickness_A", type=float, default=3.0)
    p.add_argument("--steps", "--loading-steps", dest="steps", type=int, default=1000, help="loading steps")
    p.add_argument("--equil-steps", type=int, default=500)
    p.add_argument("--hold-steps", type=int, default=0, help="optional constant-depth hold steps")
    p.add_argument("--hold-depth-A", type=float, default=None, help="hold depth; defaults to target depth")
    p.add_argument("--unload-steps", type=int, default=1000, help="unloading steps after loading/hold")
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--temperature", "--temperature-K", dest="temperature", type=float, default=300.0)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--indenter-radius-A", type=float, default=8.0)
    p.add_argument("--indenter-stiffness", type=float, default=5.0, help="eV/A^3")
    p.add_argument("--initial-depth-A", type=float, default=0.0)
    p.add_argument("--target-depth-A", type=float, default=2.0)
    p.add_argument("--final-unload-depth-A", type=float, default=0.0)
    p.add_argument("--indent-rate-A-ps", type=float, default=None)
    p.add_argument("--unload-rate-A-ps", type=float, default=None)
    p.add_argument("--print-interval", type=int, default=50)
    p.add_argument("--traj-interval", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    return p


def _public_summary(summary: dict) -> dict:
    keys = [
        "system",
        "structure",
        "slab_structure",
        "eam",
        "orientation",
        "replicas",
        "n_atoms",
        "temperature_K",
        "device",
        "loading_steps",
        "hold_steps",
        "unload_steps",
        "dt_ps",
        "indenter_radius_A",
        "indenter_stiffness_ev_A3",
        "initial_depth_A",
        "target_depth_A",
        "max_depth_A",
        "hold_depth_A",
        "final_unload_depth_A",
        "bottom_fixed_thickness_A",
        "indent_rate_A_ps",
        "unload_rate_A_ps",
        "equil_steps",
        "max_load_nN",
        "residual_depth_A",
        "unloading_stiffness_nN_per_A",
        "work_loading",
        "work_unloading",
        "plastic_work_fraction",
        "contact_area_A2",
        "hardness_GPa",
        "hardness_method",
        "max_contact_atoms",
        "pop_in_detected",
        "pop_in_depth_A",
        "pop_in_load_nN",
        "pop_in_reason",
        "max_temperature_K",
        "mean_temperature_K",
        "final_load_nN",
        "peak_load_depth_A",
        "no_nan",
        "plasticity_indicator_available",
        "plasticity_indicator",
        "plasticity_indicator_note",
        "notes",
        "output_dir",
        "csv",
        "legacy_csv",
        "plot",
        "pop_in_plot",
        "report",
        "traj",
        "snapshots_dir",
        "snapshots_png_dir",
        "snapshots",
        "snapshots_png",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


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


def _contact_center_height(coords, axis_unit, lateral_origin, radius: float) -> float:
    heights = coords @ axis_unit
    lateral = coords - heights.unsqueeze(1) * axis_unit
    rho = torch.linalg.norm(lateral - lateral_origin.unsqueeze(0), dim=1)
    inside = rho < float(radius)
    if not bool(inside.any().item()):
        return float(heights.max().item()) + float(radius)
    clearance = torch.sqrt(torch.clamp(float(radius) ** 2 - rho[inside].pow(2), min=0.0))
    return float((heights[inside] + clearance).max().item())


def _write_traj_frame(path: Path, coords: torch.Tensor, atom_types: list[str], comment: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{coords.shape[0]}\n")
        f.write(f"{comment}\n")
        coords_cpu = coords.detach().cpu().tolist()
        for atom_type, xyz in zip(atom_types, coords_cpu):
            f.write(f"{atom_type} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\n")


def _snapshot_comment(snapshot: dict, temperature_k: float) -> str:
    return (
        f"label={snapshot['label']} step={snapshot['step']} time_ps={snapshot['time_ps']:.6f} "
        f"phase={snapshot['phase']} depth={snapshot['depth_A']:.6f} load={snapshot['load_nN']:.6f} "
        f"temperature={temperature_k:.2f}K"
    )


def _write_snapshot_png(path: Path, snapshot: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = snapshot["coords"].detach().cpu()
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.scatter(coords[:, 0], coords[:, 2], s=1.0, c="#303030", alpha=0.55, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (A)")
    ax.set_ylabel("z (A)")
    ax.set_title(
        (
            f"{snapshot['label']} phase={snapshot['phase']} depth={snapshot['depth_A']:.3f} A\n"
            f"load={snapshot['load_nN']:.3f} nN, time={snapshot['time_ps']:.3f} ps"
        ),
        fontsize=9,
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_indent_snapshots(
    output_dir: Path,
    snapshots: dict[str, dict],
    atom_types: list[str],
    temperature_k: float,
) -> tuple[dict[str, str], dict[str, str]]:
    xyz_dir = output_dir / "snapshots"
    png_dir = output_dir / "snapshots_png"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    for old_path in xyz_dir.glob("*.xyz"):
        old_path.unlink()
    for old_path in png_dir.glob("*.png"):
        old_path.unlink()
    xyz_paths = {}
    png_paths = {}
    order = ["initial", "max_depth", "max_load", "unload_start", "final", "pop_in"]
    for key in order:
        snapshot = snapshots.get(key)
        if snapshot is None:
            continue
        xyz_path = xyz_dir / f"{key}.xyz"
        png_path = png_dir / f"{key}.png"
        _write_traj_frame(xyz_path, snapshot["coords"], atom_types, _snapshot_comment(snapshot, temperature_k))
        _write_snapshot_png(png_path, snapshot)
        xyz_paths[key] = str(xyz_path)
        png_paths[key] = str(png_path)
    return xyz_paths, png_paths


def run_w_indent(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / f"orientation_{args.orientation}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.steps = min(args.steps, 60)
        args.equil_steps = min(args.equil_steps, 5)
        args.hold_steps = min(args.hold_steps, 10)
        args.unload_steps = min(max(args.unload_steps, 10), 30)
        args.print_interval = min(args.print_interval, 10)
        args.target_depth_A = min(args.target_depth_A, 0.08)
        args.final_unload_depth_A = min(args.final_unload_depth_A, 0.02)
        if args.hold_depth_A is not None:
            args.hold_depth_A = min(args.hold_depth_A, args.target_depth_A)
        if args.indent_rate_A_ps is not None:
            args.indent_rate_A_ps = min(args.indent_rate_A_ps, 0.5)
        if args.unload_rate_A_ps is not None:
            args.unload_rate_A_ps = min(args.unload_rate_A_ps, 0.5)
        if args.replicas is None:
            args.replicas = "3,3,3"

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
    lateral_center = lateral_center - torch.dot(lateral_center, axis_unit) * axis_unit
    top_layer = projected >= top - 0.35
    top_coords = mol.coordinates[top_layer]
    top_projected = projected[top_layer]
    top_perp = top_coords - top_projected.unsqueeze(1) * axis_unit
    top_dist = torch.linalg.norm(top_perp - lateral_center.unsqueeze(0), dim=1)
    anchor_local = int(torch.argmin(top_dist).item())
    anchor_perp = top_perp[anchor_local].detach().clone()
    anchor_height = float(top_projected[anchor_local].item())

    contact_center_height = _contact_center_height(
        mol.coordinates,
        axis_unit,
        anchor_perp,
        args.indenter_radius_A,
    )

    def center_at_depth(depth_A: float) -> torch.Tensor:
        normal_position = contact_center_height - float(depth_A)
        return anchor_perp + normal_position * axis_unit

    if args.target_depth_A <= args.initial_depth_A:
        raise ValueError("target-depth-A must be larger than initial-depth-A")
    hold_depth_A = float(args.target_depth_A) if args.hold_depth_A is None else float(args.hold_depth_A)
    if hold_depth_A < float(args.initial_depth_A) or hold_depth_A > float(args.target_depth_A):
        raise ValueError("hold-depth-A must be between initial-depth-A and target-depth-A")
    if args.final_unload_depth_A > hold_depth_A:
        raise ValueError("final-unload-depth-A must be <= hold-depth-A/target-depth-A")

    if args.indent_rate_A_ps is None:
        indent_rate = (float(args.target_depth_A) - float(args.initial_depth_A)) / (
            max(1, int(args.steps)) * float(args.dt)
        )
    else:
        indent_rate = float(args.indent_rate_A_ps)
    if int(args.unload_steps) > 0:
        if args.unload_rate_A_ps is None:
            unload_rate = (hold_depth_A - float(args.final_unload_depth_A)) / (
                max(1, int(args.unload_steps)) * float(args.dt)
            )
        else:
            unload_rate = float(args.unload_rate_A_ps)
    else:
        unload_rate = 0.0

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

    if int(args.equil_steps) > 0:
        indenter.set_center(center_at_depth(-0.5))
        for eq_step in range(int(args.equil_steps)):
            eq_out = model()
            with torch.no_grad():
                mol.coordinates[fixed_mask] = fixed_positions
                mol.atom_velocities[fixed_mask] = 0.0
                mol.update_coordinates(mol.coordinates)
                mol.last_positions = mol.coordinates.detach().clone()
            if (eq_step + 1) % max(1, args.print_interval) == 0:
                print(f"Equil {eq_step + 1}/{args.equil_steps}: T={float(eq_out['temperature']):.2f} K")
        contact_center_height = _contact_center_height(
            mol.coordinates,
            axis_unit,
            anchor_perp,
            args.indenter_radius_A,
        )

    traj_path = output_dir / "trajectory.xyz"
    if traj_path.exists():
        traj_path.unlink()
    _write_traj_frame(
        traj_path,
        mol.coordinates,
        mol.atom_types,
        "step=0 time_ps=0.000000 phase=equilibrated depth=0.000000 load=0.000000",
    )
    snapshots = {
        "initial": {
            "label": "initial",
            "step": 0,
            "time_ps": 0.0,
            "phase": "equilibrated",
            "depth_A": 0.0,
            "load_nN": 0.0,
            "coords": mol.coordinates.detach().clone(),
        }
    }
    max_depth_seen = float("-inf")
    max_load_seen = float("-inf")
    pop_in_snapshot_saved = False
    previous_loading_depth = None
    previous_loading_load = None

    csv_path = output_dir / "nanoindent_log.csv"
    legacy_csv_path = output_dir / "load_depth.csv"
    fields = [
        "step",
        "time_ps",
        "phase",
        "phase_step",
        "depth_A",
        "command_depth_A",
        "load_nN",
        "indenter_z",
        "temp",
        "pot",
        "kin",
        "total",
        "indenter_force_axis_ev_per_A",
        "contact_center_height_A",
        "indenter_energy_ev",
        "contact_atoms",
        "max_force_ev_per_A",
        "temperature_k",
        "potential_energy_ev",
        "kinetic_energy_ev",
        "total_energy_ev",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f, open(
        legacy_csv_path, "w", encoding="utf-8", newline=""
    ) as legacy_f:
        writer = csv.writer(f)
        legacy_writer = csv.writer(legacy_f)
        writer.writerow(fields)
        legacy_writer.writerow(fields)

        total_steps = int(args.steps) + int(args.hold_steps) + int(args.unload_steps)
        step_global = 0
        load_steps = [("loading", i + 1) for i in range(int(args.steps))]
        hold_steps = [("hold", i + 1) for i in range(int(args.hold_steps))]
        unload_steps = [("unloading", i + 1) for i in range(int(args.unload_steps))]
        for phase, phase_step in load_steps + hold_steps + unload_steps:
            if phase == "loading":
                command_depth = float(args.initial_depth_A) + phase_step * indent_rate * float(args.dt)
                command_depth = min(command_depth, float(args.target_depth_A))
            elif phase == "hold":
                command_depth = hold_depth_A
            else:
                command_depth = hold_depth_A - phase_step * unload_rate * float(args.dt)
                command_depth = max(command_depth, float(args.final_unload_depth_A))
            step_global += 1

            indenter.set_center(center_at_depth(command_depth))
            out = model()

            with torch.no_grad():
                mol.coordinates[fixed_mask] = fixed_positions
                mol.atom_velocities[fixed_mask] = 0.0
                mol.update_coordinates(mol.coordinates)
                mol.last_positions = mol.coordinates.detach().clone()

            center_height = float(torch.dot(indenter.center, axis_unit).item())
            depth = contact_center_height - center_height
            indenter_axis_force = float(torch.dot(indenter.force_on_indenter, axis_unit).item())
            load_nN = max(0.0, indenter_axis_force * _EV_PER_ANG_TO_NN)
            indenter_energy = float(indenter.last_energy.item())
            potential = float(out["energy"].item())
            kinetic = float(out["kinetic_energy"].item())
            total = potential + kinetic
            temp = float(out["temperature"].item())
            max_force = float(torch.linalg.norm(out["forces"], dim=1).max().item())

            time_ps = step_global * float(args.dt)
            row = [
                step_global,
                time_ps,
                phase,
                phase_step,
                depth,
                command_depth,
                load_nN,
                center_height,
                temp,
                potential,
                kinetic,
                total,
                indenter_axis_force,
                contact_center_height,
                indenter_energy,
                int(indenter.contact_atoms),
                max_force,
                temp,
                potential,
                kinetic,
                total,
            ]
            writer.writerow(row)
            legacy_writer.writerow(row)

            snapshot = {
                "label": phase,
                "step": step_global,
                "time_ps": time_ps,
                "phase": phase,
                "depth_A": depth,
                "load_nN": load_nN,
                "coords": mol.coordinates.detach().clone(),
            }
            if depth >= max_depth_seen:
                max_depth_seen = depth
                snapshots["max_depth"] = {**snapshot, "label": "max_depth"}
            if load_nN >= max_load_seen:
                max_load_seen = load_nN
                snapshots["max_load"] = {**snapshot, "label": "max_load"}
            if phase == "unloading" and "unload_start" not in snapshots:
                snapshots["unload_start"] = {**snapshot, "label": "unload_start"}
                _write_traj_frame(
                    traj_path,
                    mol.coordinates,
                    mol.atom_types,
                    f"step={step_global} time_ps={time_ps:.6f} phase=unloading_start depth={depth:.6f} load={load_nN:.6f}",
                )
            if (
                phase == "loading"
                and not pop_in_snapshot_saved
                and previous_loading_depth is not None
                and previous_loading_load is not None
                and previous_loading_load - load_nN > max(0.05, 0.03 * max(max_load_seen, 1.0e-12))
            ):
                snapshots["pop_in"] = {**snapshot, "label": "pop_in"}
                pop_in_snapshot_saved = True
            if phase == "loading":
                previous_loading_depth = depth
                previous_loading_load = load_nN
            snapshots["final"] = {**snapshot, "label": "final"}

            if int(args.traj_interval) > 0 and step_global % max(1, int(args.traj_interval)) == 0:
                _write_traj_frame(
                    traj_path,
                    mol.coordinates,
                    mol.atom_types,
                    f"step={step_global} time_ps={time_ps:.6f} phase={phase} depth={depth:.6f} load={load_nN:.6f}",
                )

            if step_global % max(1, args.print_interval) == 0:
                print(
                    f"Step {step_global}/{total_steps}: phase={phase}, depth={depth:.4f} A, "
                    f"load={load_nN:.4f} nN, contact={indenter.contact_atoms}, "
                    f"Pot_E={potential:.4f} eV, T={temp:.2f} K"
                )

    if "final" in snapshots:
        final_snapshot = snapshots["final"]
        _write_traj_frame(
            traj_path,
            final_snapshot["coords"],
            mol.atom_types,
            (
                f"step={final_snapshot['step']} time_ps={final_snapshot['time_ps']:.6f} "
                f"phase=final depth={final_snapshot['depth_A']:.6f} load={final_snapshot['load_nN']:.6f}"
            ),
        )

    summary = summarize_load_depth(csv_path, indenter_radius_A=float(args.indenter_radius_A))
    plot_path = output_dir / "load_depth.png"
    popin_plot_path = output_dir / "load_depth_with_popin.png"
    plot_load_depth(csv_path, plot_path)
    plot_load_depth(csv_path, popin_plot_path, pop_in=summary)
    if summary.get("pop_in_detected") and "pop_in" not in snapshots:
        snapshots["pop_in"] = snapshots.get("max_load", snapshots.get("final"))
        if snapshots["pop_in"] is not None:
            snapshots["pop_in"] = {**snapshots["pop_in"], "label": "pop_in"}
    snapshot_xyz, snapshot_png = _write_indent_snapshots(
        output_dir,
        snapshots,
        mol.atom_types,
        float(args.temperature),
    )
    notes = [
        "Hardness uses geometric spherical contact area A = pi(2Rh - h^2); treat it as a workflow-level estimate.",
    ]
    if not bool(summary.get("plasticity_indicator_available")):
        notes.append("Plasticity proxy is not yet available; plasticity is inferred from load-depth response only.")
    if int(summary.get("max_contact_atoms", 0)) < 30:
        notes.append("Maximum contact atoms is below 30, so hardness is not reliable for physical production conclusions.")
    if float(summary.get("max_load_nN", 0.0)) > 1.0e-12:
        final_ratio = float(summary.get("final_load_nN", 0.0)) / float(summary.get("max_load_nN", 1.0))
        if final_ratio > 0.1:
            notes.append("Final unloading load is more than 10% of peak load; indenter contact may remain at the final depth.")
    if summary.get("mean_temperature_K") is not None:
        if abs(float(summary["mean_temperature_K"]) - float(args.temperature)) > 15.0:
            notes.append("Mean reported temperature differs from the target by more than 15 K; fixed bottom atoms can lower the global temperature statistic.")
    summary.update(
        {
            "system": "pure W",
            "structure": str(args.structure),
            "slab_structure": str(structure_path),
            "eam": str(args.eam),
            "orientation": str(args.orientation),
            "replicas": list(replicas) if replicas is not None else None,
            "n_atoms": int(mol.coordinates.shape[0]),
            "loading_steps": int(args.steps),
            "steps": int(args.steps),
            "hold_steps": int(args.hold_steps),
            "unload_steps": int(args.unload_steps),
            "dt_ps": float(args.dt),
            "temperature_K": float(args.temperature),
            "temperature_k": float(args.temperature),
            "indenter_radius_A": float(args.indenter_radius_A),
            "indenter_stiffness_ev_A3": float(args.indenter_stiffness),
            "initial_depth_A": float(args.initial_depth_A),
            "target_depth_A": float(args.target_depth_A),
            "hold_depth_A": float(hold_depth_A),
            "final_unload_depth_A": float(args.final_unload_depth_A),
            "bottom_fixed_thickness_A": float(args.bottom_thickness_A),
            "indent_rate_A_ps": float(indent_rate),
            "unload_rate_A_ps": float(unload_rate),
            "equil_steps": int(args.equil_steps),
            "anchor_height_A": float(anchor_height),
            "anchor_lateral_distance_A": float(top_dist[anchor_local].item()),
            "contact_center_height_A": float(contact_center_height),
            "output_dir": str(output_dir),
            "csv": str(csv_path),
            "legacy_csv": str(legacy_csv_path),
            "plot": str(plot_path),
            "pop_in_plot": str(popin_plot_path),
            "snapshots": snapshot_xyz,
            "snapshots_png": snapshot_png,
            "snapshots_dir": str(output_dir / "snapshots"),
            "snapshots_png_dir": str(output_dir / "snapshots_png"),
            "traj": str(traj_path),
            "device": str(device),
            "smoke": bool(args.smoke),
            "notes": notes,
        }
    )
    report_path = output_dir / "report.md"
    summary["report"] = write_indentation_report(summary, report_path)

    summary_path = output_dir / "summary.json"
    file_summary = _public_summary(summary)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(file_summary, f, indent=2)

    print(f"W indentation completed. Data: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(f"Plot: {plot_path}")
    print(f"Plot with pop-in: {popin_plot_path}")
    print(f"Trajectory: {traj_path}")
    if args.smoke:
        if int(summary.get("max_contact_atoms", 0)) <= 0:
            raise AssertionError("indentation smoke test requires at least one contact atom")
        if not bool(summary.get("no_nan", False)):
            raise AssertionError("indentation smoke test requires finite log values")
        if float(summary.get("max_load_nN", 0.0)) <= 0.0:
            raise AssertionError("indentation smoke test requires positive load")
        if not bool(summary.get("loading_trend_pass", False)):
            raise AssertionError("loading load should generally increase with depth")
        if int(args.unload_steps) > 0 and not bool(summary.get("unloading_trend_pass", False)):
            raise AssertionError("unloading load should generally decrease")
        print("SMOKE TEST PASS")
    return summary


def main():
    args = _build_parser().parse_args()
    run_w_indent(args)


if __name__ == "__main__":
    main()
