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


EV_ANG3_TO_BAR = 1_602_176.6208


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths():
    root = _project_root()
    return (
        root / "run_data" / "W" / "WRe_YC2.eam.fs",
        root / "run_output" / "w_tensile_diagnostics",
    )


def _build_parser() -> argparse.ArgumentParser:
    eam_default, out_default = _default_paths()
    p = argparse.ArgumentParser(description="Diagnose W tensile engine/protocol behavior")
    p.add_argument("--eam", default=str(eam_default))
    p.add_argument("--output-dir", default=str(out_default))
    p.add_argument("--orientation", choices=("100", "110", "111"), default="100")
    p.add_argument("--replicas", default="6,6,6")
    p.add_argument("--lattice-param", type=float, default=3.1652)
    p.add_argument("--axis", choices=("x", "y", "z"), default="x")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--hold-steps", type=int, default=500)
    p.add_argument("--tensile-steps", type=int, default=1000)
    p.add_argument("--strain-rate", type=float, default=0.001)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--barostat-tau", type=float, default=0.2)
    p.add_argument("--barostat-gamma", type=float, default=2.0)
    p.add_argument("--barostat-compressibility-bar-inv", type=float, default=3.2e-7)
    p.add_argument("--barostat-pressure-tolerance-bar", type=float, default=25.0)
    p.add_argument("--print-interval", type=int, default=250)
    p.add_argument("--disable-extension", action="store_true")
    return p


def _parse_replicas(value: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3 or any(x <= 0 for x in parts):
        raise ValueError(f"invalid replicas={value}")
    return tuple(parts)


def _axis_to_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def _make_structure(args, output_dir: Path) -> tuple[str, torch.Tensor]:
    coords, box_vectors = generate_oriented_bcc_w(
        lattice_param=args.lattice_param,
        orientation=args.orientation,
        replicas=_parse_replicas(args.replicas),
    )
    path = output_dir / f"W_{args.orientation}_diagnostic.xyz"
    write_xyz(
        path,
        coords,
        atom_type="W",
        comment=f"W tensile diagnostic orientation={args.orientation} replicas={args.replicas}",
    )
    return str(path), box_vectors


def _build_model(
    args,
    parser,
    structure_path: str,
    box_vectors: torch.Tensor,
    ensemble: str,
    barostat=None,
    init_velocity: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mol = AtomFileReader(
        filename=structure_path,
        box_length=float(torch.linalg.norm(box_vectors[0]).item()),
        cutoff=parser.cutoff,
        device=device,
        skin_thickness=1.0,
        is_mlp=True,
        box_vectors=box_vectors,
    )
    if ensemble.upper() == "NVE" and init_velocity:
        mol.set_maxwell_boltzmann_velocity(torch.tensor(float(args.temperature), device=device))
    ff = EAMForce(parser, mol, use_extension=not bool(args.disable_extension))
    sb = SumBackboneInterface([ff], mol)
    integ = VerletIntegrator(
        mol,
        dt=float(args.dt),
        ensemble=ensemble,
        temperature=(args.temperature, args.temperature) if ensemble.upper() in ("NVT", "NPT") else None,
        gamma=args.gamma if ensemble.upper() in ("NVT", "NPT") else None,
    )
    return BaseModel(sb, integ, mol, barostat=barostat), mol


def _kinetic_tensor(model) -> torch.Tensor:
    vel = model.molecular.atom_velocities
    masses = model.Integrator.atom_mass[:, :1]
    return torch.einsum("ni,nj->ij", masses * vel, vel)


def _measure(model) -> dict:
    out = model.sum_bone()
    kinetic_tensor = _kinetic_tensor(model)
    kinetic_energy = (0.5 * model.Integrator.atom_mass * model.molecular.atom_velocities.pow(2)).sum()
    temperature = (2.0 / 3.0) * kinetic_energy / (
        model.molecular.atom_count * model.Integrator.BOLTZMAN
    )
    sigma_tensor = ((kinetic_tensor + out["virial_tensor"].to(kinetic_tensor.dtype)) / float(model.molecular.box.volume)) * EV_ANG3_TO_BAR
    return {
        "energy": float(out["energy"]),
        "kinetic": float(kinetic_energy),
        "total": float(out["energy"] + kinetic_energy),
        "temperature": float(temperature),
        "sigma": sigma_tensor.detach().clone(),
        "virial": float(out["virial"]),
    }


def _axis_stress(sigma_tensor: torch.Tensor, box, axis_idx: int) -> torch.Tensor:
    axes = box.H.to(device=sigma_tensor.device, dtype=sigma_tensor.dtype)
    axes = axes / torch.linalg.norm(axes, dim=1, keepdim=True).clamp_min(1e-12)
    return torch.einsum("ai,ij,aj->a", axes, sigma_tensor, axes)[axis_idx]


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x0, x1 = xs[0], xs[-1]
    if abs(x1 - x0) < 1e-12:
        return 0.0
    return (ys[-1] - ys[0]) / (x1 - x0)


def run_static_elastic(args, parser, structure_path, box_vectors, output_dir: Path) -> dict:
    axis_idx = _axis_to_index(args.axis)
    strains = [0.0, 0.001, 0.002, 0.005, 0.010]
    rows = []
    base_stress = None
    for strain in strains:
        model, mol = _build_model(args, parser, structure_path, box_vectors, "NVE", init_velocity=False)
        loader = UniaxialTensileLoader(mol, axis=axis_idx, strain_rate=strain, lateral_mode="fixed")
        if strain > 0.0:
            loader.step(1.0)
        state = _measure(model)
        stress_abs = float(_axis_stress(state["sigma"], mol.box, axis_idx))
        if base_stress is None:
            base_stress = stress_abs
        tension = -(stress_abs - base_stress)
        rows.append(
            {
                "strain": strain,
                "tension_bar": tension,
                "stress_abs_bar": stress_abs,
                "energy_ev": state["energy"],
                "temperature_k": state["temperature"],
            }
        )
    slope = _linear_slope([r["strain"] for r in rows[:4]], [r["tension_bar"] for r in rows[:4]])
    csv_path = _write_csv(
        output_dir / "static_elastic.csv",
        ["strain", "tension_bar", "stress_abs_bar", "energy_ev", "temperature_k"],
        rows,
    )
    return {"csv": csv_path, "elastic_slope_bar": slope, "max_tension_bar": max(r["tension_bar"] for r in rows)}


def run_hold(args, parser, structure_path, box_vectors, output_dir: Path, ensemble: str) -> dict:
    model, mol = _build_model(args, parser, structure_path, box_vectors, ensemble)
    rows = []
    initial = _measure(model)
    for step in range(int(args.hold_steps) + 1):
        if step > 0:
            out = model()
            total = float(out["energy"] + out["kinetic_energy"])
            temp = float(out["temperature"])
            pot = float(out["energy"])
            kin = float(out["kinetic_energy"])
        else:
            total = initial["total"]
            temp = initial["temperature"]
            pot = initial["energy"]
            kin = initial["kinetic"]
        if step == 0 or step % max(1, int(args.print_interval)) == 0 or step == int(args.hold_steps):
            rows.append({"step": step, "time_ps": step * args.dt, "potential_ev": pot, "kinetic_ev": kin, "total_ev": total, "temperature_k": temp})
    totals = [r["total_ev"] for r in rows]
    temps = [r["temperature_k"] for r in rows]
    drift = 0.0 if abs(totals[0]) < 1e-12 else (totals[-1] - totals[0]) / abs(totals[0])
    csv_path = _write_csv(
        output_dir / f"{ensemble.lower()}_hold.csv",
        ["step", "time_ps", "potential_ev", "kinetic_ev", "total_ev", "temperature_k"],
        rows,
    )
    return {"csv": csv_path, "energy_drift_fraction": drift, "temperature_mean_k": sum(temps) / len(temps), "temperature_final_k": temps[-1]}


def _make_lateral_barostat(args, mol):
    axis_idx = _axis_to_index(args.axis)
    control_axes = [True, True, True]
    control_axes[axis_idx] = False
    return AnisotropicNPTBarostat(
        molecular=mol,
        target_pressure_bar=[0.0, 0.0, 0.0],
        temperature_k=args.temperature,
        tau_p=args.barostat_tau,
        gamma_p=args.barostat_gamma,
        control_axes=tuple(control_axes),
        compressibility_bar_inv=args.barostat_compressibility_bar_inv,
        pressure_tolerance_bar=args.barostat_pressure_tolerance_bar,
    )


def _refresh_force_cache(model) -> None:
    out = model.sum_bone()
    model.force_cache = out["forces"]
    model.energy_cache = out["energy"]
    model.virial_cache = out.get("virial", torch.tensor(0.0, device=model.molecular.device))
    model.virial_tensor_cache = out.get(
        "virial_tensor", torch.zeros(3, 3, device=model.molecular.device)
    )
    model._first_call = False


def run_tensile_case(
    args,
    parser,
    structure_path,
    box_vectors,
    output_dir: Path,
    lateral_mode: str,
    refresh_force_after_load: bool = False,
) -> dict:
    axis_idx = _axis_to_index(args.axis)
    model, mol = _build_model(args, parser, structure_path, box_vectors, "NVT")
    if lateral_mode == "stress_free":
        model.barostat = _make_lateral_barostat(args, mol)
        loader_mode = "fixed"
    else:
        loader_mode = "fixed"
    loader = UniaxialTensileLoader(mol, axis=axis_idx, strain_rate=args.strain_rate, lateral_mode=loader_mode)
    base = _measure(model)
    base_axis = float(_axis_stress(base["sigma"], mol.box, axis_idx))
    rows = [{"step": 0, "strain": 0.0, "tension_bar": 0.0, "temperature_k": base["temperature"], "box_x": float(mol.box.lengths[0]), "box_y": float(mol.box.lengths[1]), "box_z": float(mol.box.lengths[2])}]
    label = lateral_mode + ("_force_refresh" if refresh_force_after_load else "_current_order")
    for step in range(1, int(args.tensile_steps) + 1):
        strain = loader.step(args.dt)
        if refresh_force_after_load:
            _refresh_force_cache(model)
        out = model()
        sigma = ((_kinetic_tensor(model) + out["virial_tensor"].to(_kinetic_tensor(model).dtype)) / float(mol.box.volume)) * EV_ANG3_TO_BAR
        stress_abs = float(_axis_stress(sigma, mol.box, axis_idx))
        tension = -(stress_abs - base_axis)
        if step % max(1, int(args.print_interval)) == 0 or step == int(args.tensile_steps):
            print(f"{label} tensile {step}/{args.tensile_steps}: strain={strain:.6g}, tension={tension:.2f} bar, T={float(out['temperature']):.2f} K")
        rows.append({"step": step, "strain": strain, "tension_bar": tension, "temperature_k": float(out["temperature"]), "box_x": float(mol.box.lengths[0]), "box_y": float(mol.box.lengths[1]), "box_z": float(mol.box.lengths[2])})
    slope = _linear_slope([r["strain"] for r in rows], [r["tension_bar"] for r in rows])
    csv_path = _write_csv(
        output_dir / f"tensile_{label}.csv",
        ["step", "strain", "tension_bar", "temperature_k", "box_x", "box_y", "box_z"],
        rows,
    )
    return {"csv": csv_path, "elastic_slope_bar": slope, "final_tension_bar": rows[-1]["tension_bar"], "final_strain": rows[-1]["strain"], "temperature_final_k": rows[-1]["temperature_k"]}


def _write_report(summary: dict, output_dir: Path) -> str:
    lines = [
        "# W Tensile Diagnostic Report",
        "",
        "## Purpose",
        "Separate EAM/integrator behavior from tensile protocol effects.",
        "",
        "## Key Results",
        f"- Static elastic slope: {summary['static_elastic']['elastic_slope_bar']:.3e} bar",
        f"- NVE energy drift: {summary['nve_hold']['energy_drift_fraction']:.3e}",
        f"- NVT final temperature: {summary['nvt_hold']['temperature_final_k']:.2f} K",
        f"- Fixed lateral current-order slope: {summary['tensile_fixed_current_order']['elastic_slope_bar']:.3e} bar",
        f"- Fixed lateral force-refresh slope: {summary['tensile_fixed_force_refresh']['elastic_slope_bar']:.3e} bar",
        f"- Stress-free current-order slope: {summary['tensile_stress_free_current_order']['elastic_slope_bar']:.3e} bar",
        f"- Stress-free force-refresh slope: {summary['tensile_stress_free_force_refresh']['elastic_slope_bar']:.3e} bar",
        "",
        "## Interpretation",
    ]
    nve_ok = abs(summary["nve_hold"]["energy_drift_fraction"]) < 1e-3
    static_slope = summary["static_elastic"]["elastic_slope_bar"]
    fixed_slope = summary["tensile_fixed_force_refresh"]["elastic_slope_bar"]
    stress_free_slope = summary["tensile_stress_free_force_refresh"]["elastic_slope_bar"]
    current_fixed_slope = summary["tensile_fixed_current_order"]["elastic_slope_bar"]
    if nve_ok and static_slope > 0.0:
        lines.append("- Basic EAM force/integration behavior looks usable in this diagnostic window.")
    else:
        lines.append("- Basic EAM force/integration behavior needs attention before tuning tensile protocol.")
    if fixed_slope > 0.0 and stress_free_slope > 0.0 and abs(stress_free_slope - fixed_slope) / max(abs(fixed_slope), 1.0) > 0.3:
        lines.append("- Lateral stress control changes the tensile response strongly; focus on barostat/protocol before interpreting production curves.")
    else:
        lines.append("- Fixed and stress-free tensile responses are not strongly separated in this short diagnostic.")
    if current_fixed_slope * fixed_slope < 0.0:
        lines.append("- Force-cache refresh changes the sign of the tensile slope; external loading must refresh forces before integration.")
    path = output_dir / "diagnostic_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def run_diagnostics(args) -> dict:
    torch.manual_seed(1234)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = EAMParser(filepath=args.eam, device=device)
    structure_path, box_vectors = _make_structure(args, output_dir)

    summary = {
        "structure": structure_path,
        "eam": str(args.eam),
        "orientation": args.orientation,
        "replicas": args.replicas,
        "temperature_k": float(args.temperature),
        "dt_ps": float(args.dt),
        "device": str(device),
        "use_extension": not bool(args.disable_extension),
    }
    summary["static_elastic"] = run_static_elastic(args, parser, structure_path, box_vectors, output_dir)
    summary["nve_hold"] = run_hold(args, parser, structure_path, box_vectors, output_dir, "NVE")
    summary["nvt_hold"] = run_hold(args, parser, structure_path, box_vectors, output_dir, "NVT")
    summary["tensile_fixed_current_order"] = run_tensile_case(
        args, parser, structure_path, box_vectors, output_dir, "fixed", refresh_force_after_load=False
    )
    summary["tensile_fixed_force_refresh"] = run_tensile_case(
        args, parser, structure_path, box_vectors, output_dir, "fixed", refresh_force_after_load=True
    )
    summary["tensile_stress_free_current_order"] = run_tensile_case(
        args, parser, structure_path, box_vectors, output_dir, "stress_free", refresh_force_after_load=False
    )
    summary["tensile_stress_free_force_refresh"] = run_tensile_case(
        args, parser, structure_path, box_vectors, output_dir, "stress_free", refresh_force_after_load=True
    )
    numeric_values = []
    for section_name in (
        "static_elastic",
        "nve_hold",
        "nvt_hold",
        "tensile_fixed_current_order",
        "tensile_fixed_force_refresh",
        "tensile_stress_free_current_order",
        "tensile_stress_free_force_refresh",
    ):
        for value in summary[section_name].values():
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
    summary["no_nan"] = all(torch.isfinite(torch.tensor(numeric_values)).tolist())
    summary["report"] = _write_report(summary, output_dir)

    summary_path = output_dir / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"W tensile diagnostics completed. Summary: {summary_path}")
    print(f"Report: {summary['report']}")
    return summary


def main():
    args = _build_parser().parse_args()
    run_diagnostics(args)


if __name__ == "__main__":
    main()
