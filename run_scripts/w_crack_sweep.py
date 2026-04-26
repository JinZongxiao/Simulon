import argparse
import csv
import json
import os
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.w_crack import _build_parser as _build_crack_parser
from run_scripts.w_crack import run_w_crack


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("list cannot be empty")
    return values


def _parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("list cannot be empty")
    return values


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a W crack propagation acceptance parameter sweep")
    p.add_argument("--structure", default=None)
    p.add_argument("--eam", default=None)
    p.add_argument("--output-dir", default=str(_project_root() / "run_output" / "w_crack_sweep"))
    p.add_argument("--box-length", type=float, default=16.0)
    p.add_argument("--orientation", choices=("100", "110", "111", "custom"), default="100")
    p.add_argument("--replicas", default=None)
    p.add_argument("--temperatures", default="100")
    p.add_argument("--crack-half-lengths-A", default="28.0")
    p.add_argument("--target-strains", default="0.10")
    p.add_argument("--grip-thicknesses-A", default="5.0")
    p.add_argument("--steps-list", default="15000")
    p.add_argument("--equil-steps-list", default="2000")
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--crack-gap-A", type=float, default=2.5)
    p.add_argument("--crack-open-threshold-A", type=float, default=1.0)
    p.add_argument("--crack-length-bins", type=int, default=240)
    p.add_argument("--traj-interval", type=int, default=1000)
    p.add_argument("--print-interval", type=int, default=1000)
    p.add_argument("--max-cases", type=int, default=0, help="0 means run all generated cases")
    p.add_argument("--smoke", action="store_true")
    return p


def _case_name(temp, half_length, target_strain, grip, steps, equil_steps) -> str:
    return (
        f"T_{int(round(temp))}K_a_{half_length:g}_strain_{target_strain:g}_"
        f"grip_{grip:g}_steps_{steps}_eq_{equil_steps}"
    ).replace(".", "p")


def _write_sweep_csv(rows: list[dict], csv_path: Path) -> str:
    fields = [
        "case",
        "temperature_k",
        "crack_half_length_A",
        "target_strain",
        "grip_thickness_A",
        "steps",
        "equil_steps",
        "classification",
        "stress_drop_ratio",
        "max_cmod_A",
        "max_crack_extension_A",
        "significant_crack_propagation_pass",
        "physics_acceptance_pass",
        "peak_stress_at_final_step",
        "geometry_warning",
        "summary_path",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(csv_path)


def run_w_crack_sweep(args) -> dict:
    temps = _parse_float_list(args.temperatures)
    half_lengths = _parse_float_list(args.crack_half_lengths_A)
    strains = _parse_float_list(args.target_strains)
    grips = _parse_float_list(args.grip_thicknesses_A)
    steps_values = _parse_int_list(args.steps_list)
    equil_values = _parse_int_list(args.equil_steps_list)
    if args.smoke:
        temps = [100.0]
        half_lengths = [4.5]
        strains = [0.003]
        grips = [3.0]
        steps_values = [30]
        equil_values = [5]

    cases = list(product(temps, half_lengths, strains, grips, steps_values, equil_values))
    if args.max_cases > 0:
        cases = cases[: int(args.max_cases)]

    root_dir = Path(args.output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    crack_parser = _build_crack_parser()
    rows = []
    for temp, half_length, target_strain, grip, steps, equil_steps in cases:
        name = _case_name(temp, half_length, target_strain, grip, steps, equil_steps)
        crack_args = crack_parser.parse_args([])
        crack_args.output_dir = str(root_dir / name)
        crack_args.orientation = args.orientation
        crack_args.box_length = float(args.box_length)
        crack_args.temperature = float(temp)
        crack_args.crack_half_length_A = float(half_length)
        crack_args.crack_gap_A = float(args.crack_gap_A)
        crack_args.crack_open_threshold_A = float(args.crack_open_threshold_A)
        crack_args.crack_length_bins = int(args.crack_length_bins)
        crack_args.grip_thickness_A = float(grip)
        crack_args.target_strain = float(target_strain)
        crack_args.steps = int(steps)
        crack_args.equil_steps = int(equil_steps)
        crack_args.dt = float(args.dt)
        crack_args.gamma = float(args.gamma)
        crack_args.traj_interval = int(args.traj_interval)
        crack_args.print_interval = int(args.print_interval)
        crack_args.smoke = bool(args.smoke)
        if args.replicas:
            crack_args.replicas = str(args.replicas)
        if args.structure:
            crack_args.structure = str(args.structure)
        if args.eam:
            crack_args.eam = str(args.eam)

        print(f"Running crack sweep case: {name}")
        summary = run_w_crack(crack_args)
        rows.append(
            {
                "case": name,
                "temperature_k": temp,
                "crack_half_length_A": half_length,
                "target_strain": target_strain,
                "grip_thickness_A": grip,
                "steps": steps,
                "equil_steps": equil_steps,
                "classification": summary.get("classification", "invalid"),
                "stress_drop_ratio": summary.get("stress_drop_ratio", 0.0),
                "max_cmod_A": summary.get("max_cmod_A", 0.0),
                "max_crack_extension_A": summary.get("max_crack_extension_A", 0.0),
                "significant_crack_propagation_pass": summary.get(
                    "significant_crack_propagation_pass", False
                ),
                "physics_acceptance_pass": summary.get("physics_acceptance_pass", False),
                "peak_stress_at_final_step": summary.get("peak_stress_at_final_step", True),
                "geometry_warning": summary.get("geometry_warning"),
                "summary_path": summary.get("summary_json", ""),
            }
        )

    csv_path = root_dir / "crack_sweep_summary.csv"
    json_path = root_dir / "crack_sweep_summary.json"
    _write_sweep_csv(rows, csv_path)
    best = max(rows, key=lambda row: float(row["max_crack_extension_A"]), default=None)
    result = {
        "n_cases": len(rows),
        "csv": str(csv_path),
        "best_case": best,
        "significant_propagation_cases": [
            row for row in rows if bool(row["significant_crack_propagation_pass"])
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Crack sweep summary: {csv_path}")
    print(f"Crack sweep json: {json_path}")
    if args.smoke:
        print("SMOKE TEST PASS")
    return result


def main():
    args = _build_parser().parse_args()
    run_w_crack_sweep(args)


if __name__ == "__main__":
    main()
