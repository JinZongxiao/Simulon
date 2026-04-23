import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from postprocess.dbtt import collect_dbtt_rows, plot_dbtt, summarize_dbtt, write_dbtt_csv
from run_scripts.w_crack import _build_parser as _build_crack_parser
from run_scripts.w_crack import run_w_crack


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_output_dir() -> Path:
    return _project_root() / "run_output" / "w_dbtt"


def _parse_temperatures(value: str) -> list[float]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    temps = [float(x) for x in parts]
    if not temps:
        raise ValueError("temperatures cannot be empty")
    return temps


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a W crack-based DBTT temperature scan")
    p.add_argument("--structure", default=None)
    p.add_argument("--eam", default=None)
    p.add_argument("--temperatures", default="100,200,300,400,500,600")
    p.add_argument("--output-dir", default=str(_default_output_dir()))
    p.add_argument("--box-length", type=float, default=16.0)
    p.add_argument("--orientation", choices=("100", "110", "111", "custom"), default="100")
    p.add_argument("--replicas", default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--equil-steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--temperature-scale", type=float, default=1.0)
    p.add_argument("--crack-half-length-A", type=float, default=8.0)
    p.add_argument("--crack-gap-A", type=float, default=1.2)
    p.add_argument("--grip-thickness-A", type=float, default=3.0)
    p.add_argument("--target-strain", type=float, default=0.02)
    p.add_argument("--opening-rate-A-ps", type=float, default=None)
    p.add_argument("--smoke", action="store_true")
    return p


def run_w_dbtt_scan(args) -> dict:
    crack_parser = _build_crack_parser()
    crack_args = crack_parser.parse_args([])
    crack_args.output_dir = str(Path(args.output_dir))
    crack_args.orientation = args.orientation
    crack_args.smoke = bool(args.smoke)
    crack_args.box_length = float(args.box_length)
    crack_args.steps = int(args.steps)
    crack_args.equil_steps = int(args.equil_steps)
    crack_args.dt = float(args.dt)
    crack_args.gamma = float(args.gamma)
    crack_args.crack_half_length_A = float(args.crack_half_length_A)
    crack_args.crack_gap_A = float(args.crack_gap_A)
    crack_args.grip_thickness_A = float(args.grip_thickness_A)
    crack_args.target_strain = float(args.target_strain)
    crack_args.opening_rate_A_ps = args.opening_rate_A_ps
    if args.replicas:
        crack_args.replicas = str(args.replicas)
    if args.structure:
        crack_args.structure = str(args.structure)
    if args.eam:
        crack_args.eam = str(args.eam)
    if args.smoke:
        temperatures = [200.0, 400.0]
        crack_args.steps = 20
        crack_args.equil_steps = 5
        crack_args.target_strain = 0.002
        crack_args.replicas = "4,4,3"
    else:
        temperatures = _parse_temperatures(args.temperatures)

    root_dir = Path(args.output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for temp in temperatures:
        crack_args.temperature = float(temp) * float(args.temperature_scale)
        crack_args.output_dir = str(root_dir / f"T_{int(round(temp))}K")
        print(f"Running DBTT crack case: orientation={args.orientation}, T={temp:.1f} K")
        summary = run_w_crack(crack_args)
        runs.append(summary)

    rows = collect_dbtt_rows(root_dir)
    csv_path = root_dir / "dbtt_summary.csv"
    json_path = root_dir / "dbtt_summary.json"
    plot_path = root_dir / "dbtt_summary.png"
    write_dbtt_csv(rows, csv_path)
    plot_dbtt(rows, plot_path)
    summary = summarize_dbtt(rows)
    summary.update(
        {
            "orientation": str(args.orientation),
            "temperatures_k": temperatures,
            "temperature_scale": float(args.temperature_scale),
            "csv": str(csv_path),
            "plot": str(plot_path),
            "runs": [run["output_dir"] for run in runs],
            "smoke": bool(args.smoke),
        }
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"DBTT summary: {csv_path}")
    print(f"DBTT plot: {plot_path}")
    print(f"DBTT json: {json_path}")
    if args.smoke:
        print("SMOKE TEST PASS")
    return summary


def main():
    args = _build_parser().parse_args()
    run_w_dbtt_scan(args)


if __name__ == "__main__":
    main()
