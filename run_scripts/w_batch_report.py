import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.w_crack import _build_parser as _build_crack_parser
from run_scripts.w_crack import run_w_crack
from run_scripts.w_dbtt_scan import _build_parser as _build_dbtt_parser
from run_scripts.w_dbtt_scan import run_w_dbtt_scan
from run_scripts.w_indent import _build_parser as _build_indent_parser
from run_scripts.w_indent import run_w_indent
from run_scripts.w_tensile import _build_parser as _build_tensile_parser
from run_scripts.w_tensile import run_w_tensile


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_output_dir() -> Path:
    return _project_root() / "run_output" / "w_batch_report"


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _replicas_for_orientation(args, orientation: str) -> str | None:
    return {
        "100": args.replicas_100,
        "110": args.replicas_110,
        "111": args.replicas_111,
        "custom": None,
    }.get(orientation)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run selected W mechanics workflows and build a combined report")
    p.add_argument("--workflows", default="tensile,indent,crack,dbtt")
    p.add_argument("--orientations", default="100,110,111")
    p.add_argument("--output-dir", default=str(_default_output_dir()))
    p.add_argument("--structure", default=None)
    p.add_argument("--box-length", type=float, default=16.0)
    p.add_argument("--eam", default=None)
    p.add_argument("--smoke", action="store_true")

    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--gamma", type=float, default=2.0)

    p.add_argument("--replicas-100", default=None)
    p.add_argument("--replicas-110", default=None)
    p.add_argument("--replicas-111", default=None)

    p.add_argument("--tensile-steps", type=int, default=5000)
    p.add_argument("--tensile-strain-rate", type=float, default=0.00005)
    p.add_argument("--tensile-lateral-mode", choices=("fixed", "poisson", "stress-free"), default="stress-free")
    p.add_argument("--tensile-barostat-tau", type=float, default=0.1)
    p.add_argument("--tensile-barostat-gamma", type=float, default=1.0)

    p.add_argument("--indent-steps", type=int, default=5000)
    p.add_argument("--indent-equil-steps", type=int, default=1000)
    p.add_argument("--indent-initial-depth-A", type=float, default=0.0)
    p.add_argument("--indent-target-depth-A", type=float, default=2.0)
    p.add_argument("--indent-radius-A", type=float, default=8.0)
    p.add_argument("--indent-stiffness", type=float, default=5.0)
    p.add_argument("--indent-rate-A-ps", type=float, default=None)

    p.add_argument("--crack-steps", type=int, default=5000)
    p.add_argument("--crack-equil-steps", type=int, default=500)
    p.add_argument("--crack-half-length-A", type=float, default=8.0)
    p.add_argument("--crack-gap-A", type=float, default=1.2)
    p.add_argument("--crack-grip-thickness-A", type=float, default=3.0)
    p.add_argument("--crack-target-strain", type=float, default=0.02)
    p.add_argument("--crack-opening-rate-A-ps", type=float, default=None)

    p.add_argument("--dbtt-temperatures", default="100,200,300,400,500,600")
    p.add_argument("--dbtt-steps", type=int, default=1000)
    p.add_argument("--dbtt-equil-steps", type=int, default=200)
    p.add_argument("--dbtt-target-strain", type=float, default=0.02)
    p.add_argument("--dbtt-opening-rate-A-ps", type=float, default=None)
    return p


def _flatten_summary(workflow: str, orientation: str, summary: dict) -> dict:
    row = {"workflow": workflow, "orientation": orientation}
    for key, value in summary.items():
        if isinstance(value, (str, int, float, bool)):
            row[key] = value
    return row


def _write_report(entries: list[dict], report_dir: Path) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "batch_report.csv"
    json_path = report_dir / "batch_report.json"
    md_path = report_dir / "batch_report.md"

    all_fields = ["workflow", "orientation"]
    extra_fields = sorted({key for entry in entries for key in entry.keys() if key not in all_fields})
    fields = all_fields + extra_fields

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    lines = ["# W Mechanics Batch Report", ""]
    for entry in entries:
        lines.append(f"## {entry['workflow']} | {entry['orientation']}")
        for key in fields:
            if key in ("workflow", "orientation"):
                continue
            if key in entry:
                lines.append(f"- `{key}`: {entry[key]}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "md": str(md_path)}


def run_w_batch_report(args) -> dict:
    workflows = _parse_list(args.workflows)
    orientations = _parse_list(args.orientations)
    root_output = Path(args.output_dir)
    root_output.mkdir(parents=True, exist_ok=True)
    entries = []

    for orientation in orientations:
        replicas = _replicas_for_orientation(args, orientation)

        if "tensile" in workflows:
            tensile_args = _build_tensile_parser().parse_args([])
            tensile_args.orientation = orientation
            tensile_args.output_dir = str(root_output / "tensile")
            tensile_args.box_length = float(args.box_length)
            tensile_args.temperature = float(args.temperature)
            tensile_args.dt = float(args.dt)
            tensile_args.gamma = float(args.gamma)
            tensile_args.steps = int(args.tensile_steps)
            tensile_args.strain_rate = float(args.tensile_strain_rate)
            tensile_args.lateral_mode = str(args.tensile_lateral_mode)
            tensile_args.barostat_tau = float(args.tensile_barostat_tau)
            tensile_args.barostat_gamma = float(args.tensile_barostat_gamma)
            tensile_args.smoke = bool(args.smoke)
            if args.eam:
                tensile_args.eam = str(args.eam)
            if args.structure:
                tensile_args.structure = str(args.structure)
            if replicas:
                tensile_args.replicas = replicas
            print(f"Running tensile: orientation={orientation}")
            tensile_summary = run_w_tensile(tensile_args)
            entries.append(_flatten_summary("tensile", orientation, tensile_summary))

        if "indent" in workflows or "indentation" in workflows:
            indent_args = _build_indent_parser().parse_args([])
            indent_args.orientation = orientation
            indent_args.output_dir = str(root_output / "indent")
            indent_args.box_length = float(args.box_length)
            indent_args.temperature = float(args.temperature)
            indent_args.dt = float(args.dt)
            indent_args.gamma = float(args.gamma)
            indent_args.steps = int(args.indent_steps)
            indent_args.equil_steps = int(args.indent_equil_steps)
            indent_args.initial_depth_A = float(args.indent_initial_depth_A)
            indent_args.target_depth_A = float(args.indent_target_depth_A)
            indent_args.indenter_radius_A = float(args.indent_radius_A)
            indent_args.indenter_stiffness = float(args.indent_stiffness)
            indent_args.indent_rate_A_ps = args.indent_rate_A_ps
            indent_args.smoke = bool(args.smoke)
            if args.eam:
                indent_args.eam = str(args.eam)
            if args.structure:
                indent_args.structure = str(args.structure)
            if replicas:
                indent_args.replicas = replicas
            print(f"Running indentation: orientation={orientation}")
            indent_summary = run_w_indent(indent_args)
            entries.append(_flatten_summary("indent", orientation, indent_summary))

        if "crack" in workflows:
            crack_args = _build_crack_parser().parse_args([])
            crack_args.orientation = orientation
            crack_args.output_dir = str(root_output / "crack")
            crack_args.box_length = float(args.box_length)
            crack_args.temperature = float(args.temperature)
            crack_args.dt = float(args.dt)
            crack_args.gamma = float(args.gamma)
            crack_args.steps = int(args.crack_steps)
            crack_args.equil_steps = int(args.crack_equil_steps)
            crack_args.crack_half_length_A = float(args.crack_half_length_A)
            crack_args.crack_gap_A = float(args.crack_gap_A)
            crack_args.grip_thickness_A = float(args.crack_grip_thickness_A)
            crack_args.target_strain = float(args.crack_target_strain)
            crack_args.opening_rate_A_ps = args.crack_opening_rate_A_ps
            crack_args.smoke = bool(args.smoke)
            if args.eam:
                crack_args.eam = str(args.eam)
            if args.structure:
                crack_args.structure = str(args.structure)
            if replicas:
                crack_args.replicas = replicas
            print(f"Running crack: orientation={orientation}")
            crack_summary = run_w_crack(crack_args)
            entries.append(_flatten_summary("crack", orientation, crack_summary))

        if "dbtt" in workflows:
            dbtt_args = _build_dbtt_parser().parse_args([])
            dbtt_args.orientation = orientation
            dbtt_args.output_dir = str(root_output / "dbtt" / f"orientation_{orientation}")
            dbtt_args.box_length = float(args.box_length)
            dbtt_args.temperatures = str(args.dbtt_temperatures)
            dbtt_args.steps = int(args.dbtt_steps)
            dbtt_args.equil_steps = int(args.dbtt_equil_steps)
            dbtt_args.dt = float(args.dt)
            dbtt_args.gamma = float(args.gamma)
            dbtt_args.target_strain = float(args.dbtt_target_strain)
            dbtt_args.opening_rate_A_ps = args.dbtt_opening_rate_A_ps
            dbtt_args.smoke = bool(args.smoke)
            if args.eam:
                dbtt_args.eam = str(args.eam)
            if args.structure:
                dbtt_args.structure = str(args.structure)
            if replicas:
                dbtt_args.replicas = replicas
            print(f"Running DBTT: orientation={orientation}")
            dbtt_summary = run_w_dbtt_scan(dbtt_args)
            entries.append(_flatten_summary("dbtt", orientation, dbtt_summary))

    report_paths = _write_report(entries, root_output / "report")
    result = {
        "n_entries": len(entries),
        "workflows": workflows,
        "orientations": orientations,
        "output_dir": str(root_output),
        **report_paths,
    }
    print(f"Batch report CSV: {report_paths['csv']}")
    print(f"Batch report JSON: {report_paths['json']}")
    print(f"Batch report MD: {report_paths['md']}")
    if args.smoke:
        print("SMOKE TEST PASS")
    return result


def main():
    args = _build_parser().parse_args()
    run_w_batch_report(args)


if __name__ == "__main__":
    main()
