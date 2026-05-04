from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.build_odsw_dft_dataset import _build_parser as _build_dataset_parser  # noqa: E402
from run_scripts.build_odsw_dft_dataset import run_odsw_dft_dataset  # noqa: E402
from run_scripts.run_dft_qe_batch import _build_parser as _build_qe_batch_parser  # noqa: E402
from run_scripts.run_dft_qe_batch import run_qe_batch  # noqa: E402


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _read_metadata(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _field_count(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "") or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _label_health(label: dict | None, atom_count: int | None) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if not label:
        return False, ["missing_or_invalid_label_json"]
    if label.get("label_ready") is not True:
        problems.append("label_ready_false")
    if label.get("converged") is not True:
        problems.append("not_converged")
    if label.get("job_done") is not True:
        problems.append("job_not_done")
    if label.get("energy_eV") is None:
        problems.append("missing_energy_eV")
    forces = label.get("forces_eV_A")
    if not isinstance(forces, list) or not forces:
        problems.append("missing_forces_eV_A")
    elif atom_count is not None and len(forces) != atom_count:
        problems.append(f"force_count_mismatch_{len(forces)}_vs_{atom_count}")
    stress = label.get("stress_GPa")
    if not isinstance(stress, list) or len(stress) != 3:
        problems.append("missing_stress_GPa")
    if label.get("no_nan") is not True:
        problems.append("nan_or_inf_detected")
    return len(problems) == 0, problems


def audit_closed_loop(dataset_dir: Path) -> dict:
    metadata_csv = dataset_dir / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv}")
    rows = _read_metadata(metadata_csv)
    metadata_base = metadata_csv.parent
    audit_rows: list[dict] = []
    ready_count = 0
    for row in rows:
        task_dir = _resolve_path(row.get("dft_task_dir", ""), metadata_base)
        label_path = task_dir / "dft_label.json"
        label = _read_json(label_path)
        atom_count = int(float(row["atom_count"])) if row.get("atom_count") else None
        ready, problems = _label_health(label, atom_count)
        if ready:
            ready_count += 1
        audit_rows.append(
            {
                "task_id": row.get("task_id", ""),
                "label_source": row.get("label_source", ""),
                "diversity_role": row.get("diversity_role", ""),
                "atom_count": atom_count,
                "label_ready": ready,
                "energy_eV": None if label is None else label.get("energy_eV"),
                "force_count": None if label is None else label.get("force_count"),
                "stress_available": None if label is None else label.get("stress_available"),
                "label": str(label_path),
                "problems": ";".join(problems),
            }
        )

    source_counts = _field_count(rows, "label_source")
    role_counts = _field_count(rows, "diversity_role")
    ready_by_source: dict[str, int] = {}
    for item in audit_rows:
        if item["label_ready"]:
            key = str(item["label_source"] or "unknown")
            ready_by_source[key] = ready_by_source.get(key, 0) + 1
    required_sources = ["pure_w_bulk", "pure_w_defect", "pure_w_surface", "solute_in_w", "ods_interface"]
    missing_sources = [source for source in required_sources if source_counts.get(source, 0) == 0]
    coverage_pass = not missing_sources
    all_labels_ready = ready_count == len(rows) and len(rows) > 0
    closed_loop_pass = coverage_pass and all_labels_ready

    audit_csv = dataset_dir / "dft_label_audit.csv"
    fieldnames = [
        "task_id",
        "label_source",
        "diversity_role",
        "atom_count",
        "label_ready",
        "energy_eV",
        "force_count",
        "stress_available",
        "label",
        "problems",
    ]
    with open(audit_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    report_path = dataset_dir / "closed_loop_report.md"
    report_lines = [
        "# W-Zr-Y-O ODS-W QE Closed-Loop Report",
        "",
        "## Scope",
        "- Chemistry: W-Zr-Y-O, also referred to as W-Y-Zr-O in this workflow.",
        "- Purpose: generate and audit QE DFT labels for the first ODS-W MLP data loop.",
        "- Required labels: total energy, forces, stress, cell, species, positions.",
        "",
        "## Dataset Coverage",
        f"- Total tasks: {len(rows)}",
        f"- Label-ready tasks: {ready_count}",
        f"- Coverage pass: {coverage_pass}",
        f"- All labels ready: {all_labels_ready}",
        f"- Closed-loop pass: {closed_loop_pass}",
        "",
        "## Label Sources",
        *[f"- {key}: {value}" for key, value in source_counts.items()],
        "",
        "## Diversity Roles",
        *[f"- {key}: {value}" for key, value in role_counts.items()],
        "",
        "## Interpretation",
        "- `coverage_pass=true` means the task list contains the minimum W/defect/solute/interface categories needed for a first ODS-W label loop.",
        "- `closed_loop_pass=true` additionally requires every selected task to have a usable `dft_label.json`.",
        "- If labels are not ready, this report is still useful as a queue audit, but it is not yet an MLP-ready dataset.",
        "",
        "## Outputs",
        f"- Metadata: `{metadata_csv}`",
        f"- Label audit: `{audit_csv}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary = {
        "workflow": "odsw_qe_closed_loop",
        "chemistry": "W-Zr-Y-O",
        "chemistry_alias": "W-Y-Zr-O",
        "dataset_dir": str(dataset_dir),
        "metadata_csv": str(metadata_csv),
        "audit_csv": str(audit_csv),
        "report_md": str(report_path),
        "task_count": len(rows),
        "label_ready_count": ready_count,
        "all_labels_ready": all_labels_ready,
        "coverage_pass": coverage_pass,
        "closed_loop_pass": closed_loop_pass,
        "required_label_sources": required_sources,
        "missing_label_sources": missing_sources,
        "label_source_counts": source_counts,
        "diversity_role_counts": role_counts,
    }
    _write_json(dataset_dir / "closed_loop_summary.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the first W-Zr-Y-O ODS-W QE closed loop: export, QE batch, and label audit")
    p.add_argument("--output-dir", default=str(_project_root() / "run_output" / "odsw_qe_closed_loop_WYZrO"))
    p.add_argument("--stage", choices=("all", "export", "batch", "audit"), default="all")
    p.add_argument("--replicas", default="4,4,4")
    p.add_argument("--orientation", choices=("100", "110", "111"), default="100")
    p.add_argument("--lattice-param", type=float, default=3.1652)
    p.add_argument("--max-tasks", type=int, default=16)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--oxide-formulas", default="ABO3,A2B2O7")
    p.add_argument("--particle-radii-A", default="3.0,4.0")
    p.add_argument("--oxide-lattice-params-A", default="4.4,4.8")
    p.add_argument("--interface-clearances-A", default="0.8,1.2")
    p.add_argument("--qe-pseudo-dir", default="/public/home/normal_bgd/J1N/software/pseudopotentials")
    p.add_argument("--qe-ecutwfc", type=float, default=40.0)
    p.add_argument("--qe-ecutrho", type=float, default=320.0)
    p.add_argument("--qe-kmesh", default="1 1 1")
    p.add_argument("--qe-conv-thr", default="1.0d-6")
    p.add_argument("--np", type=int, default=8)
    p.add_argument("--omp", type=int, default=1)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--batch-limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="plan QE batch without launching pw.x")
    p.add_argument("--rerun-completed", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


def _dataset_args(args: argparse.Namespace) -> argparse.Namespace:
    values = [
        "--campaign",
        "pilot_diverse",
        "--replicas",
        args.replicas,
        "--orientation",
        args.orientation,
        "--lattice-param",
        str(args.lattice_param),
        "--seed",
        str(args.seed),
        "--max-tasks",
        str(args.max_tasks),
        "--ods-a-element",
        "Zr",
        "--ods-b-element",
        "Y",
        "--oxide-formulas",
        args.oxide_formulas,
        "--particle-radii-A",
        args.particle_radii_A,
        "--oxide-lattice-params-A",
        args.oxide_lattice_params_A,
        "--interface-clearances-A",
        args.interface_clearances_A,
        "--dft-backends",
        "qe",
        "--qe-pseudo-dir",
        args.qe_pseudo_dir,
        "--qe-ecutwfc",
        str(args.qe_ecutwfc),
        "--qe-ecutrho",
        str(args.qe_ecutrho),
        "--qe-kmesh",
        args.qe_kmesh,
        "--qe-conv-thr",
        args.qe_conv_thr,
        "--output-dir",
        args.output_dir,
    ]
    return _build_dataset_parser().parse_args(values)


def _batch_args(args: argparse.Namespace) -> argparse.Namespace:
    values = [
        args.output_dir,
        "--np",
        str(args.np),
        "--omp",
        str(args.omp),
        "--timeout",
        str(args.timeout),
    ]
    if args.batch_limit:
        values.extend(["--limit", str(args.batch_limit)])
    if args.dry_run:
        values.append("--dry-run")
    if args.rerun_completed:
        values.append("--rerun-completed")
    if args.stop_on_error:
        values.append("--stop-on-error")
    return _build_qe_batch_parser().parse_args(values)


def run_closed_loop(args: argparse.Namespace) -> dict:
    if args.smoke:
        args.output_dir = str(_project_root() / "run_output" / "smoke_odsw_qe_closed_loop")
        args.replicas = "4,4,4"
        args.max_tasks = 4
        args.particle_radii_A = "3.0"
        args.oxide_lattice_params_A = "4.4"
        args.interface_clearances_A = "0.8"
        args.dry_run = True
        if not args.batch_limit:
            args.batch_limit = 2

    output_dir = Path(args.output_dir)
    result: dict = {
        "workflow": "odsw_qe_closed_loop",
        "chemistry": "W-Zr-Y-O",
        "chemistry_alias": "W-Y-Zr-O",
        "output_dir": str(output_dir),
        "stage": args.stage,
    }
    if args.stage in {"all", "export"}:
        result["export"] = run_odsw_dft_dataset(_dataset_args(args))
    if args.stage in {"all", "batch"}:
        result["batch"] = run_qe_batch(_batch_args(args))
    if args.stage in {"all", "audit"}:
        result["audit"] = audit_closed_loop(output_dir)
    summary_path = output_dir / "closed_loop_run.json"
    result["closed_loop_run_json"] = str(summary_path)
    _write_json(summary_path, result)
    print(f"W-Zr-Y-O QE closed loop completed. Output: {output_dir}")
    if "audit" in result:
        print(f"Closed-loop pass: {result['audit']['closed_loop_pass']}")
        print(f"Report: {result['audit']['report_md']}")
    return result


def main() -> None:
    args = _build_parser().parse_args()
    run_closed_loop(args)


if __name__ == "__main__":
    main()
