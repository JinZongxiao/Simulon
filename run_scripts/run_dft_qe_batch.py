from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.run_dft_qe_task import run_qe_task  # noqa: E402


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _split_filter(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _write_json(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    base_candidate = (base / path).resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if base_candidate.exists() or not cwd_candidate.exists():
        return base_candidate
    return cwd_candidate


def _label_ready(path: Path) -> bool:
    label = _read_json(path)
    return bool(label and label.get("label_ready") is True)


def _load_rows(metadata_csv: Path) -> list[dict]:
    with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run QE labels for many Simulon DFT tasks listed in metadata.csv")
    p.add_argument("dataset_dir", nargs="?", default=None, help="dataset root containing metadata.csv")
    p.add_argument("--metadata", default=None, help="metadata.csv path; overrides <dataset_dir>/metadata.csv")
    p.add_argument("--summary-dir", default=None, help="where to write qe_batch_summary.*; defaults to dataset root")
    p.add_argument("--limit", type=int, default=0, help="maximum number of non-skipped tasks to run; 0 means no cap")
    p.add_argument("--task-ids", default=None, help="comma-separated exact task_id filter")
    p.add_argument("--label-sources", default=None, help="comma-separated label_source filter")
    p.add_argument("--diversity-roles", default=None, help="comma-separated diversity_role filter")
    p.add_argument("--rerun-completed", action="store_true", help="rerun tasks even if dft_label.json has label_ready=true")
    p.add_argument("--stop-on-error", action="store_true", help="abort the batch on first failed task")
    p.add_argument("--dry-run", action="store_true", help="write planned commands/status files without running QE")
    p.add_argument("--mpirun", default="mpirun")
    p.add_argument("--pw", default="pw.x")
    p.add_argument("--np", type=int, default=2)
    p.add_argument("--omp", type=int, default=1)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--allow-failed-label", action="store_true")
    return p


def _row_selected(row: dict, task_ids: set[str], label_sources: set[str], diversity_roles: set[str]) -> bool:
    if task_ids and row.get("task_id") not in task_ids:
        return False
    if label_sources and row.get("label_source") not in label_sources:
        return False
    if diversity_roles and row.get("diversity_role") not in diversity_roles:
        return False
    return True


def _task_paths(row: dict, metadata_base: Path) -> tuple[Path, Path, Path, Path, Path]:
    task_dir_value = row.get("dft_task_dir") or row.get("task_dir")
    if not task_dir_value:
        raise ValueError(f"metadata row {row.get('task_id', '<unknown>')} does not include dft_task_dir")
    task_dir = _resolve_path(task_dir_value, metadata_base)
    input_path = _resolve_path(row["qe_input"], metadata_base) if row.get("qe_input") else task_dir / "qe" / "pw.in"
    output_path = task_dir / "qe" / "qe.out"
    status_path = task_dir / "qe" / "qe_status.json"
    label_path = task_dir / "dft_label.json"
    return task_dir, input_path, output_path, status_path, label_path


def _status_row(task_id: str, state: str, message: str = "", status: dict | None = None, paths: dict | None = None) -> dict:
    status = status or {}
    paths = paths or {}
    return {
        "task_id": task_id,
        "state": state,
        "label_ready": bool(status.get("label_ready", False)),
        "returncode": status.get("returncode"),
        "elapsed_s": status.get("elapsed_s"),
        "energy_eV": status.get("energy_eV"),
        "converged": status.get("converged"),
        "job_done": status.get("job_done"),
        "task_dir": paths.get("task_dir", status.get("task_dir", "")),
        "input": paths.get("input", status.get("input", "")),
        "output": paths.get("output", status.get("output", "")),
        "label": paths.get("label", status.get("label", "")),
        "status_json": paths.get("status", status.get("status", "")),
        "message": message,
    }


def run_qe_batch(args: argparse.Namespace) -> dict:
    if args.metadata:
        metadata_csv = Path(args.metadata).resolve()
        dataset_dir = metadata_csv.parent
    elif args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).resolve()
        metadata_csv = dataset_dir / "metadata.csv"
    else:
        dataset_dir = _project_root() / "run_output" / "odsw_dft_dataset_WZrYO"
        metadata_csv = dataset_dir / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv}")

    rows = _load_rows(metadata_csv)
    metadata_base = metadata_csv.parent
    summary_dir = Path(args.summary_dir).resolve() if args.summary_dir else dataset_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "qe_batch_summary.csv"
    json_path = summary_dir / "qe_batch_summary.json"

    task_ids = _split_filter(args.task_ids)
    label_sources = _split_filter(args.label_sources)
    diversity_roles = _split_filter(args.diversity_roles)

    selected = [row for row in rows if _row_selected(row, task_ids, label_sources, diversity_roles)]
    results: list[dict] = []
    started = time.time()
    run_count = 0

    for row in selected:
        task_id = row.get("task_id", "")
        try:
            task_dir, input_path, output_path, status_path, label_path = _task_paths(row, metadata_base)
            paths = {
                "task_dir": str(task_dir),
                "input": str(input_path),
                "output": str(output_path),
                "label": str(label_path),
                "status": str(status_path),
            }
            if _label_ready(label_path) and not args.rerun_completed:
                results.append(_status_row(task_id, "skipped_completed", "existing label_ready=true", {"label_ready": True}, paths))
                continue
            if args.limit and run_count >= int(args.limit):
                results.append(_status_row(task_id, "skipped_limit", "batch limit reached", paths=paths))
                continue
            if not input_path.exists():
                results.append(_status_row(task_id, "missing_input", f"QE input not found: {input_path}", paths=paths))
                if args.stop_on_error:
                    break
                continue

            task_args = argparse.Namespace(
                task_dir=str(task_dir),
                input=str(input_path),
                output=str(output_path),
                label=str(label_path),
                status=str(status_path),
                mpirun=args.mpirun,
                pw=args.pw,
                np=int(args.np),
                omp=int(args.omp),
                timeout=int(args.timeout),
                dry_run=bool(args.dry_run),
                allow_failed_label=bool(args.allow_failed_label),
            )
            run_count += 1
            status = run_qe_task(task_args)
            state = "planned" if args.dry_run else ("completed" if status.get("label_ready") else "failed")
            results.append(_status_row(task_id, state, status=status, paths=paths))
            if state == "failed" and args.stop_on_error:
                break
        except Exception as exc:
            results.append(_status_row(task_id, "error", repr(exc)))
            if args.stop_on_error:
                break

    counts: dict[str, int] = {}
    for item in results:
        counts[item["state"]] = counts.get(item["state"], 0) + 1
    label_ready_count = sum(1 for item in results if item.get("label_ready") is True)
    summary = {
        "workflow": "qe_dft_batch",
        "metadata_csv": str(metadata_csv),
        "dataset_dir": str(dataset_dir),
        "summary_csv": str(csv_path),
        "summary_json": str(json_path),
        "total_metadata_rows": len(rows),
        "selected_rows": len(selected),
        "attempted_runs": run_count,
        "label_ready_count": label_ready_count,
        "state_counts": counts,
        "np": int(args.np),
        "omp": int(args.omp),
        "timeout_s": int(args.timeout),
        "dry_run": bool(args.dry_run),
        "rerun_completed": bool(args.rerun_completed),
        "elapsed_s": time.time() - started,
        "results": results,
    }

    fieldnames = [
        "task_id",
        "state",
        "label_ready",
        "returncode",
        "elapsed_s",
        "energy_eV",
        "converged",
        "job_done",
        "task_dir",
        "input",
        "output",
        "label",
        "status_json",
        "message",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    _write_json(json_path, summary)

    print(f"QE batch completed. Selected: {len(selected)}, attempted: {run_count}, label_ready: {label_ready_count}")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    run_qe_batch(args)


if __name__ == "__main__":
    main()
