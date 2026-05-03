from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from postprocess.dft_qe import parse_qe_output, write_qe_label  # noqa: E402


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one Simulon DFT task through the Quantum ESPRESSO backend")
    p.add_argument("task_dir", help="path to dft_tasks/<task_id> directory")
    p.add_argument("--input", default=None, help="QE input path; defaults to <task_dir>/qe/pw.in")
    p.add_argument("--output", default=None, help="QE output path; defaults to <task_dir>/qe/qe.out")
    p.add_argument("--label", default=None, help="label JSON path; defaults to <task_dir>/dft_label.json")
    p.add_argument("--status", default=None, help="status JSON path; defaults to <task_dir>/qe/qe_status.json")
    p.add_argument("--mpirun", default="mpirun")
    p.add_argument("--pw", default="pw.x")
    p.add_argument("--np", type=int, default=2)
    p.add_argument("--omp", type=int, default=1)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-failed-label", action="store_true", help="write parsed label even when QE exits non-zero")
    return p


def _write_json(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def run_qe_task(args: argparse.Namespace) -> dict:
    task_dir = Path(args.task_dir).resolve()
    input_path = Path(args.input).resolve() if args.input else task_dir / "qe" / "pw.in"
    output_path = Path(args.output).resolve() if args.output else task_dir / "qe" / "qe.out"
    label_path = Path(args.label).resolve() if args.label else task_dir / "dft_label.json"
    status_path = Path(args.status).resolve() if args.status else task_dir / "qe" / "qe_status.json"
    if not input_path.exists():
        raise FileNotFoundError(f"QE input not found: {input_path}")

    cmd = [args.mpirun, "-np", str(args.np), args.pw, "-in", str(input_path)]
    status = {
        "backend": "qe",
        "task_dir": str(task_dir),
        "input": str(input_path),
        "output": str(output_path),
        "label": str(label_path),
        "status": str(status_path),
        "cmd": cmd,
        "cmd_string": " ".join(shlex.quote(x) for x in cmd),
        "np": int(args.np),
        "omp": int(args.omp),
        "timeout_s": int(args.timeout),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        status.update({"completed": False, "returncode": None, "label_ready": False})
        _write_json(status_path, status)
        print(status["cmd_string"])
        return status

    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.omp)
    start = time.time()
    try:
        with open(output_path, "w", encoding="utf-8") as out:
            proc = subprocess.run(
                cmd,
                cwd=str(input_path.parent),
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=int(args.timeout),
                check=False,
            )
        elapsed = time.time() - start
        status.update({"completed": True, "timed_out": False, "returncode": int(proc.returncode), "elapsed_s": elapsed})
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        status.update({"completed": False, "timed_out": True, "returncode": None, "elapsed_s": elapsed})
        _write_json(status_path, status)
        raise

    if int(status["returncode"]) == 0 or args.allow_failed_label:
        label = parse_qe_output(output_path, input_path=input_path)
        label.update(
            {
                "task_id": task_dir.name,
                "task_dir": str(task_dir),
                "backend": "qe",
                "runner_status": str(status_path),
            }
        )
        write_qe_label(label, label_path)
        status["label_ready"] = bool(label.get("label_ready", False))
        status["label_json"] = str(label_path)
        status["energy_eV"] = label.get("energy_eV")
        status["converged"] = label.get("converged")
        status["job_done"] = label.get("job_done")
    else:
        status["label_ready"] = False
    _write_json(status_path, status)
    print(f"QE task completed: {task_dir}")
    print(f"Status: {status_path}")
    if status.get("label_json"):
        print(f"Label: {label_path}")
    return status


def main() -> None:
    args = _build_parser().parse_args()
    run_qe_task(args)


if __name__ == "__main__":
    main()
