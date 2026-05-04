import csv
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.run_dft_qe_batch import _build_parser, run_qe_batch


def _write_qe_input(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "&CONTROL",
                "  calculation = 'scf'",
                "  prefix = 'batch_smoke'",
                "  pseudo_dir = './pseudo'",
                "  outdir = './tmp'",
                "  tstress = .true.",
                "  tprnfor = .true.",
                "/",
                "&SYSTEM",
                "  ibrav = 0",
                "  nat = 1",
                "  ntyp = 1",
                "  ecutwfc = 30",
                "  ecutrho = 240",
                "/",
                "&ELECTRONS",
                "/",
                "ATOMIC_SPECIES",
                "W 183.84 W.UPF",
                "CELL_PARAMETERS angstrom",
                "3.1652 0 0",
                "0 3.1652 0",
                "0 0 3.1652",
                "ATOMIC_POSITIONS angstrom",
                "W 0 0 0",
                "K_POINTS gamma",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main():
    repo = Path(__file__).resolve().parents[1]
    root = Path(__file__).resolve().parents[1] / "run_output" / "smoke_dft_qe_batch"
    if root.exists():
        shutil.rmtree(root)
    dataset = root / "dataset"
    tasks_dir = dataset / "dft_tasks"
    task_a = tasks_dir / "task_a"
    task_b = tasks_dir / "task_b"
    _write_qe_input(task_a / "qe" / "pw.in")
    _write_qe_input(task_b / "qe" / "pw.in")
    (task_a / "dft_label.json").write_text(json.dumps({"label_ready": True}, indent=2), encoding="utf-8")

    dataset.mkdir(parents=True, exist_ok=True)
    metadata = dataset / "metadata.csv"
    rows = [
        {
            "task_id": "task_a",
            "label_source": "pure_w_bulk",
            "diversity_role": "equilibrium_reference",
            "dft_task_dir": str(task_a.relative_to(repo)),
            "qe_input": str((task_a / "qe" / "pw.in").relative_to(repo)),
        },
        {
            "task_id": "task_b",
            "label_source": "ods_interface",
            "diversity_role": "interface_reference",
            "dft_task_dir": str(task_b.relative_to(repo)),
            "qe_input": str((task_b / "qe" / "pw.in").relative_to(repo)),
        },
    ]
    with open(metadata, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    parser = _build_parser()
    args = parser.parse_args([str(dataset), "--dry-run", "--limit", "1", "--np", "2", "--omp", "1"])
    summary = run_qe_batch(args)
    assert summary["selected_rows"] == 2
    assert summary["attempted_runs"] == 1
    assert summary["state_counts"]["skipped_completed"] == 1
    assert summary["state_counts"]["planned"] == 1
    assert Path(summary["summary_csv"]).exists()
    assert Path(summary["summary_json"]).exists()
    assert not (task_b / "qe" / "qe_status.json").exists()
    assert not (task_b / "qe" / "qe.out").exists()
    with open(summary["summary_csv"], "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    planned = [row for row in rows if row["task_id"] == "task_b"][0]
    assert planned["message"] == "dry run only; task qe_status.json/qe.out were not modified"
    print("DFT QE batch smoke test passed.")


if __name__ == "__main__":
    main()
