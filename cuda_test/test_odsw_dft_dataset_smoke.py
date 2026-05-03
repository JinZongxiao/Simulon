import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.build_odsw_dft_dataset import _build_parser, run_odsw_dft_dataset


def main():
    output_dir = Path(__file__).resolve().parents[1] / "run_output" / "smoke_odsw_dft_dataset"
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--smoke",
            "--output-dir",
            str(output_dir),
        ]
    )
    summary = run_odsw_dft_dataset(args)
    assert summary["workflow"] == "odsw_dft_dataset_export"
    assert summary["dft_runner_included"] is False
    assert summary["dft_software_required"] is True
    assert summary["task_count"] == 2
    assert Path(summary["metadata_csv"]).exists()
    assert Path(summary["dataset_report"]).exists()
    assert Path(summary["dft_tasks_dir"]).exists()
    assert summary["dft_backends"] == ["qe", "vasp"]

    with open(Path(output_dir) / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["required_labels"] == ["energy", "forces", "stress", "cell", "species", "positions"]

    task_dirs = sorted(Path(summary["dft_tasks_dir"]).glob("*"))
    assert len(task_dirs) == 2
    for task_dir in task_dirs:
        assert (task_dir / "common" / "structure.xyz").exists()
        assert (task_dir / "common" / "builder_summary.json").exists()
        assert (task_dir / "qe" / "pw.in").exists()
        assert (task_dir / "vasp" / "POSCAR").exists()
        assert (task_dir / "vasp" / "INCAR.template").exists()
        assert (task_dir / "vasp" / "KPOINTS.template").exists()
        assert (task_dir / "vasp" / "POTCAR.required.txt").exists()
        assert (task_dir / "README.md").exists()

    diverse_output_dir = Path(__file__).resolve().parents[1] / "run_output" / "smoke_odsw_dft_dataset_diverse"
    diverse_args = parser.parse_args(
        [
            "--smoke",
            "--campaign",
            "pilot_diverse",
            "--max-tasks",
            "4",
            "--output-dir",
            str(diverse_output_dir),
        ]
    )
    diverse_summary = run_odsw_dft_dataset(diverse_args)
    assert diverse_summary["campaign"] == "pilot_diverse"
    assert diverse_summary["task_count"] == 4
    assert "pure_w_bulk" in diverse_summary["label_source_counts"]
    assert "elastic_strain" in diverse_summary["diversity_roles"]
    assert Path(diverse_summary["dft_tasks_dir"]).exists()
    print("ODS-W DFT dataset smoke test passed.")


if __name__ == "__main__":
    main()
