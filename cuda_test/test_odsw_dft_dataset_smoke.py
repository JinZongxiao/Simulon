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
    assert Path(summary["vasp_inputs_dir"]).exists()

    with open(Path(output_dir) / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["required_labels"] == ["energy", "forces", "stress", "cell", "species", "positions"]

    task_dirs = sorted(Path(summary["vasp_inputs_dir"]).glob("*"))
    assert len(task_dirs) == 2
    for task_dir in task_dirs:
        assert (task_dir / "POSCAR").exists()
        assert (task_dir / "structure.xyz").exists()
        assert (task_dir / "builder_summary.json").exists()
        assert (task_dir / "INCAR.template").exists()
        assert (task_dir / "KPOINTS.template").exists()
        assert (task_dir / "POTCAR.required.txt").exists()
        assert (task_dir / "README.md").exists()
    print("ODS-W DFT dataset smoke test passed.")


if __name__ == "__main__":
    main()
