import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.odsw_qe_closed_loop import _build_parser, run_closed_loop


def main():
    output_dir = Path(__file__).resolve().parents[1] / "run_output" / "smoke_odsw_qe_closed_loop"
    parser = _build_parser()
    args = parser.parse_args(["--smoke", "--output-dir", str(output_dir)])
    summary = run_closed_loop(args)
    audit = summary["audit"]
    assert summary["chemistry"] == "W-Zr-Y-O"
    assert audit["task_count"] == 4
    assert audit["coverage_pass"] is False
    assert audit["closed_loop_pass"] is False
    assert Path(audit["metadata_csv"]).exists()
    assert Path(audit["audit_csv"]).exists()
    assert Path(audit["report_md"]).exists()
    assert Path(summary["batch"]["summary_csv"]).exists()
    assert Path(summary["closed_loop_run_json"]).exists()
    print("ODS-W QE closed-loop smoke test passed.")


if __name__ == "__main__":
    main()
