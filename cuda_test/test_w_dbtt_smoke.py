import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.w_dbtt_scan import _build_parser, run_w_dbtt_scan


def main():
    default_out = Path(__file__).resolve().parents[1] / "run_output" / "smoke_w_dbtt"
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--smoke",
            "--orientation",
            "100",
            "--output-dir",
            str(default_out),
        ]
    )
    result = run_w_dbtt_scan(args)
    assert result["n_runs"] == 2, "DBTT smoke should produce two temperature points"
    assert Path(result["csv"]).exists(), "DBTT summary CSV must exist"
    assert Path(result["plot"]).exists(), "DBTT summary plot must exist"
    print("W DBTT smoke test passed.")


if __name__ == "__main__":
    main()
