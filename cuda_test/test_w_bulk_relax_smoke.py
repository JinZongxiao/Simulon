import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.w_bulk_relax import _build_parser, run_w_bulk_relax


def main():
    default_out = Path(__file__).resolve().parents[1] / "run_output" / "smoke_w_bulk_relax"
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--smoke",
            "--steps",
            "20",
            "--print-interval",
            "5",
            "--output-dir",
            str(default_out),
        ]
    )
    result = run_w_bulk_relax(args)
    assert Path(result["csv"]).exists(), "relaxation.csv must exist"
    assert Path(result["summary"]).exists(), "summary.json must exist"
    assert Path(result["relaxed_structure"]).exists(), "relaxed xyz must exist"
    assert abs(float(result["final_pressure_bar"])) < 1.0e6, "pressure should stay finite"
    assert float(result["recommended_box_length_A"]) > 0.0, "recommended box length must be positive"
    print("W bulk relax smoke test passed.")


if __name__ == "__main__":
    main()
