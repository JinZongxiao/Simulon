import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.w_structure_baseline import _build_parser, run_w_structure_baseline


def main():
    output_dir = Path(__file__).resolve().parents[1] / "run_output" / "smoke_w_structure_baseline"
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--smoke",
            "--output-dir",
            str(output_dir),
            "--cases",
            "all",
        ]
    )
    summary = run_w_structure_baseline(args)
    assert summary["workflow"] == "w_structure_baseline"
    assert summary["preset"] == "smoke"
    assert summary["workflow_pass"] is True
    assert summary["case_count"] >= 9
    assert Path(summary["baseline_csv"]).exists()
    assert Path(summary["summary_json"]).exists()
    assert Path(summary["report"]).exists()
    names = {case["case_name"] for case in summary["cases"]}
    assert "bulk_100" in names
    assert "surface_100_z" in names
    assert "vacancy_1" in names
    assert "interstitial_1" in names
    assert "void_r4" in names
    assert "crack_seed" in names
    assert "notch_seed" in names
    assert "bicrystal_seed_sigma5_310_001" in names
    assert "gb_search_sigma5_310_001" in names
    for case in summary["cases"]:
        assert case["acceptance_pass"] is True, case["case_name"]
        assert Path(case["structure"]).exists()
        assert Path(case["summary"]).exists()
    print("W structure baseline smoke test passed.")


if __name__ == "__main__":
    main()
