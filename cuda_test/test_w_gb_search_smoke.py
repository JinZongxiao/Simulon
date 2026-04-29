import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.w_gb_search import _build_parser, run_w_gb_search


def main():
    output_dir = Path(__file__).resolve().parents[1] / "run_output" / "smoke_w_gb_search"
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--smoke",
            "--gb-plane",
            "3,1,0",
            "--translations-x",
            "2",
            "--translations-z",
            "2",
            "--relax-steps",
            "8",
            "--output-dir",
            str(output_dir),
        ]
    )
    summary = run_w_gb_search(args)
    best = summary["best"]
    assert summary["candidate_count"] == 4
    assert best["sigma"] == 5
    assert best["csl_exact"] is True
    assert best["gb_area_A2"] > 0.0
    assert best["n_boundaries"] == 2
    assert best["gb_energy_J_m2"] == best["gb_energy_J_m2"]
    assert best["gb_energy_valid"] is True
    assert Path(best["best_relaxed_structure"]).exists()
    assert Path(best["candidates_csv"]).exists()
    assert Path(summary["gb_energy_report"]).exists()
    print("W GB search smoke test passed.")


if __name__ == "__main__":
    main()
