import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.w_tensile import _build_parser, run_w_tensile


def _read_rows(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    return rows


def main():
    default_out = Path(__file__).resolve().parents[1] / "run_output" / "smoke_w_tensile"
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
    result = run_w_tensile(args)
    rows = _read_rows(result["csv"])
    assert rows, "stress_strain.csv must contain data rows"

    strains = [float(row["strain"]) for row in rows]
    stresses = [float(row["stress_xx_bar"]) for row in rows]
    temps = [float(row["temperature_k"]) for row in rows]

    assert all(strains[i] >= strains[i - 1] for i in range(1, len(strains))), "strain must be monotonic"
    assert all(value == value for value in stresses), "stress contains NaN"
    assert all(value == value for value in temps), "temperature contains NaN"
    assert max(abs(s) for s in stresses) > 0.0, "stress should not be identically zero"
    assert result["n_points"] == len(rows), "summary point count mismatch"
    assert Path(result["plot"]).exists(), "stress_strain plot must exist"
    print("W tensile smoke test passed.")


if __name__ == "__main__":
    main()
