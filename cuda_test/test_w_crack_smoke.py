import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.w_crack import _build_parser, run_w_crack


def _read_rows(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    return rows


def main():
    default_out = Path(__file__).resolve().parents[1] / "run_output" / "smoke_w_crack"
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--smoke",
            "--steps",
            "40",
            "--equil-steps",
            "5",
            "--print-interval",
            "10",
            "--output-dir",
            str(default_out),
        ]
    )
    result = run_w_crack(args)
    rows = _read_rows(result["csv"])
    assert rows, "crack_response.csv must contain data rows"

    strains = [float(row["applied_strain"]) for row in rows]
    cods = [float(row["cmod_A"]) for row in rows]
    stresses = [float(row["stress_bar"]) for row in rows]
    temps = [float(row["temperature_k"]) for row in rows]

    assert all(strains[i] >= strains[i - 1] for i in range(1, len(strains))), "strain must be monotonic"
    assert all(value == value for value in cods), "CMOD contains NaN"
    assert all(value == value for value in stresses), "stress contains NaN"
    assert all(value == value for value in temps), "temperature contains NaN"
    assert max(cods) > 0.0, "CMOD should become positive"
    assert max(stresses) > 0.0, "crack opening stress should be positive in the reported tension-positive convention"
    assert result["n_points"] == len(rows), "summary point count mismatch"
    assert result.get("stress_sign_convention") == "stress_bar is tension-positive"
    assert Path(result["plot"]).exists(), "crack response plot must exist"
    print("W crack smoke test passed.")


if __name__ == "__main__":
    main()
