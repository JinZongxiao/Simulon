import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.w_indent import _build_parser, run_w_indent


def _read_rows(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    return rows


def main():
    default_out = Path(__file__).resolve().parents[1] / "run_output" / "smoke_w_indent"
    parser = _build_parser()
    for orientation in ("100", "110", "111"):
        args = parser.parse_args(
            [
                "--smoke",
                "--orientation",
                orientation,
                "--steps",
                "30",
                "--print-interval",
                "10",
                "--output-dir",
                str(default_out),
            ]
        )
        result = run_w_indent(args)
        rows = _read_rows(result["csv"])
        assert rows, "load_depth.csv must contain data rows"

        depths = [float(row["depth_A"]) for row in rows]
        loads = [float(row["load_nN"]) for row in rows]
        temps = [float(row["temperature_k"]) for row in rows]
        contacts = [int(row["contact_atoms"]) for row in rows]

        assert all(depths[i] >= depths[i - 1] for i in range(1, len(depths))), "depth must be monotonic"
        assert all(value == value for value in loads), "load contains NaN"
        assert all(value == value for value in temps), "temperature contains NaN"
        assert max(loads) > 0.0, "load should not be identically zero"
        assert max(contacts) > 0, "indentation smoke must contact the slab"
        assert result["n_points"] == len(rows), "summary point count mismatch"
        assert Path(result["plot"]).exists(), "load-depth plot must exist"
    print("W indentation smoke test passed.")


if __name__ == "__main__":
    main()
