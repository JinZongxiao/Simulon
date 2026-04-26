import os
import sys
import csv
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.w_indent import _build_parser, run_w_indent


def _read_rows(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


REQUIRED_LOG_FIELDS = {
    "step",
    "time_ps",
    "phase",
    "depth_A",
    "load_nN",
    "indenter_z",
    "temp",
    "pot",
    "kin",
    "total",
}

REQUIRED_SUMMARY_FIELDS = {
    "max_depth_A",
    "max_load_nN",
    "residual_depth_A",
    "unloading_stiffness_nN_per_A",
    "work_loading",
    "work_unloading",
    "plastic_work_fraction",
    "contact_area_A2",
    "hardness_GPa",
    "hardness_method",
    "pop_in_detected",
    "pop_in_depth_A",
    "pop_in_load_nN",
    "max_temperature_K",
    "no_nan",
    "output_dir",
    "report",
}


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
                "--hold-steps",
                "5",
                "--unload-steps",
                "20",
                "--print-interval",
                "10",
                "--output-dir",
                str(default_out),
            ]
        )
        result = run_w_indent(args)
        rows = _read_rows(result["csv"])
        assert rows, "nanoindent_log.csv must contain data rows"
        assert REQUIRED_LOG_FIELDS.issubset(rows[0].keys()), "nanoindent_log.csv is missing required columns"
        assert REQUIRED_SUMMARY_FIELDS.issubset(result.keys()), "summary is missing required nanoindent fields"

        loads = [float(row["load_nN"]) for row in rows]
        temps = [float(row["temp"]) for row in rows]
        contacts = [int(row["contact_atoms"]) for row in rows]
        phases = {row["phase"] for row in rows}
        loading_rows = [row for row in rows if row["phase"] == "loading"]
        unloading_rows = [row for row in rows if row["phase"] == "unloading"]
        loading_depths = [float(row["depth_A"]) for row in loading_rows]
        unloading_depths = [float(row["depth_A"]) for row in unloading_rows]
        unloading_loads = [float(row["load_nN"]) for row in unloading_rows]

        assert "loading" in phases, "loading phase must be present"
        assert "hold" in phases, "hold phase must be present"
        assert "unloading" in phases, "unloading phase must be present"
        assert all(loading_depths[i] >= loading_depths[i - 1] for i in range(1, len(loading_depths))), "loading depth must increase"
        assert all(
            unloading_depths[i] <= unloading_depths[i - 1] for i in range(1, len(unloading_depths))
        ), "unloading depth must decrease"
        assert unloading_loads[-1] <= unloading_loads[0], "unloading load should decrease"
        assert all(value == value for value in loads), "load contains NaN"
        assert all(value == value for value in temps), "temperature contains NaN"
        assert max(loads) > 0.0, "load should not be identically zero"
        assert max(contacts) > 0, "indentation smoke must contact the slab"
        assert result["no_nan"], "summary no_nan must be true"
        assert result["loading_trend_pass"], "loading load should generally rise"
        assert result["unloading_trend_pass"], "unloading load should generally fall"
        assert result["n_points"] == len(rows), "summary point count mismatch"
        assert result["hardness_method"] == "geometric_spherical_contact_area"
        assert result["plasticity_indicator_available"] is False
        assert result["max_load_nN"] > 0.0
        assert result["contact_area_A2"] > 0.0
        assert result["hardness_GPa"] > 0.0
        assert Path(result["legacy_csv"]).exists(), "legacy load_depth.csv must exist"
        assert Path(result["plot"]).exists(), "load-depth plot must exist"
        assert Path(result["pop_in_plot"]).exists(), "pop-in plot must exist"
        assert Path(result["report"]).exists(), "report.md must exist"
        assert Path(result["traj"]).exists(), "trajectory must exist"
        assert Path(result["snapshots"]["initial"]).exists(), "initial snapshot must exist"
        assert Path(result["snapshots"]["final"]).exists(), "final snapshot must exist"
        assert Path(result["snapshots_png"]["initial"]).exists(), "initial snapshot PNG must exist"
        assert Path(result["snapshots_png"]["final"]).exists(), "final snapshot PNG must exist"
    print("W indentation smoke test passed.")


if __name__ == "__main__":
    main()
