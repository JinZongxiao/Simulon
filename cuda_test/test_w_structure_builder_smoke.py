import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_scripts.build_w_structure import _build_parser, run_build_w_structure


def _run_case(kind: str, extra_args: list[str], output_dir: Path) -> dict:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--kind",
            kind,
            "--orientation",
            "100",
            "--replicas",
            "4,4,4",
            "--output-dir",
            str(output_dir),
            "--case-name",
            kind,
            *extra_args,
        ]
    )
    return run_build_w_structure(args)


def _run_relax_case(output_dir: Path) -> dict:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--kind",
            "vacancy",
            "--orientation",
            "100",
            "--replicas",
            "3,3,3",
            "--vacancy-count",
            "1",
            "--relax",
            "--relax-steps",
            "5",
            "--relax-print-interval",
            "1",
            "--output-dir",
            str(output_dir),
            "--case-name",
            "vacancy_relax",
        ]
    )
    return run_build_w_structure(args)


def _read_xyz_atom_count(path: str | Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return int(f.readline().strip())


def main():
    output_dir = Path(__file__).resolve().parents[1] / "run_output" / "smoke_w_structure_builder"
    cases = {
        "bulk": [],
        "surface": ["--vacuum-A", "12.0"],
        "vacancy": ["--vacancy-count", "3"],
        "interstitial": ["--interstitial-count", "2"],
        "substitution": ["--substitution-count", "4", "--substitution-element", "Re"],
        "ods_w_precursor": [
            "--ods-a-element",
            "Zr",
            "--ods-b-element",
            "Y",
            "--ods-oxide-formula",
            "ABO3",
            "--ods-particle-radius-A",
            "5.0",
            "--ods-oxide-lattice-param-A",
            "4.0",
            "--ods-interface-clearance-A",
            "0.5",
        ],
        "void": ["--void-radius-A", "4.0"],
        "crack": ["--crack-half-length-A", "8.0", "--crack-opening-A", "2.5"],
        "notch": ["--notch-radius-A", "5.0", "--notch-depth-A", "5.0"],
        "bicrystal": ["--gb-plane", "3,1,0", "--gb-overlap-cutoff-A", "1.6"],
    }
    summaries = {}
    for kind, extra in cases.items():
        summary = _run_case(kind, extra, output_dir)
        summaries[kind] = summary
        assert Path(summary["structure"]).exists(), f"{kind} structure.xyz missing"
        assert Path(summary["summary"]).exists(), f"{kind} summary.json missing"
        assert Path(summary["composition_csv"]).exists(), f"{kind} composition.csv missing"
        if kind == "ods_w_precursor":
            assert Path(summary["report"]).exists(), "ODS-W report.md missing"
            assert Path(summary["interface_sanity"]["csv"]).exists(), "ODS-W interface sanity CSV missing"
        assert _read_xyz_atom_count(summary["structure"]) == summary["final_atom_count"]
        assert summary["final_atom_count"] > 0
        assert summary["composition"], f"{kind} composition missing"
        assert summary["box_vectors_A"], f"{kind} box vectors missing"
        assert summary["min_distance_A"] is None or summary["min_distance_A"] > 0.0
        with open(summary["summary"], "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk["kind"] == kind

    assert summaries["surface"]["box_lengths_A"][2] > summaries["bulk"]["box_lengths_A"][2]
    assert summaries["vacancy"]["final_atom_count"] == summaries["bulk"]["final_atom_count"] - 3
    assert summaries["interstitial"]["final_atom_count"] == summaries["bulk"]["final_atom_count"] + 2
    assert summaries["substitution"]["composition"]["Re"] == 4
    assert summaries["ods_w_precursor"]["operations"]["ods_precursor"] is True
    assert summaries["ods_w_precursor"]["operations"]["physics_ready"] is False
    assert summaries["ods_w_precursor"]["operations"]["requires_multielement_potential"] is True
    assert summaries["ods_w_precursor"]["interface_sanity"]["available"] is True
    assert summaries["ods_w_precursor"]["interface_sanity"]["oxide_atom_count"] > 0
    assert summaries["ods_w_precursor"]["interface_sanity"]["interface_oxide_atoms_within_3p5_A"] > 0
    assert summaries["ods_w_precursor"]["composition"]["Zr"] > 0
    assert summaries["ods_w_precursor"]["composition"]["Y"] > 0
    assert summaries["ods_w_precursor"]["composition"]["O"] > 0
    assert summaries["void"]["final_atom_count"] < summaries["bulk"]["final_atom_count"]
    assert summaries["crack"]["final_atom_count"] < summaries["bulk"]["final_atom_count"]
    assert summaries["notch"]["final_atom_count"] < summaries["bulk"]["final_atom_count"]
    assert summaries["bicrystal"]["final_atom_count"] > 0
    assert summaries["bicrystal"]["operations"]["bicrystal_type"] == "csl_001_symmetric_tilt"
    assert summaries["bicrystal"]["operations"]["sigma"] == 5
    assert summaries["bicrystal"]["operations"]["csl_exact"] is True
    assert summaries["bicrystal"]["operations"]["grain_a_atoms"] > 0
    assert summaries["bicrystal"]["operations"]["grain_b_atoms"] > 0

    relax_summary = _run_relax_case(output_dir)
    assert "relaxation" in relax_summary
    assert Path(relax_summary["relaxation"]["relaxed_structure"]).exists()
    assert Path(relax_summary["relaxation"]["relaxation_csv"]).exists()
    assert relax_summary["relaxation"]["final_energy_ev"] <= relax_summary["relaxation"]["initial_energy_ev"] + 1e-5
    print("W structure builder smoke test passed.")


if __name__ == "__main__":
    main()
