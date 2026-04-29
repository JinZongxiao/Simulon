import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from io_utils.w_structure_builder import (  # noqa: E402
    build_w_structure,
    parse_miller,
    parse_replicas,
    relax_structure_steepest_descent,
    write_build_outputs,
)
from postprocess.w_structure_baseline import (  # noqa: E402
    write_structure_baseline_csv,
    write_structure_baseline_report,
    write_structure_baseline_summary,
)
from run_scripts.w_gb_search import run_w_gb_search  # noqa: E402


@dataclass(frozen=True)
class BaselineCase:
    name: str
    kind: str
    replicas: tuple[int, int, int]
    params: dict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build and relax a production pure-W structure baseline matrix")
    p.add_argument("--preset", choices=("production", "smoke"), default="production")
    p.add_argument("--smoke", action="store_true", help="alias for --preset smoke")
    p.add_argument("--cases", default="all", help="comma-separated cases or 'all'")
    p.add_argument("--orientation", choices=("100", "110", "111"), default="100")
    p.add_argument("--lattice-param", type=float, default=3.1652)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--eam", default=str(_project_root() / "run_data" / "W" / "WRe_YC2.eam.fs"))
    p.add_argument("--output-dir", default=str(_project_root() / "run_output" / "prod_w_structure_baseline"))

    p.add_argument("--relax-method", choices=("sd", "fire"), default="fire")
    p.add_argument("--relax-steps", type=int, default=3000)
    p.add_argument("--relax-step-size-A", type=float, default=0.01)
    p.add_argument("--relax-force-threshold", type=float, default=0.05)
    p.add_argument("--relax-print-interval", type=int, default=100)
    p.add_argument("--skip-relax", action="store_true")

    p.add_argument("--gb-plane", default="3,1,0")
    p.add_argument("--gb-translations-x", type=int, default=7)
    p.add_argument("--gb-translations-z", type=int, default=5)
    p.add_argument("--gb-overlap-cutoff-A", type=float, default=1.6)
    p.add_argument("--gb-search-width-A", type=float, default=6.0)
    p.add_argument("--gb-bulk-reference-replicas", default=None)
    p.add_argument("--skip-gb-search", action="store_true")
    return p


def _case_names(raw: str, include_gb_search: bool) -> set[str]:
    all_cases = {
        "bulk",
        "surface",
        "vacancy",
        "interstitial",
        "void",
        "crack",
        "notch",
        "bicrystal",
    }
    if include_gb_search:
        all_cases.add("gb_search")
    if raw.strip().lower() == "all":
        return all_cases
    requested = {item.strip().lower() for item in raw.split(",") if item.strip()}
    unknown = requested - all_cases
    if unknown:
        raise ValueError(f"unknown baseline cases: {sorted(unknown)}; valid={sorted(all_cases)}")
    return requested


def _production_cases(args) -> list[BaselineCase]:
    return [
        BaselineCase("bulk_100", "bulk", (12, 12, 12), {}),
        BaselineCase("surface_100_z", "surface", (12, 12, 8), {"surface_axis": "z", "vacuum_a": 30.0}),
        BaselineCase("vacancy_1", "vacancy", (12, 12, 12), {"vacancy_count": 1}),
        BaselineCase("interstitial_1", "interstitial", (10, 10, 10), {"interstitial_count": 1}),
        BaselineCase("void_r8", "void", (16, 16, 16), {"void_radius_a": 8.0}),
        BaselineCase(
            "crack_seed",
            "crack",
            (24, 12, 12),
            {"crack_half_length_a": 35.0, "crack_opening_a": 2.0},
        ),
        BaselineCase(
            "notch_seed",
            "notch",
            (24, 12, 12),
            {"surface_axis": "z", "notch_radius_a": 10.0, "notch_depth_a": 10.0},
        ),
        BaselineCase(
            "bicrystal_seed_sigma5_310_001",
            "bicrystal",
            (12, 8, 8),
            {
                "gb_plane": parse_miller(args.gb_plane),
                "gb_overlap_cutoff_a": args.gb_overlap_cutoff_A,
                "gb_search_width_a": args.gb_search_width_A,
            },
        ),
    ]


def _smoke_cases(args) -> list[BaselineCase]:
    return [
        BaselineCase("bulk_100", "bulk", (4, 4, 4), {}),
        BaselineCase("surface_100_z", "surface", (4, 4, 4), {"surface_axis": "z", "vacuum_a": 12.0}),
        BaselineCase("vacancy_1", "vacancy", (4, 4, 4), {"vacancy_count": 1}),
        BaselineCase("interstitial_1", "interstitial", (4, 4, 4), {"interstitial_count": 1}),
        BaselineCase("void_r4", "void", (4, 4, 4), {"void_radius_a": 4.0}),
        BaselineCase(
            "crack_seed",
            "crack",
            (4, 4, 4),
            {"crack_half_length_a": 8.0, "crack_opening_a": 2.5},
        ),
        BaselineCase(
            "notch_seed",
            "notch",
            (4, 4, 4),
            {"surface_axis": "z", "notch_radius_a": 5.0, "notch_depth_a": 5.0},
        ),
        BaselineCase(
            "bicrystal_seed_sigma5_310_001",
            "bicrystal",
            (4, 4, 4),
            {
                "gb_plane": parse_miller(args.gb_plane),
                "gb_overlap_cutoff_a": args.gb_overlap_cutoff_A,
                "gb_search_width_a": args.gb_search_width_A,
            },
        ),
    ]


def _is_finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _run_build_case(case: BaselineCase, args, cases_dir: Path) -> dict:
    result = build_w_structure(
        kind=case.kind,
        orientation=args.orientation,
        replicas=case.replicas,
        lattice_param=args.lattice_param,
        seed=args.seed,
        **case.params,
    )
    summary = write_build_outputs(result, cases_dir, case_name=case.name)
    relax_summary = None
    if not args.skip_relax:
        relax_summary = relax_structure_steepest_descent(
            structure_path=summary["structure"],
            box_vectors=result.box_vectors,
            atom_types=result.atom_types,
            eam_path=args.eam,
            output_dir=summary["output_dir"],
            max_steps=args.relax_steps,
            step_size=args.relax_step_size_A,
            force_threshold=args.relax_force_threshold,
            print_interval=args.relax_print_interval,
            method=args.relax_method,
        )
        summary["relaxation"] = relax_summary
        Path(summary["summary"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    final_energy = relax_summary["final_energy_ev"] if relax_summary else None
    initial_energy = relax_summary["initial_energy_ev"] if relax_summary else None
    atom_count = int(summary["final_atom_count"])
    energy_per_atom = final_energy / max(atom_count, 1) if final_energy is not None else None
    final_max_force = relax_summary["final_max_force_ev_A"] if relax_summary else None
    relax_force_pass = (
        final_max_force is None
        or (_is_finite(final_max_force) and final_max_force <= float(args.relax_force_threshold))
    )
    acceptance = (
        atom_count > 0
        and Path(summary["structure"]).exists()
        and (summary["min_distance_A"] is None or summary["min_distance_A"] > 0.0)
        and (
            relax_summary is None
            or (
                _is_finite(initial_energy)
                and _is_finite(final_energy)
                and final_energy <= initial_energy + 1e-5
                and Path(relax_summary["relaxed_structure"]).exists()
            )
        )
    )
    production_ready = acceptance and relax_force_pass
    return {
        "case_name": case.name,
        "workflow": "build_w_structure",
        "kind": case.kind,
        "orientation": args.orientation,
        "replicas": ",".join(str(x) for x in case.replicas),
        "atom_count": atom_count,
        "min_distance_A": summary["min_distance_A"],
        "initial_energy_ev": initial_energy,
        "final_energy_ev": final_energy,
        "energy_per_atom_ev": energy_per_atom,
        "energy_drop_ev": relax_summary["energy_drop_ev"] if relax_summary else None,
        "final_max_force_ev_A": final_max_force,
        "relax_force_pass": relax_force_pass,
        "converged": relax_summary["converged"] if relax_summary else None,
        "acceptance_pass": acceptance,
        "production_ready": production_ready,
        "structure": summary["structure"],
        "relaxed_structure": relax_summary["relaxed_structure"] if relax_summary else None,
        "summary": summary["summary"],
        "operations": summary.get("operations", {}),
    }


def _run_gb_search_case(args, output_dir: Path) -> dict:
    gb_replicas = "4,4,4" if args.preset == "smoke" else "12,8,8"
    gb_args = SimpleNamespace(
        smoke=False,
        replicas=gb_replicas,
        gb_plane=args.gb_plane,
        lattice_param=args.lattice_param,
        translations_x=2 if args.preset == "smoke" else args.gb_translations_x,
        translations_z=2 if args.preset == "smoke" else args.gb_translations_z,
        gb_overlap_cutoff_A=args.gb_overlap_cutoff_A,
        gb_search_width_A=args.gb_search_width_A,
        eam=args.eam,
        bulk_energy_per_atom_ev="auto",
        bulk_reference_replicas=args.gb_bulk_reference_replicas,
        relax_steps=min(args.relax_steps, 8) if args.preset == "smoke" else args.relax_steps,
        relax_step_size_A=args.relax_step_size_A,
        relax_force_threshold=args.relax_force_threshold,
        relax_print_interval=min(args.relax_print_interval, 4) if args.preset == "smoke" else args.relax_print_interval,
        output_dir=str(output_dir / "gb_search"),
        case_name=None,
    )
    summary = run_w_gb_search(gb_args)
    best = summary["best"]
    relax_force_pass = best["best_final_max_force_ev_A"] <= float(args.relax_force_threshold)
    return {
        "case_name": "gb_search_sigma5_310_001",
        "workflow": "w_gb_search",
        "kind": "gb_search",
        "orientation": "csl_001",
        "replicas": ",".join(str(x) for x in summary["replicas"]),
        "atom_count": None,
        "min_distance_A": None,
        "initial_energy_ev": None,
        "final_energy_ev": best["best_energy_ev"],
        "energy_per_atom_ev": best["best_energy_per_atom_ev"],
        "energy_drop_ev": None,
        "final_max_force_ev_A": best["best_final_max_force_ev_A"],
        "relax_force_pass": relax_force_pass,
        "converged": None,
        "acceptance_pass": bool(best["gb_energy_valid"]),
        "production_ready": bool(best["gb_energy_valid"]) and relax_force_pass,
        "structure": best["best_structure"],
        "relaxed_structure": best["best_relaxed_structure"],
        "summary": summary["gb_energy_report"],
        "gb_energy_J_m2": best["gb_energy_J_m2"],
        "operations": {
            "sigma": best["sigma"],
            "misorientation_deg": best["misorientation_deg"],
            "csl_exact": best["csl_exact"],
        },
    }


def run_w_structure_baseline(args) -> dict:
    if args.smoke:
        args.preset = "smoke"
    if args.preset == "smoke":
        args.output_dir = str(Path(args.output_dir).with_name("smoke_w_structure_baseline"))
        args.relax_steps = min(args.relax_steps, 5)
        args.relax_print_interval = 1

    output_dir = Path(args.output_dir)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    requested = _case_names(args.cases, include_gb_search=not args.skip_gb_search)
    specs = _smoke_cases(args) if args.preset == "smoke" else _production_cases(args)

    rows = []
    for case in specs:
        if case.kind not in requested and case.name not in requested:
            continue
        row = _run_build_case(case, args, cases_dir)
        rows.append(row)
        print(
            f"Baseline case {case.name}: atoms={row['atom_count']}, "
            f"E/N={row['energy_per_atom_ev']}, pass={row['acceptance_pass']}"
        )

    if "gb_search" in requested and not args.skip_gb_search:
        row = _run_gb_search_case(args, output_dir)
        rows.append(row)
        print(
            f"Baseline case gb_search: GB energy={row['gb_energy_J_m2']:.4f} J/m^2, "
            f"pass={row['acceptance_pass']}"
        )

    passed = sum(1 for row in rows if row["acceptance_pass"])
    ready = sum(1 for row in rows if row["production_ready"])
    baseline_csv = write_structure_baseline_csv(output_dir / "structure_baseline.csv", rows)
    summary = {
        "workflow": "w_structure_baseline",
        "preset": args.preset,
        "orientation": args.orientation,
        "lattice_param_A": float(args.lattice_param),
        "eam": str(args.eam),
        "relax_method": args.relax_method,
        "relax_steps": int(args.relax_steps),
        "relax_force_threshold_ev_A": float(args.relax_force_threshold),
        "output_dir": str(output_dir),
        "case_count": len(rows),
        "passed_case_count": passed,
        "failed_case_count": len(rows) - passed,
        "production_ready_case_count": ready,
        "workflow_pass": bool(rows) and passed == len(rows),
        "baseline_csv": baseline_csv,
        "cases": rows,
        "notes": [
            "Pure-W structure baseline matrix.",
            "Fixed-box steepest-descent relaxation is a geometry cleanup stage, not production thermodynamic equilibration.",
            "Use GB-search output for grain-boundary production simulations.",
        ],
    }
    summary_json = write_structure_baseline_summary(output_dir / "summary.json", summary)
    summary["summary_json"] = summary_json
    report = write_structure_baseline_report(output_dir / "report.md", summary)
    summary["report"] = report
    write_structure_baseline_summary(summary_json, summary)
    print(f"W structure baseline completed. Summary: {summary_json}")
    print(f"Report: {report}")
    return summary


def main():
    args = _build_parser().parse_args()
    run_w_structure_baseline(args)


if __name__ == "__main__":
    main()
