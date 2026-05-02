import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from io_utils.w_structure_builder import (  # noqa: E402
    build_w_structure,
    parse_miller,
    parse_replicas,
    relax_structure_steepest_descent,
    write_build_outputs,
)
from postprocess.grain_boundary import (  # noqa: E402
    grain_boundary_area_A2,
    grain_boundary_energy_j_m2,
    write_candidates_csv,
    write_gb_energy_report,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rigid-body translation search for W CSL grain boundaries")
    p.add_argument("--gb-plane", default="3,1,0")
    p.add_argument("--replicas", default="8,6,6")
    p.add_argument("--lattice-param", type=float, default=3.1652)
    p.add_argument("--translations-x", type=int, default=5)
    p.add_argument("--translations-z", type=int, default=3)
    p.add_argument("--gb-overlap-cutoff-A", type=float, default=1.6)
    p.add_argument("--gb-search-width-A", type=float, default=6.0)
    p.add_argument("--eam", default=str(_project_root() / "run_data" / "W" / "WRe_YC2.eam.fs"))
    p.add_argument(
        "--bulk-energy-per-atom-ev",
        default="auto",
        help="Bulk reference energy in eV/atom, or 'auto' to evaluate BCC W with the same EAM path.",
    )
    p.add_argument(
        "--bulk-reference-replicas",
        default=None,
        help="Optional replicas for the auto bulk reference; defaults to --replicas.",
    )
    p.add_argument("--relax-steps", type=int, default=300)
    p.add_argument("--relax-method", choices=("sd", "fire"), default="sd")
    p.add_argument("--relax-step-size-A", type=float, default=0.01)
    p.add_argument("--relax-force-threshold", type=float, default=0.1)
    p.add_argument("--relax-print-interval", type=int, default=100)
    p.add_argument("--output-dir", default=str(_project_root() / "run_output" / "w_gb_search"))
    p.add_argument("--case-name", default=None)
    p.add_argument("--smoke", action="store_true")
    return p


def _candidate_shifts(nx: int, nz: int) -> list[tuple[float, float]]:
    nx = max(1, int(nx))
    nz = max(1, int(nz))
    return [(ix / nx, iz / nz) for ix in range(nx) for iz in range(nz)]


def _resolve_bulk_reference(args, replicas: tuple[int, int, int], output_dir: Path) -> tuple[float, dict]:
    raw = str(args.bulk_energy_per_atom_ev).strip().lower()
    if raw != "auto":
        value = float(args.bulk_energy_per_atom_ev)
        return value, {
            "mode": "user_supplied",
            "bulk_energy_per_atom_ev": value,
        }

    bulk_replicas = parse_replicas(args.bulk_reference_replicas) if args.bulk_reference_replicas else replicas
    result = build_w_structure(
        kind="bulk",
        orientation="100",
        replicas=bulk_replicas,
        lattice_param=args.lattice_param,
    )
    ref_dir = output_dir / "_bulk_reference"
    build_summary = write_build_outputs(result, ref_dir, case_name="bulk_100")
    relax_summary = relax_structure_steepest_descent(
        structure_path=build_summary["structure"],
        box_vectors=result.box_vectors,
        atom_types=result.atom_types,
        eam_path=args.eam,
        output_dir=build_summary["output_dir"],
        max_steps=0,
        step_size=args.relax_step_size_A,
        force_threshold=args.relax_force_threshold,
        print_interval=args.relax_print_interval,
        method=args.relax_method,
    )
    atom_count = int(build_summary["final_atom_count"])
    energy_per_atom = float(relax_summary["final_energy_ev"]) / max(atom_count, 1)
    return energy_per_atom, {
        "mode": "auto",
        "bulk_energy_per_atom_ev": energy_per_atom,
        "bulk_reference_replicas": list(bulk_replicas),
        "bulk_reference_atom_count": atom_count,
        "bulk_reference_energy_ev": float(relax_summary["final_energy_ev"]),
        "bulk_reference_structure": build_summary["structure"],
        "bulk_reference_summary": str(Path(build_summary["output_dir"]) / "relax_summary.json"),
    }


def run_w_gb_search(args) -> dict:
    if args.smoke:
        args.replicas = "4,4,4"
        args.translations_x = min(args.translations_x, 2)
        args.translations_z = min(args.translations_z, 2)
        args.relax_steps = min(args.relax_steps, 8)
        args.relax_print_interval = min(args.relax_print_interval, 4)

    replicas = parse_replicas(args.replicas)
    gb_plane = parse_miller(args.gb_plane)
    case_name = args.case_name or f"sigma_gb_{gb_plane[0]}{gb_plane[1]}{gb_plane[2]}_001"
    output_dir = Path(args.output_dir) / case_name
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    bulk_energy_per_atom, bulk_reference = _resolve_bulk_reference(args, replicas, output_dir)

    rows = []
    best_row = None
    best_summary = None
    best_builder_summary = None
    for candidate_id, (sx, sz) in enumerate(_candidate_shifts(args.translations_x, args.translations_z)):
        result = build_w_structure(
            kind="bicrystal",
            replicas=replicas,
            lattice_param=args.lattice_param,
            gb_plane=gb_plane,
            gb_overlap_cutoff_a=args.gb_overlap_cutoff_A,
            gb_search_width_a=args.gb_search_width_A,
            gb_translation_frac=(sx, 0.0, sz),
        )
        candidate_name = f"candidate_{candidate_id:03d}_x{sx:.4f}_z{sz:.4f}".replace(".", "p")
        build_summary = write_build_outputs(result, candidates_dir, case_name=candidate_name)
        relax_summary = relax_structure_steepest_descent(
            structure_path=build_summary["structure"],
            box_vectors=result.box_vectors,
            atom_types=result.atom_types,
            eam_path=args.eam,
            output_dir=build_summary["output_dir"],
            max_steps=args.relax_steps,
            step_size=args.relax_step_size_A,
            force_threshold=args.relax_force_threshold,
            print_interval=args.relax_print_interval,
            method=args.relax_method,
        )
        final_energy = float(relax_summary["final_energy_ev"])
        atom_count = int(build_summary["final_atom_count"])
        row = {
            "candidate_id": candidate_id,
            "shift_x_frac": sx,
            "shift_z_frac": sz,
            "atom_count": atom_count,
            "initial_energy_ev": float(relax_summary["initial_energy_ev"]),
            "final_energy_ev": final_energy,
            "energy_per_atom_ev": final_energy / max(atom_count, 1),
            "energy_drop_ev": float(relax_summary["energy_drop_ev"]),
            "final_max_force_ev_A": float(relax_summary["final_max_force_ev_A"]),
            "final_mean_force_ev_A": float(relax_summary["final_mean_force_ev_A"]),
            "converged": bool(relax_summary["converged"]),
            "structure": build_summary["structure"],
            "relaxed_structure": relax_summary["relaxed_structure"],
            "summary": build_summary["summary"],
            "relax_summary": str(Path(build_summary["output_dir"]) / "relax_summary.json"),
        }
        rows.append(row)
        if best_row is None or row["final_energy_ev"] < best_row["final_energy_ev"]:
            best_row = row
            best_summary = relax_summary
            best_builder_summary = build_summary
        print(
            f"GB candidate {candidate_id}: shift=({sx:.4f},{sz:.4f}), "
            f"E={row['final_energy_ev']:.6f} eV, maxF={row['final_max_force_ev_A']:.4f} eV/A"
        )

    if best_row is None or best_summary is None or best_builder_summary is None:
        raise RuntimeError("GB search produced no candidates")

    candidates_csv = write_candidates_csv(output_dir / "candidates.csv", rows)
    best_structure = output_dir / "best_structure.xyz"
    best_relaxed = output_dir / "best_relaxed_structure.xyz"
    shutil.copyfile(best_row["structure"], best_structure)
    shutil.copyfile(best_row["relaxed_structure"], best_relaxed)

    box_vectors = torch.tensor(best_builder_summary["box_vectors_A"], dtype=torch.float64)
    area = grain_boundary_area_A2(box_vectors)
    gb_energy = grain_boundary_energy_j_m2(
        gb_energy_ev=best_row["final_energy_ev"],
        atom_count=best_row["atom_count"],
        bulk_energy_per_atom_ev=bulk_energy_per_atom,
        area_A2=area,
        n_boundaries=2,
    )
    energy_per_atom_delta = best_row["energy_per_atom_ev"] - float(bulk_energy_per_atom)
    gb_energy_valid = gb_energy > 0.0 and abs(energy_per_atom_delta) < 2.0
    gb_report = {
        "gb_plane_hkl": list(gb_plane),
        "tilt_axis_uvw": [0, 0, 1],
        "sigma": best_builder_summary["operations"]["sigma"],
        "misorientation_deg": best_builder_summary["operations"]["misorientation_deg"],
        "best_candidate_id": best_row["candidate_id"],
        "best_shift_x_frac": best_row["shift_x_frac"],
        "best_shift_z_frac": best_row["shift_z_frac"],
        "best_energy_ev": best_row["final_energy_ev"],
        "best_energy_per_atom_ev": best_row["energy_per_atom_ev"],
        "energy_per_atom_delta_vs_bulk_ev": energy_per_atom_delta,
        "best_final_max_force_ev_A": best_row["final_max_force_ev_A"],
        "bulk_energy_per_atom_ev": float(bulk_energy_per_atom),
        "bulk_reference": bulk_reference,
        "gb_area_A2": area,
        "n_boundaries": 2,
        "gb_energy_J_m2": gb_energy,
        "gb_energy_valid": gb_energy_valid,
        "csl_exact": best_builder_summary["operations"]["csl_exact"],
        "best_structure": str(best_structure),
        "best_relaxed_structure": str(best_relaxed),
        "candidates_csv": candidates_csv,
        "notes": [
            "Rigid-body translation search over one grain only.",
            "GB energy uses the provided bulk_energy_per_atom_ev and assumes two periodic GBs.",
            "If gb_energy_valid is false, treat the structure or bulk reference as not production-ready.",
            "Use larger translation grids and longer relaxation for production values.",
        ],
    }
    gb_energy_report = write_gb_energy_report(output_dir / "gb_energy_report.json", gb_report)
    summary = {
        "workflow": "w_gb_search",
        "output_dir": str(output_dir),
        "candidate_count": len(rows),
        "translations_x": int(args.translations_x),
        "translations_z": int(args.translations_z),
        "replicas": list(replicas),
        "eam": str(args.eam),
        "relax_steps": int(args.relax_steps),
        "relax_method": args.relax_method,
        "relax_force_threshold_ev_A": float(args.relax_force_threshold),
        "best": gb_report,
        "gb_energy_report": gb_energy_report,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"GB search completed. Summary: {summary_path}")
    print(f"Best relaxed structure: {best_relaxed}")
    print(f"GB energy report: {gb_energy_report}")
    return summary


def main():
    args = _build_parser().parse_args()
    run_w_gb_search(args)


if __name__ == "__main__":
    main()
