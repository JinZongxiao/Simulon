import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from io_utils.w_structure_builder import (  # noqa: E402
    SUPPORTED_KINDS,
    build_w_structure,
    parse_miller,
    parse_replicas,
    parse_vector,
    write_build_outputs,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build pure W structures for Simulon workflows")
    p.add_argument("--kind", choices=SUPPORTED_KINDS, default="bulk")
    p.add_argument("--orientation", choices=("100", "110", "111"), default="100")
    p.add_argument("--replicas", default="6,6,6", help="supercell replicas as nx,ny,nz")
    p.add_argument("--lattice-param", type=float, default=3.1652)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output-dir", default=str(_project_root() / "run_output" / "w_structure_builder"))
    p.add_argument("--case-name", default=None, help="optional subdirectory name under output-dir")

    p.add_argument("--surface-axis", choices=("x", "y", "z"), default="z")
    p.add_argument("--vacuum-A", type=float, default=20.0)

    p.add_argument("--vacancy-count", type=int, default=0)
    p.add_argument("--vacancy-fraction", type=float, default=0.0)

    p.add_argument("--interstitial-count", type=int, default=1)
    p.add_argument("--interstitial-element", default="W")

    p.add_argument("--substitution-count", type=int, default=0)
    p.add_argument("--substitution-fraction", type=float, default=0.0)
    p.add_argument("--substitution-element", default="Re")

    p.add_argument("--defect-center", default=None, help="cartesian center as x,y,z in Angstrom")
    p.add_argument("--void-radius-A", type=float, default=5.0)

    p.add_argument("--crack-half-length-A", type=float, default=15.0)
    p.add_argument("--crack-opening-A", type=float, default=2.0)
    p.add_argument("--crack-length-axis", choices=("x", "y", "z"), default="x")
    p.add_argument("--crack-opening-axis", choices=("x", "y", "z"), default="y")

    p.add_argument("--notch-radius-A", type=float, default=6.0)
    p.add_argument("--notch-depth-A", type=float, default=6.0)
    p.add_argument("--notch-surface-side", choices=("min", "max"), default="min")

    p.add_argument("--gb-plane", default="3,1,0", help="strict CSL [001] STGB plane as h,k,0; default is Sigma5(310)[001]")
    p.add_argument("--gb-overlap-cutoff-A", type=float, default=2.0)
    p.add_argument("--gb-search-width-A", type=float, default=6.0)
    return p


def run_build_w_structure(args) -> dict:
    replicas = parse_replicas(args.replicas)
    center = parse_vector(args.defect_center)
    result = build_w_structure(
        kind=args.kind,
        orientation=args.orientation,
        replicas=replicas,
        lattice_param=args.lattice_param,
        seed=args.seed,
        surface_axis=args.surface_axis,
        vacuum_a=args.vacuum_A,
        vacancy_count=args.vacancy_count,
        vacancy_fraction=args.vacancy_fraction,
        interstitial_count=args.interstitial_count,
        interstitial_element=args.interstitial_element,
        substitution_count=args.substitution_count,
        substitution_fraction=args.substitution_fraction,
        substitution_element=args.substitution_element,
        void_radius_a=args.void_radius_A,
        defect_center=center,
        crack_half_length_a=args.crack_half_length_A,
        crack_opening_a=args.crack_opening_A,
        crack_length_axis=args.crack_length_axis,
        crack_opening_axis=args.crack_opening_axis,
        notch_radius_a=args.notch_radius_A,
        notch_depth_a=args.notch_depth_A,
        notch_surface_side=args.notch_surface_side,
        gb_plane=parse_miller(args.gb_plane),
        gb_overlap_cutoff_a=args.gb_overlap_cutoff_A,
        gb_search_width_a=args.gb_search_width_A,
    )
    case_name = args.case_name or f"{args.kind}_orientation_{args.orientation}"
    summary = write_build_outputs(result, args.output_dir, case_name=case_name)
    print(f"W structure build completed: {summary['structure']}")
    print(f"Summary: {summary['summary']}")
    if summary.get("preview"):
        print(f"Preview: {summary['preview']}")
    return summary


def main():
    args = _build_parser().parse_args()
    run_build_w_structure(args)


if __name__ == "__main__":
    main()
