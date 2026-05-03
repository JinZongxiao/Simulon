from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from io_utils.w_structure_builder import build_w_structure, parse_replicas, write_build_outputs  # noqa: E402


@dataclass(frozen=True)
class DFTTask:
    task_id: str
    label_source: str
    builder_kind: str
    diversity_role: str
    formula: str
    radius_a: float
    oxide_lattice_param_a: float
    interface_clearance_a: float
    seed: int
    replicas: tuple[int, int, int] | None = None
    strain: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rattle_sigma_a: float = 0.0
    vacancy_count: int = 0
    substitution_element: str = "Re"
    substitution_count: int = 0
    interstitial_count: int = 1
    surface_vacuum_a: float = 20.0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_float_list(value: str) -> list[float]:
    values = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError(f"expected at least one float, got {value!r}")
    return values


def _parse_str_list(value: str) -> list[str]:
    values = [x.strip() for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError(f"expected at least one value, got {value!r}")
    return values


def _element_order(elements: list[str] | set[str]) -> list[str]:
    preferred = ("W", "Zr", "Ti", "Hf", "Y", "Er", "O")
    return sorted(elements, key=lambda x: preferred.index(x) if x in preferred else 99)


def _write_poscar(
    path: Path,
    coords: torch.Tensor,
    atom_types: list[str],
    box_vectors: torch.Tensor,
    comment: str,
) -> str:
    species = _element_order(set(atom_types))
    grouped = []
    counts = []
    for element in species:
        indices = [i for i, atom_type in enumerate(atom_types) if atom_type == element]
        counts.append(len(indices))
        grouped.extend(indices)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{comment}\n")
        f.write("1.0\n")
        for vec in box_vectors.detach().cpu().to(torch.float64).tolist():
            f.write(f"{vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}\n")
        f.write(" ".join(species) + "\n")
        f.write(" ".join(str(x) for x in counts) + "\n")
        f.write("Cartesian\n")
        for idx in grouped:
            xyz = coords[idx].detach().cpu().to(torch.float64).tolist()
            f.write(f"{xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")
    return str(path)


def _qe_pseudopotential_map() -> dict[str, str]:
    return {
        "W": "W_pbe_v1.2.uspp.F.UPF",
        "Zr": "Zr_pbe_v1.uspp.F.UPF",
        "Ti": "ti_pbe_v1.4.uspp.F.UPF",
        "Hf": "Hf-sp.oncvpsp.upf",
        "Y": "Y_pbe_v1.uspp.F.UPF",
        "Er": "Er.paw.z_22.atompaw.wentzcovitch.v1.2.upf",
        "O": "O.pbe-n-kjpaw_psl.0.1.UPF",
    }


def _atomic_masses() -> dict[str, float]:
    return {
        "W": 183.84,
        "Zr": 91.224,
        "Ti": 47.867,
        "Hf": 178.49,
        "Y": 88.90584,
        "Er": 167.259,
        "O": 15.999,
    }


def _write_qe_input_template(
    path: Path,
    coords: torch.Tensor,
    atom_types: list[str],
    box_vectors: torch.Tensor,
    task: DFTTask,
    args: argparse.Namespace,
) -> str:
    species = _element_order(set(atom_types))
    pseudo_map = _qe_pseudopotential_map()
    masses = _atomic_masses()
    missing = [element for element in species if element not in pseudo_map]
    if missing:
        raise ValueError(f"missing QE pseudopotential mapping for {missing}")

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "&CONTROL",
        f"  calculation = '{args.qe_calculation}'",
        f"  prefix = '{task.task_id}'",
        f"  pseudo_dir = '{args.qe_pseudo_dir}'",
        "  outdir = './tmp'",
        "/",
        "&SYSTEM",
        "  ibrav = 0",
        f"  nat = {len(atom_types)}",
        f"  ntyp = {len(species)}",
        f"  ecutwfc = {args.qe_ecutwfc}",
        f"  ecutrho = {args.qe_ecutrho}",
        "  occupations = 'smearing'",
        f"  smearing = '{args.qe_smearing}'",
        f"  degauss = {args.qe_degauss}",
        "/",
        "&ELECTRONS",
        f"  conv_thr = {args.qe_conv_thr}",
        "/",
    ]
    if args.qe_calculation in {"relax", "vc-relax"}:
        lines.extend(
            [
                "&IONS",
                "/",
            ]
        )
    if args.qe_calculation == "vc-relax":
        lines.extend(
            [
                "&CELL",
                "/",
            ]
        )
    lines.append("ATOMIC_SPECIES")
    for element in species:
        lines.append(f"{element} {masses[element]} {pseudo_map[element]}")
    lines.append("CELL_PARAMETERS angstrom")
    for vec in box_vectors.detach().cpu().to(torch.float64).tolist():
        lines.append(f"{vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}")
    lines.append("ATOMIC_POSITIONS angstrom")
    for atom_type, xyz in zip(atom_types, coords.detach().cpu().to(torch.float64).tolist()):
        lines.append(f"{atom_type} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}")
    lines.append("K_POINTS automatic")
    lines.append(f"{args.qe_kmesh} 0 0 0")
    lines.append("")
    lines.append("! Template only. Check ecut, k mesh, pseudopotentials, smearing, and convergence before production.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _write_incar_template(path: Path, task: DFTTask, args: argparse.Namespace) -> str:
    lines = [
        "SYSTEM = Simulon ODS-W DFT task",
        "PREC = Accurate",
        "ENCUT = 520",
        "EDIFF = 1E-6",
        "EDIFFG = -0.02",
        "ISMEAR = 1",
        "SIGMA = 0.2",
        "IBRION = 2",
        "NSW = 100",
        "ISIF = 2",
        "LREAL = Auto",
        "ALGO = Normal",
        "LASPH = .TRUE.",
        "LWAVE = .FALSE.",
        "LCHARG = .FALSE.",
        "",
        "# Template only. Review POTCAR choices, ENCUT, k spacing, and relaxation settings before production.",
        f"# Chemistry: W-{args.ods_a_element}-{args.ods_b_element}-O, oxide template {task.formula}",
        "# Suggested validation sequence: static single-point -> ionic relaxation -> force/stress label export.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _write_kpoints_template(path: Path, args: argparse.Namespace) -> str:
    lines = [
        "KPOINTS template generated by Simulon",
        "0",
        "Gamma",
        f"{args.kmesh}",
        "0 0 0",
        "",
        "# For larger interface cells, Gamma-only or low-density k meshes may be appropriate.",
        "# Keep k-point density consistent across comparable DFT labels.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _write_potcar_required(path: Path, species: list[str]) -> str:
    lines = [
        "This file is a checklist, not a POTCAR.",
        "Use licensed/locally available PAW datasets consistently across the dataset.",
        "Recommended element order must match POSCAR:",
        " ".join(species),
        "",
        "Record exact pseudopotential names and versions in completed DFT metadata.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _write_task_readme(path: Path, row: dict, backends: list[str]) -> str:
    backend_lines = []
    if "qe" in backends:
        backend_lines.append("- `qe/pw.in`: Quantum ESPRESSO template.")
    if "vasp" in backends:
        backend_lines.append("- `vasp/POSCAR`, `vasp/INCAR.template`, `vasp/KPOINTS.template`, `vasp/POTCAR.required.txt`: VASP-style template files.")
    lines = [
        f"# DFT Task {row['task_id']}",
        "",
        "This directory is DFT-ready input, not a completed DFT calculation.",
        "",
        "## Purpose",
        "- Generate first-principles labels for W-Zr-Y-O ODS-W interface configurations.",
        "- Labels needed for MLP training: total energy, atomic forces, stress tensor, final cell, species, and positions.",
        "",
        "## Files",
        "- `common/structure.xyz`: Simulon XYZ export with atom symbols.",
        "- `common/builder_summary.json`: geometry and interface sanity from Simulon.",
        *backend_lines,
        "",
        "## Current Metadata",
        f"- Oxide formula template: `{row['oxide_formula']}`",
        f"- Particle radius: `{row['particle_radius_A']}` A",
        f"- Oxide lattice parameter: `{row['oxide_lattice_param_A']}` A",
        f"- Interface clearance: `{row['interface_clearance_A']}` A",
        f"- Atom count: `{row['atom_count']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _write_dataset_report(path: Path, manifest: dict, rows: list[dict]) -> str:
    composition_set = sorted({row["composition"] for row in rows})
    lines = [
        "# ODS-W DFT Dataset Export Report",
        "",
        "## Scope",
        "This export prepares DFT-ready structures for a first W-Zr-Y-O ODS-W MLP dataset.",
        "It does not run DFT and does not claim that the geometry is DFT-relaxed.",
        "",
        "## Chemistry",
        f"- Matrix: W",
        f"- A element: {manifest['ods_a_element']}",
        f"- B element: {manifest['ods_b_element']}",
        "- Oxide particle model: geometry-only spherical pseudo-oxide precursor",
        "",
        "## Dataset Size",
        f"- Tasks: {len(rows)}",
        f"- Output root: `{manifest['output_dir']}`",
        f"- DFT backends: {', '.join(manifest['dft_backends'])}",
        f"- Campaign: {manifest['campaign']}",
        "",
        "## Configuration Space",
        f"- Replicas: `{manifest['replicas']}`",
        f"- W lattice parameter: {manifest['lattice_param_A']} A",
        f"- Oxide formulas: {', '.join(manifest['oxide_formulas'])}",
        f"- Particle radii: {manifest['particle_radius_A_values']} A",
        f"- Interface clearances: {manifest['interface_clearance_A_values']} A",
        f"- Oxide lattice parameters: {manifest['oxide_lattice_param_A_values']} A",
        "",
        "## Composition Summary",
        *[f"- {comp}" for comp in composition_set],
        "",
        "## Label Source Coverage",
        *[f"- {key}: {value}" for key, value in manifest.get("label_source_counts", {}).items()],
        "",
        "## Diversity Roles",
        *[f"- {role}" for role in manifest.get("diversity_roles", [])],
        "",
        "## Required DFT Labels",
        "- Total energy",
        "- Forces on all atoms",
        "- Stress tensor",
        "- Final cell vectors",
        "- Species and positions",
        "",
        "## Production Warning",
        "These structures are intended as DFT starting points. Before using them as MLP training labels, run DFT relaxation/static calculations and record exact pseudopotential, cutoff, k-point, and convergence settings.",
        "",
        "## Interface Design",
        "Simulon treats DFT as a backend-independent label source. The stable task root is `dft_tasks/<task_id>/`; `qe/` and `vasp/` are backend-specific writers under the same task.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _task_grid(args: argparse.Namespace) -> list[DFTTask]:
    formulas = _parse_str_list(args.oxide_formulas)
    radii = _parse_float_list(args.particle_radii_A)
    lattice_params = _parse_float_list(args.oxide_lattice_params_A)
    clearances = _parse_float_list(args.interface_clearances_A)
    tasks: list[DFTTask] = []
    for formula in formulas:
        for radius in radii:
            for oxide_a in lattice_params:
                for clearance in clearances:
                    seed = args.seed + len(tasks)
                    task_id = (
                        f"W{args.ods_a_element}{args.ods_b_element}O_"
                        f"{formula}_r{radius:g}_a{oxide_a:g}_c{clearance:g}_s{seed}"
                    ).replace(".", "p")
                    tasks.append(
                        DFTTask(
                            task_id=task_id,
                            label_source="ods_interface",
                            builder_kind="ods_w_precursor",
                            diversity_role="interface_chemistry_geometry",
                            formula=formula,
                            radius_a=radius,
                            oxide_lattice_param_a=oxide_a,
                            interface_clearance_a=clearance,
                            seed=seed,
                        )
                    )
                    if args.max_tasks and len(tasks) >= args.max_tasks:
                        return tasks
    return tasks


def _pilot_diverse_tasks(args: argparse.Namespace) -> list[DFTTask]:
    base_replicas = parse_replicas(args.replicas)
    tasks = [
        DFTTask("pure_w_bulk_eq", "pure_w_bulk", "bulk", "equilibrium_reference", "none", 0.0, 0.0, 0.0, args.seed, base_replicas),
        DFTTask("pure_w_bulk_tensile_x_1pct", "pure_w_bulk", "bulk", "elastic_strain", "none", 0.0, 0.0, 0.0, args.seed + 1, base_replicas, (0.01, 0.0, 0.0)),
        DFTTask("pure_w_bulk_compress_1pct", "pure_w_bulk", "bulk", "elastic_strain", "none", 0.0, 0.0, 0.0, args.seed + 2, base_replicas, (-0.01, -0.01, -0.01)),
        DFTTask("pure_w_bulk_rattle_0p03A", "pure_w_bulk", "bulk", "thermal_displacement_proxy", "none", 0.0, 0.0, 0.0, args.seed + 3, base_replicas, (0.0, 0.0, 0.0), 0.03),
        DFTTask("pure_w_surface_100", "pure_w_surface", "surface", "free_surface", "none", 0.0, 0.0, 0.0, args.seed + 4, base_replicas, surface_vacuum_a=15.0),
        DFTTask("pure_w_vacancy_1", "pure_w_defect", "vacancy", "point_defect", "none", 0.0, 0.0, 0.0, args.seed + 5, base_replicas, vacancy_count=1),
        DFTTask("w_zr_substitution_1", "solute_in_w", "substitution", "dilute_solute", "none", 0.0, 0.0, 0.0, args.seed + 6, base_replicas, substitution_element=args.ods_a_element, substitution_count=1),
        DFTTask("w_y_substitution_1", "solute_in_w", "substitution", "dilute_solute", "none", 0.0, 0.0, 0.0, args.seed + 7, base_replicas, substitution_element=args.ods_b_element, substitution_count=1),
        DFTTask("w_self_interstitial_1", "pure_w_defect", "interstitial", "high_energy_point_defect", "none", 0.0, 0.0, 0.0, args.seed + 8, base_replicas, interstitial_count=1),
    ]

    interface_formulas = _parse_str_list(args.oxide_formulas)
    interface_radii = _parse_float_list(args.particle_radii_A)
    oxide_lattice_params = _parse_float_list(args.oxide_lattice_params_A)
    clearances = _parse_float_list(args.interface_clearances_A)
    variants = [
        ((0.0, 0.0, 0.0), 0.0, "interface_reference"),
        ((0.01, 0.0, 0.0), 0.0, "interface_elastic_strain"),
        ((0.0, 0.0, 0.0), 0.03, "interface_thermal_displacement_proxy"),
    ]
    task_seed = args.seed + 100
    for formula in interface_formulas:
        for radius in interface_radii:
            for oxide_a in oxide_lattice_params:
                for clearance in clearances:
                    for strain, rattle_sigma, role in variants:
                        task_id = (
                            f"odsw_{formula}_r{radius:g}_a{oxide_a:g}_c{clearance:g}_{role}_s{task_seed}"
                        ).replace(".", "p")
                        tasks.append(
                            DFTTask(
                                task_id=task_id,
                                label_source="ods_interface",
                                builder_kind="ods_w_precursor",
                                diversity_role=role,
                                formula=formula,
                                radius_a=radius,
                                oxide_lattice_param_a=oxide_a,
                                interface_clearance_a=clearance,
                                seed=task_seed,
                                replicas=base_replicas,
                                strain=strain,
                                rattle_sigma_a=rattle_sigma,
                            )
                        )
                        task_seed += 1
                        if args.max_tasks and len(tasks) >= args.max_tasks:
                            return tasks
    return tasks[: args.max_tasks] if args.max_tasks else tasks


def _apply_task_perturbations(
    coords: torch.Tensor,
    box_vectors: torch.Tensor,
    task: DFTTask,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    perturbed_coords = coords.to(torch.float64)
    perturbed_box = box_vectors.to(torch.float64)
    perturbation = {
        "strain": [float(x) for x in task.strain],
        "rattle_sigma_A": float(task.rattle_sigma_a),
    }
    if any(abs(x) > 0.0 for x in task.strain):
        scale = torch.tensor([1.0 + task.strain[0], 1.0 + task.strain[1], 1.0 + task.strain[2]], dtype=torch.float64)
        perturbed_box = perturbed_box * scale.reshape(3, 1)
        perturbed_coords = perturbed_coords * scale.reshape(1, 3)
    if task.rattle_sigma_a > 0.0:
        generator = torch.Generator(device=perturbed_coords.device)
        generator.manual_seed(int(task.seed))
        noise = torch.randn(perturbed_coords.shape, dtype=perturbed_coords.dtype, generator=generator) * float(task.rattle_sigma_a)
        perturbed_coords = perturbed_coords + noise
    return perturbed_coords, perturbed_box, perturbation


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export DFT-ready ODS-W structure tasks without running DFT")
    p.add_argument("--output-dir", default=str(_project_root() / "run_output" / "odsw_dft_dataset_WZrYO"))
    p.add_argument("--replicas", default="8,8,8")
    p.add_argument("--orientation", choices=("100", "110", "111"), default="100")
    p.add_argument("--lattice-param", type=float, default=3.1652)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-tasks", type=int, default=0, help="optional cap for generated tasks; 0 means no cap")
    p.add_argument("--campaign", choices=("interface_grid", "pilot_diverse"), default="interface_grid")
    p.add_argument("--ods-a-element", choices=("Zr", "Ti", "Hf"), default="Zr")
    p.add_argument("--ods-b-element", choices=("Y", "Er"), default="Y")
    p.add_argument("--oxide-formulas", default="ABO3,A2B2O7")
    p.add_argument("--particle-radii-A", default="5.0,7.0")
    p.add_argument("--oxide-lattice-params-A", default="4.4,4.8")
    p.add_argument("--interface-clearances-A", default="0.8,1.2")
    p.add_argument("--dft-backends", default="qe,vasp", help="comma-separated backend writers: qe,vasp")
    p.add_argument("--kmesh", default="1 1 1", help="legacy VASP KPOINTS mesh")
    p.add_argument("--qe-kmesh", default="1 1 1")
    p.add_argument("--qe-pseudo-dir", default="/public/home/normal_bgd/J1N/software/pseudopotentials")
    p.add_argument("--qe-calculation", choices=("scf", "relax", "vc-relax"), default="scf")
    p.add_argument("--qe-ecutwfc", type=float, default=40.0)
    p.add_argument("--qe-ecutrho", type=float, default=320.0)
    p.add_argument("--qe-smearing", default="mv")
    p.add_argument("--qe-degauss", type=float, default=0.02)
    p.add_argument("--qe-conv-thr", default="1.0d-6")
    p.add_argument("--smoke", action="store_true")
    return p


def run_odsw_dft_dataset(args: argparse.Namespace) -> dict:
    if args.smoke:
        args.replicas = "4,4,4"
        if not args.max_tasks:
            args.max_tasks = 2
        args.particle_radii_A = "4.0"
        args.oxide_lattice_params_A = "4.4"
        args.interface_clearances_A = "0.8,1.2"

    replicas = parse_replicas(args.replicas)
    backends = [backend.lower() for backend in _parse_str_list(args.dft_backends)]
    unsupported = sorted(set(backends) - {"qe", "vasp"})
    if unsupported:
        raise ValueError(f"unsupported DFT backends: {unsupported}")
    output_dir = Path(args.output_dir)
    structures_dir = output_dir / "structures"
    tasks_dir = output_dir / "dft_tasks"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    tasks = _pilot_diverse_tasks(args) if args.campaign == "pilot_diverse" else _task_grid(args)
    if args.max_tasks:
        tasks = tasks[: int(args.max_tasks)]
    if not tasks:
        raise ValueError("no DFT tasks requested")

    for task in tasks:
        task_replicas = task.replicas or replicas
        result = build_w_structure(
            kind=task.builder_kind,
            orientation=args.orientation,
            replicas=task_replicas,
            lattice_param=args.lattice_param,
            seed=task.seed,
            surface_axis="z",
            vacuum_a=task.surface_vacuum_a,
            vacancy_count=task.vacancy_count,
            substitution_element=task.substitution_element,
            substitution_count=task.substitution_count,
            interstitial_count=task.interstitial_count,
            ods_a_element=args.ods_a_element,
            ods_b_element=args.ods_b_element,
            ods_oxide_formula=task.formula,
            ods_particle_radius_a=task.radius_a,
            ods_oxide_lattice_param_a=task.oxide_lattice_param_a,
            ods_interface_clearance_a=task.interface_clearance_a,
        )
        coords, box_vectors, perturbation = _apply_task_perturbations(result.coords, result.box_vectors, task)
        if perturbation["strain"] != [0.0, 0.0, 0.0] or perturbation["rattle_sigma_A"] > 0.0:
            result = type(result)(
                coords=coords,
                atom_types=result.atom_types,
                box_vectors=box_vectors,
                summary={
                    **result.summary,
                    "box_vectors_A": [[float(x) for x in row] for row in box_vectors.tolist()],
                    "box_lengths_A": [float(x) for x in torch.linalg.norm(box_vectors, dim=1).tolist()],
                    "dft_perturbation": perturbation,
                },
            )
        build_summary = write_build_outputs(result, structures_dir, case_name=task.task_id)
        task_dir = tasks_dir / task.task_id
        common_dir = task_dir / "common"
        common_dir.mkdir(parents=True, exist_ok=True)
        common_structure = common_dir / "structure.xyz"
        common_summary = common_dir / "builder_summary.json"
        shutil.copy2(build_summary["structure"], common_structure)
        shutil.copy2(build_summary["summary"], common_summary)
        species = _element_order(set(result.atom_types))
        poscar = ""
        incar = ""
        kpoints = ""
        potcar_required = ""
        qe_input = ""
        if "vasp" in backends:
            vasp_dir = task_dir / "vasp"
            poscar = _write_poscar(
                vasp_dir / "POSCAR",
                result.coords,
                result.atom_types,
                result.box_vectors,
                comment=f"Simulon {task.task_id}",
            )
            incar = _write_incar_template(vasp_dir / "INCAR.template", task, args)
            kpoints = _write_kpoints_template(vasp_dir / "KPOINTS.template", args)
            potcar_required = _write_potcar_required(vasp_dir / "POTCAR.required.txt", species)
        if "qe" in backends:
            qe_input = _write_qe_input_template(
                task_dir / "qe" / "pw.in",
                result.coords,
                result.atom_types,
                result.box_vectors,
                task,
                args,
            )
        composition = build_summary.get("composition", {})
        row = {
            "task_id": task.task_id,
            "label_source": task.label_source,
            "builder_kind": task.builder_kind,
            "diversity_role": task.diversity_role,
            "chemistry": f"W-{args.ods_a_element}-{args.ods_b_element}-O",
            "oxide_formula": task.formula,
            "particle_radius_A": task.radius_a,
            "oxide_lattice_param_A": task.oxide_lattice_param_a,
            "interface_clearance_A": task.interface_clearance_a,
            "strain_x": task.strain[0],
            "strain_y": task.strain[1],
            "strain_z": task.strain[2],
            "rattle_sigma_A": task.rattle_sigma_a,
            "seed": task.seed,
            "atom_count": int(build_summary["final_atom_count"]),
            "composition": json.dumps(composition, sort_keys=True),
            "min_distance_A": build_summary.get("min_distance_A"),
            "interface_sanity_pass": build_summary.get("interface_sanity", {}).get("pass"),
            "structure_xyz": build_summary["structure"],
            "builder_summary": build_summary["summary"],
            "dft_task_dir": str(task_dir),
            "common_structure_xyz": str(common_structure),
            "common_builder_summary": str(common_summary),
            "qe_input": qe_input,
            "poscar": poscar,
            "incar_template": incar,
            "kpoints_template": kpoints,
            "potcar_required": potcar_required,
            "task_readme": str(task_dir / "README.md"),
        }
        _write_task_readme(task_dir / "README.md", row, backends)
        rows.append(row)

    metadata_csv = output_dir / "metadata.csv"
    fieldnames = list(rows[0].keys())
    with open(metadata_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "workflow": "odsw_dft_dataset_export",
        "dft_runner_included": False,
        "dft_software_required": True,
        "target_chemistry": f"W-{args.ods_a_element}-{args.ods_b_element}-O",
        "ods_a_element": args.ods_a_element,
        "ods_b_element": args.ods_b_element,
        "orientation": args.orientation,
        "replicas": args.replicas,
        "campaign": args.campaign,
        "lattice_param_A": args.lattice_param,
        "oxide_formulas": _parse_str_list(args.oxide_formulas),
        "particle_radius_A_values": _parse_float_list(args.particle_radii_A),
        "oxide_lattice_param_A_values": _parse_float_list(args.oxide_lattice_params_A),
        "interface_clearance_A_values": _parse_float_list(args.interface_clearances_A),
        "task_count": len(rows),
        "output_dir": str(output_dir),
        "metadata_csv": str(metadata_csv),
        "structures_dir": str(structures_dir),
        "dft_tasks_dir": str(tasks_dir),
        "dft_backends": backends,
        "label_source_counts": dict(sorted({row["label_source"]: sum(1 for item in rows if item["label_source"] == row["label_source"]) for row in rows}.items())),
        "diversity_roles": sorted({row["diversity_role"] for row in rows}),
        "qe_pseudo_dir": args.qe_pseudo_dir if "qe" in backends else None,
        "vasp_inputs_dir": None,
        "required_labels": ["energy", "forces", "stress", "cell", "species", "positions"],
        "notes": [
            "This workflow exports DFT-ready inputs only; it does not run DFT.",
            "ODS-W precursor structures are geometry-only and require DFT relaxation/static labeling.",
            "Keep pseudopotentials, cutoffs, k-points, and convergence settings consistent across labels.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "dataset_report.md"
    manifest["dataset_report"] = str(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_dataset_report(report_path, manifest, rows)
    print(f"ODS-W DFT dataset export completed. Tasks: {len(rows)}")
    print(f"Metadata: {metadata_csv}")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")
    return manifest


def main() -> None:
    args = _build_parser().parse_args()
    run_odsw_dft_dataset(args)


if __name__ == "__main__":
    main()
