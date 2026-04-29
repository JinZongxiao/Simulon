from __future__ import annotations

import csv
import json
from pathlib import Path


BASELINE_CSV_FIELDS = [
    "case_name",
    "workflow",
    "kind",
    "orientation",
    "replicas",
    "atom_count",
    "min_distance_A",
    "initial_energy_ev",
    "final_energy_ev",
    "energy_per_atom_ev",
    "energy_drop_ev",
    "final_max_force_ev_A",
    "relax_force_pass",
    "converged",
    "acceptance_pass",
    "production_ready",
    "structure",
    "relaxed_structure",
    "summary",
]


def write_structure_baseline_csv(path: str | Path, rows: list[dict]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASELINE_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in BASELINE_CSV_FIELDS})
    return str(path)


def write_structure_baseline_summary(path: str | Path, summary: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(path)


def write_structure_baseline_report(path: str | Path, summary: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pure W Structure Baseline Report",
        "",
        "## Scope",
        "",
        f"- Preset: `{summary.get('preset')}`",
        f"- Orientation: `{summary.get('orientation')}`",
        f"- Lattice parameter: `{summary.get('lattice_param_A')}` A",
        f"- EAM: `{summary.get('eam')}`",
        f"- Relax method: `{summary.get('relax_method')}`",
        f"- Relax steps: `{summary.get('relax_steps')}`",
        f"- Relax force threshold: `{summary.get('relax_force_threshold_ev_A')}` eV/A",
        f"- Output directory: `{summary.get('output_dir')}`",
        "",
        "## Acceptance",
        "",
        f"- Workflow pass: `{summary.get('workflow_pass')}`",
        f"- Case count: `{summary.get('case_count')}`",
        f"- Passed cases: `{summary.get('passed_case_count')}`",
        f"- Failed cases: `{summary.get('failed_case_count')}`",
        f"- Production-ready cases: `{summary.get('production_ready_case_count')}`",
        "",
        "## Case Summary",
        "",
        "| Case | Kind | Atoms | E/N (eV) | Max force (eV/A) | Integrity pass | Production-ready |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("cases", []):
        epa = row.get("energy_per_atom_ev")
        max_f = row.get("final_max_force_ev_A")
        lines.append(
            "| {case} | {kind} | {atoms} | {epa} | {max_f} | {passed} | {ready} |".format(
                case=row.get("case_name"),
                kind=row.get("kind"),
                atoms=row.get("atom_count"),
                epa=f"{epa:.6f}" if isinstance(epa, (int, float)) else "",
                max_f=f"{max_f:.4f}" if isinstance(max_f, (int, float)) else "",
                passed=row.get("acceptance_pass"),
                ready=row.get("production_ready"),
            )
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- CSV: `{summary.get('baseline_csv')}`",
            f"- JSON: `{summary.get('summary_json')}`",
            "",
            "Each case directory contains `structure.xyz`, `summary.json`, `composition.csv`, `preview.png`, ",
            "and, when relaxation was requested, `relaxed_structure.xyz`, `relaxation.csv`, and `relax_summary.json`.",
            "",
            "## Notes",
            "",
            "- This workflow builds pure-W geometry baselines and fixed-box relaxed seeds.",
            "- Fixed-box steepest descent removes large local forces but does not replace production NVT/NPT relaxation.",
            "- GB production should use the `gb_search` output rather than a raw `bicrystal` seed.",
            "- Notch and crack cases are geometry seeds; mechanics workflows still define loading, grips, and boundary conditions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
