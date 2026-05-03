from __future__ import annotations

import json
import math
import re
from pathlib import Path


RY_TO_EV = 13.605693122994
BOHR_TO_A = 0.529177210903
RY_BOHR_TO_EV_A = RY_TO_EV / BOHR_TO_A
KBAR_TO_GPA = 0.1


def _parse_qe_input(input_path: str | Path) -> dict:
    path = Path(input_path)
    species: list[str] = []
    positions: list[list[float]] = []
    cell: list[list[float]] = []
    mode = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("!") or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("CELL_PARAMETERS"):
                mode = "cell"
                continue
            if upper.startswith("ATOMIC_POSITIONS"):
                mode = "positions"
                continue
            if upper.startswith("K_POINTS") or upper.startswith("ATOMIC_SPECIES"):
                mode = None
                continue
            if mode == "cell":
                parts = line.split()
                if len(parts) >= 3:
                    cell.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if len(cell) == 3:
                    mode = None
            elif mode == "positions":
                parts = line.split()
                if len(parts) >= 4:
                    species.append(parts[0])
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return {"species": species, "positions_A": positions, "cell_A": cell}


def parse_qe_output(output_path: str | Path, input_path: str | Path | None = None) -> dict:
    path = Path(output_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    energy_ry = None
    forces_ry_bohr: list[list[float]] = []
    stress_kbar: list[list[float]] = []
    scf_iterations = None
    converged = "convergence has been achieved" in text
    job_done = "JOB DONE." in text

    for line in lines:
        if line.lstrip().startswith("!"):
            match = re.search(r"total energy\s*=\s*([-+0-9.Ee]+)\s+Ry", line)
            if match:
                energy_ry = float(match.group(1))
        match = re.search(r"convergence has been achieved in\s+(\d+)\s+iterations", line)
        if match:
            scf_iterations = int(match.group(1))

    for i, line in enumerate(lines):
        if "Forces acting on atoms" in line:
            forces_ry_bohr = []
            for force_line in lines[i + 1 :]:
                if "force =" not in force_line:
                    if forces_ry_bohr and force_line.strip() == "":
                        break
                    continue
                nums = re.findall(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?", force_line)
                if len(nums) >= 3:
                    forces_ry_bohr.append([float(nums[-3]), float(nums[-2]), float(nums[-1])])
        if "total   stress" in line and "(kbar)" in line:
            stress_kbar = []
            for stress_line in lines[i + 1 : i + 4]:
                nums = re.findall(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?", stress_line)
                if len(nums) >= 6:
                    stress_kbar.append([float(nums[-3]), float(nums[-2]), float(nums[-1])])

    input_data = _parse_qe_input(input_path) if input_path else {"species": [], "positions_A": [], "cell_A": []}
    energy_ev = energy_ry * RY_TO_EV if energy_ry is not None else None
    forces_ev_a = [[value * RY_BOHR_TO_EV_A for value in row] for row in forces_ry_bohr]
    stress_gpa = [[value * KBAR_TO_GPA for value in row] for row in stress_kbar]

    numeric_values = []
    if energy_ev is not None:
        numeric_values.append(float(energy_ev))
    for block in (forces_ev_a, stress_gpa, input_data.get("positions_A", []), input_data.get("cell_A", [])):
        for row in block:
            numeric_values.extend(float(x) for x in row)
    no_nan = all(math.isfinite(x) for x in numeric_values)

    return {
        "backend": "qe",
        "output": str(path),
        "input": str(input_path) if input_path else None,
        "job_done": job_done,
        "converged": converged,
        "scf_iterations": scf_iterations,
        "energy_Ry": energy_ry,
        "energy_eV": energy_ev,
        "forces_Ry_bohr": forces_ry_bohr,
        "forces_eV_A": forces_ev_a,
        "stress_kbar": stress_kbar,
        "stress_GPa": stress_gpa,
        "cell_A": input_data.get("cell_A", []),
        "species": input_data.get("species", []),
        "positions_A": input_data.get("positions_A", []),
        "n_atoms": len(input_data.get("species", [])),
        "force_count": len(forces_ev_a),
        "stress_available": bool(stress_gpa),
        "forces_available": bool(forces_ev_a),
        "no_nan": no_nan,
        "label_ready": bool(job_done and converged and energy_ev is not None and forces_ev_a and stress_gpa and no_nan),
    }


def write_qe_label(label: dict, path: str | Path) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(label, indent=2), encoding="utf-8")
    return str(out)
