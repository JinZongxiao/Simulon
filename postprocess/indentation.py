from __future__ import annotations

import csv
import math
from pathlib import Path


def summarize_load_depth(csv_path: str | Path, indenter_radius_A: float | None = None) -> dict:
    path = Path(csv_path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"n_points": 0}

    depths = [float(row["depth_A"]) for row in rows]
    loads = [float(row["load_nN"]) for row in rows]
    temps = [float(row["temperature_k"]) for row in rows]
    contacts = [int(row["contact_atoms"]) for row in rows]
    phases = [row.get("phase", "load") for row in rows]

    peak_idx = max(range(len(loads)), key=lambda i: loads[i])
    contact_idx = next((i for i, load in enumerate(loads) if load > 1.0e-4), 0)
    fit_stop = min(len(depths), contact_idx + 5)
    if fit_stop - contact_idx >= 2:
        dd = depths[fit_stop - 1] - depths[contact_idx]
        stiffness = 0.0 if abs(dd) < 1.0e-12 else (loads[fit_stop - 1] - loads[contact_idx]) / dd
    else:
        stiffness = 0.0

    unload_idx = next((i for i, phase in enumerate(phases) if phase == "unload"), None)
    unload_stiffness = 0.0
    contact_depth = None
    projected_area = None
    hardness_gpa = None
    reduced_modulus_gpa = None
    if unload_idx is not None:
        unload_fit_stop = min(len(depths), unload_idx + 5)
        if unload_fit_stop - unload_idx >= 2:
            dd = depths[unload_fit_stop - 1] - depths[unload_idx]
            unload_stiffness = 0.0 if abs(dd) < 1.0e-12 else (loads[unload_fit_stop - 1] - loads[unload_idx]) / dd
        if indenter_radius_A and unload_stiffness > 1.0e-12:
            epsilon = 0.75
            contact_depth = depths[peak_idx] - epsilon * loads[peak_idx] / unload_stiffness
            contact_depth = max(0.0, min(contact_depth, 2.0 * float(indenter_radius_A)))
            projected_area = math.pi * max(
                0.0,
                2.0 * float(indenter_radius_A) * contact_depth - contact_depth * contact_depth,
            )
            if projected_area > 1.0e-12:
                hardness_gpa = loads[peak_idx] / projected_area * 100.0
                stiffness_n_per_m = unload_stiffness * 10.0
                area_m2 = projected_area * 1.0e-20
                reduced_modulus_gpa = (
                    (math.sqrt(math.pi) / 2.0) * stiffness_n_per_m / math.sqrt(area_m2) / 1.0e9
                )

    result = {
        "n_points": len(rows),
        "max_depth_A": max(depths),
        "max_load_nN": max(loads),
        "max_contact_atoms": max(contacts),
        "peak_load_depth_A": depths[peak_idx],
        "contact_onset_depth_A": depths[contact_idx],
        "contact_onset_load_nN": loads[contact_idx],
        "initial_loading_stiffness_nN_per_A": stiffness,
        "final_depth_A": depths[-1],
        "final_load_nN": loads[-1],
        "mean_temperature_k": sum(temps) / len(temps),
        "has_unloading": unload_idx is not None,
        "unload_initial_stiffness_nN_per_A": unload_stiffness,
    }
    if contact_depth is not None:
        result["oliver_pharr_contact_depth_A"] = contact_depth
    if projected_area is not None:
        result["projected_contact_area_A2"] = projected_area
    if hardness_gpa is not None:
        result["hardness_GPa"] = hardness_gpa
    if reduced_modulus_gpa is not None:
        result["reduced_modulus_GPa"] = reduced_modulus_gpa
    return result


def plot_load_depth(csv_path: str | Path, output_path: str | Path) -> str:
    import matplotlib.pyplot as plt

    path = Path(csv_path)
    depths_load = []
    loads_load = []
    depths_unload = []
    loads_unload = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = row.get("phase", "load")
            depth = float(row["depth_A"])
            load = float(row["load_nN"])
            if phase == "unload":
                depths_unload.append(depth)
                loads_unload.append(load)
            else:
                depths_load.append(depth)
                loads_load.append(load)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if depths_load:
        ax.plot(depths_load, loads_load, color="#1f77b4", linewidth=1.8, label="Loading")
    if depths_unload:
        ax.plot(depths_unload, loads_unload, color="#d62728", linewidth=1.8, label="Unloading")
    ax.set_xlabel("Indentation depth (A)")
    ax.set_ylabel("Load (nN)")
    ax.grid(True, alpha=0.3)
    if depths_unload:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)
