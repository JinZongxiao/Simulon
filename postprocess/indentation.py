from __future__ import annotations

import csv
from pathlib import Path


def summarize_load_depth(csv_path: str | Path) -> dict:
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
    peak_idx = max(range(len(loads)), key=lambda i: loads[i])
    contact_idx = next((i for i, load in enumerate(loads) if load > 1.0e-4), 0)
    fit_stop = min(len(depths), contact_idx + 5)
    if fit_stop - contact_idx >= 2:
        dd = depths[fit_stop - 1] - depths[contact_idx]
        stiffness = 0.0 if abs(dd) < 1.0e-12 else (loads[fit_stop - 1] - loads[contact_idx]) / dd
    else:
        stiffness = 0.0
    return {
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
    }


def plot_load_depth(csv_path: str | Path, output_path: str | Path) -> str:
    import matplotlib.pyplot as plt

    path = Path(csv_path)
    depths = []
    loads = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            depths.append(float(row["depth_A"]))
            loads.append(float(row["load_nN"]))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(depths, loads, color="#1f77b4", linewidth=1.8)
    ax.set_xlabel("Indentation depth (A)")
    ax.set_ylabel("Load (nN)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)
