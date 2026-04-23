from __future__ import annotations

import csv
from pathlib import Path


def summarize_crack(csv_path: str | Path) -> dict:
    path = Path(csv_path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"n_points": 0}

    strains = [float(row["applied_strain"]) for row in rows]
    stresses = [float(row["stress_bar"]) for row in rows]
    cods = [float(row["cmod_A"]) for row in rows]
    temps = [float(row["temperature_k"]) for row in rows]
    return {
        "n_points": len(rows),
        "max_applied_strain": max(strains),
        "max_stress_bar": max(stresses),
        "max_cmod_A": max(cods),
        "final_stress_bar": stresses[-1],
        "final_cmod_A": cods[-1],
        "mean_temperature_k": sum(temps) / len(temps),
    }


def plot_crack(csv_path: str | Path, output_path: str | Path) -> str:
    import matplotlib.pyplot as plt

    strains = []
    stresses = []
    cods = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strains.append(float(row["applied_strain"]))
            stresses.append(float(row["stress_bar"]))
            cods.append(float(row["cmod_A"]))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    axes[0].plot(strains, stresses, color="#1f77b4", linewidth=1.8)
    axes[0].set_xlabel("Applied strain")
    axes[0].set_ylabel("Stress (bar)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(strains, cods, color="#d62728", linewidth=1.8)
    axes[1].set_xlabel("Applied strain")
    axes[1].set_ylabel("CMOD (A)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)
