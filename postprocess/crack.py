from __future__ import annotations

import csv
import math
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
    crack_lengths = [float(row.get("crack_length_A", 0.0)) for row in rows]
    crack_extensions = [float(row.get("crack_extension_A", 0.0)) for row in rows]
    temps = [float(row["temperature_k"]) for row in rows]
    tensile_stresses = [max(0.0, value) for value in stresses]
    peak_stress_idx = max(range(len(tensile_stresses)), key=lambda i: tensile_stresses[i])
    peak_cmod_idx = max(range(len(cods)), key=lambda i: cods[i])
    peak_crack_idx = max(range(len(crack_lengths)), key=lambda i: crack_lengths[i])
    n_fit = min(5, len(strains))
    if n_fit >= 2:
        ds = strains[n_fit - 1] - strains[0]
        cmod_slope = 0.0 if abs(ds) < 1.0e-12 else (cods[n_fit - 1] - cods[0]) / ds
    else:
        cmod_slope = 0.0
    fracture_work = 0.0
    for i in range(1, len(cods)):
        dc = cods[i] - cods[i - 1]
        fracture_work += 0.5 * (tensile_stresses[i] + tensile_stresses[i - 1]) * abs(dc)
    peak_tensile_stress = tensile_stresses[peak_stress_idx]
    final_stress = stresses[-1]
    stress_drop_ratio = 0.0
    if peak_tensile_stress > 1.0e-12:
        stress_drop_ratio = max(0.0, (peak_tensile_stress - final_stress) / peak_tensile_stress)
    return {
        "n_points": len(rows),
        "max_applied_strain": max(strains),
        "stress_min_bar": min(stresses),
        "stress_max_bar": max(stresses),
        "peak_tensile_stress_bar": peak_tensile_stress,
        "peak_stress_magnitude_bar": peak_tensile_stress,
        "peak_stress_step": int(float(rows[peak_stress_idx]["step"])),
        "peak_stress_strain": strains[peak_stress_idx],
        "peak_stress_at_final_step": peak_stress_idx == len(rows) - 1,
        "cmod_at_peak_stress_A": cods[peak_stress_idx],
        "max_cmod_A": cods[peak_cmod_idx],
        "stress_at_max_cmod_bar": stresses[peak_cmod_idx],
        "max_crack_length_A": crack_lengths[peak_crack_idx],
        "max_crack_extension_A": max(crack_extensions),
        "final_crack_length_A": crack_lengths[-1],
        "final_crack_extension_A": crack_extensions[-1],
        "initial_cmod_slope_A_per_strain": cmod_slope,
        "final_stress_bar": final_stress,
        "final_cmod_A": cods[-1],
        "stress_retention_ratio": 0.0 if peak_tensile_stress <= 1.0e-12 else stresses[-1] / peak_tensile_stress,
        "stress_drop_ratio": stress_drop_ratio,
        "fracture_work_proxy_bar_A": fracture_work,
        "mean_temperature_k": sum(temps) / len(temps),
        "stress_sign_convention": "stress_bar is tension-positive",
    }


def plot_crack(csv_path: str | Path, output_path: str | Path) -> str:
    import matplotlib.pyplot as plt

    strains = []
    stresses = []
    cods = []
    crack_lengths = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strains.append(float(row["applied_strain"]))
            stresses.append(float(row["stress_bar"]))
            cods.append(float(row["cmod_A"]))
            crack_lengths.append(float(row.get("crack_length_A", 0.0)))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))

    axes[0].plot(strains, stresses, color="#1f77b4", linewidth=1.8)
    axes[0].set_xlabel("Applied strain")
    axes[0].set_ylabel("Opening stress, tension positive (bar)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(strains, cods, color="#d62728", linewidth=1.8)
    axes[1].set_xlabel("Applied strain")
    axes[1].set_ylabel("CMOD (A)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(strains, crack_lengths, color="#2ca02c", linewidth=1.8)
    axes[2].set_xlabel("Applied strain")
    axes[2].set_ylabel("Estimated crack length (A)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)
