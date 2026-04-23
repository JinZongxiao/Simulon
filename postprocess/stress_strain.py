from __future__ import annotations

import csv
from pathlib import Path


def plot_stress_strain(csv_path: str | Path, png_path: str | Path):
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    png_path = Path(png_path)
    strains = []
    sxx = []
    syy = []
    szz = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strains.append(float(row["strain"]))
            sxx.append(float(row["stress_xx_bar"]))
            syy.append(float(row.get("stress_yy_bar", row["stress_xx_bar"])))
            szz.append(float(row.get("stress_zz_bar", row["stress_xx_bar"])))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(strains, sxx, label=r"$\sigma_{xx}$", linewidth=2.0)
    ax.plot(strains, syy, label=r"$\sigma_{yy}$", linewidth=1.5, alpha=0.85)
    ax.plot(strains, szz, label=r"$\sigma_{zz}$", linewidth=1.5, alpha=0.85)
    ax.set_xlabel("Engineering strain")
    ax.set_ylabel("Stress (bar)")
    ax.set_title("W tensile stress-strain response")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def summarize_stress_strain(csv_path: str | Path) -> dict:
    csv_path = Path(csv_path)
    strains = []
    stresses = []
    syy = []
    szz = []
    temps = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strains.append(float(row["strain"]))
            if "stress_xx_bar" in row:
                stresses.append(float(row["stress_xx_bar"]))
                syy.append(float(row.get("stress_yy_bar", row["stress_xx_bar"])))
                szz.append(float(row.get("stress_zz_bar", row["stress_xx_bar"])))
            else:
                stresses.append(float(row["stress_bar"]))
                syy.append(float(row["stress_bar"]))
                szz.append(float(row["stress_bar"]))
            temps.append(float(row.get("temperature_k", 0.0)))

    if not strains:
        raise ValueError(f"No tensile data found in {csv_path}")

    n_fit = min(5, len(strains))
    if n_fit >= 2:
        ds = strains[n_fit - 1] - strains[0]
        modulus = 0.0 if abs(ds) < 1e-12 else (stresses[n_fit - 1] - stresses[0]) / ds
    else:
        modulus = 0.0

    peak_idx = max(range(len(stresses)), key=lambda i: stresses[i])
    n_tail = max(1, min(10, len(stresses)))
    final_stress = stresses[-1]
    final_mean_lateral = 0.5 * (sum(syy[-n_tail:]) + sum(szz[-n_tail:])) / n_tail
    return {
        "n_points": len(strains),
        "strain_min": min(strains),
        "strain_max": max(strains),
        "stress_min_bar": min(stresses),
        "stress_max_bar": stresses[peak_idx],
        "peak_strain": strains[peak_idx],
        "final_stress_bar": final_stress,
        "stress_drop_bar": stresses[peak_idx] - final_stress,
        "mean_final_lateral_stress_bar": final_mean_lateral,
        "max_temperature_k": max(temps),
        "elastic_slope_bar": modulus,
    }
