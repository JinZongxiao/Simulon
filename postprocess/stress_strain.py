from __future__ import annotations

import csv
from pathlib import Path


def _get_tension_value(row: dict, axis: str) -> float:
    tension_key = f"tension_{axis}_bar"
    stress_key = f"stress_{axis}_bar"
    if tension_key in row and row[tension_key] not in ("", None):
        return float(row[tension_key])
    if stress_key in row and row[stress_key] not in ("", None):
        return float(row[stress_key])
    if axis == "xx":
        return float(row.get("stress_bar", 0.0))
    return 0.0


def _get_axial_tension_value(row: dict) -> float:
    if "tension_bar" in row and row["tension_bar"] not in ("", None):
        return float(row["tension_bar"])
    return _get_tension_value(row, "xx")


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
            sxx.append(_get_axial_tension_value(row))
            syy.append(_get_tension_value(row, "yy"))
            szz.append(_get_tension_value(row, "zz"))
    if not strains:
        raise ValueError(f"No tensile data found in {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(strains, sxx, label="Axial tension", linewidth=2.0)
    ax.plot(strains, syy, label=r"Lateral $\sigma_{yy}$", linewidth=1.5, alpha=0.85)
    ax.plot(strains, szz, label=r"Lateral $\sigma_{zz}$", linewidth=1.5, alpha=0.85)
    ax.set_xlabel("Engineering strain")
    ax.set_ylabel("Tension-positive stress (bar)")
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
            stresses.append(_get_axial_tension_value(row))
            syy.append(_get_tension_value(row, "yy"))
            szz.append(_get_tension_value(row, "zz"))
            temps.append(float(row.get("temperature_k", 0.0)))

    if not strains:
        raise ValueError(f"No tensile data found in {csv_path}")

    fit_idx = [i for i, strain in enumerate(strains) if 0.0 <= strain <= min(0.005, max(strains))]
    if len(fit_idx) < 2:
        fit_idx = list(range(min(10, len(strains))))
    if len(fit_idx) >= 2:
        i0, i1 = fit_idx[0], fit_idx[-1]
        ds = strains[i1] - strains[i0]
        modulus = 0.0 if abs(ds) < 1e-12 else (stresses[i1] - stresses[i0]) / ds
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
        "stress_sign_convention": "tension_positive",
    }
