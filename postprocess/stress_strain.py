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
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strains.append(float(row["strain"]))
            sxx.append(_get_axial_tension_value(row))
    if not strains:
        raise ValueError(f"No tensile data found in {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(strains, sxx, label=r"Axial tension $\sigma_{xx}$", linewidth=2.2)
    ax.set_xlabel("Engineering strain")
    ax.set_ylabel(r"Axial tension $\sigma_{xx}$ (bar)")
    ax.set_title("W tensile stress-strain response")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def plot_lateral_stress(csv_path: str | Path, png_path: str | Path):
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    png_path = Path(png_path)
    strains = []
    syy = []
    szz = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strains.append(float(row["strain"]))
            syy.append(_get_tension_value(row, "yy"))
            szz.append(_get_tension_value(row, "zz"))
    if not strains:
        raise ValueError(f"No tensile data found in {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(strains, syy, label=r"Lateral tension $\sigma_{yy}$", linewidth=1.7)
    ax.plot(strains, szz, label=r"Lateral tension $\sigma_{zz}$", linewidth=1.7)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Engineering strain")
    ax.set_ylabel("Lateral tension-positive stress (bar)")
    ax.set_title("W tensile lateral stress diagnostic")
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
        "recommended_plot_column": "tension_xx_bar",
        "lateral_stress_columns": ["tension_yy_bar", "tension_zz_bar"],
        "native_stress_sign_convention": "stress_* columns keep the internal compression-positive virial sign; use tension_* for tensile interpretation",
    }


def write_tensile_report(summary: dict, report_path: str | Path) -> str:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# W Tensile Report",
        "",
        "## Sign Convention",
        "- Use `tension_xx_bar` / `tension_bar` for the stress-strain curve.",
        "- `tension_yy_bar` and `tension_zz_bar` are lateral stress diagnostics, not part of the axial tensile curve.",
        "- `stress_*` columns keep the native virial sign convention and are retained for diagnostics.",
        "- The generated `stress_strain.png` shows only axial tension-positive stress.",
        "- `lateral_stress.png` checks whether stress-free lateral control keeps transverse stresses near zero.",
        "",
        "## Main Results",
        f"- Points: {summary.get('n_points', 'n/a')}",
        f"- Maximum strain: {summary.get('strain_max', 0.0):.6g}",
        f"- Peak axial tension: {summary.get('stress_max_bar', 0.0):.6g} bar",
        f"- Peak strain: {summary.get('peak_strain', 0.0):.6g}",
        f"- Final axial tension: {summary.get('final_stress_bar', 0.0):.6g} bar",
        f"- Elastic slope proxy: {summary.get('elastic_slope_bar', 0.0):.6g} bar",
        f"- Mean final lateral tension: {summary.get('mean_final_lateral_stress_bar', 0.0):.6g} bar",
        f"- Maximum temperature: {summary.get('max_temperature_k', 0.0):.6g} K",
        "",
        "## Protocol",
        f"- Structure: {summary.get('structure', 'n/a')}",
        f"- Orientation: {summary.get('orientation', 'n/a')}",
        f"- Steps: {summary.get('steps', 'n/a')}",
        f"- Equilibration steps: {summary.get('equil_steps', 'n/a')}",
        f"- Time step: {summary.get('dt_ps', 'n/a')} ps",
        f"- Temperature: {summary.get('temperature_k', 'n/a')} K",
        f"- Strain rate: {summary.get('strain_rate_ps_inv', 'n/a')} ps^-1",
        f"- Lateral mode: {summary.get('lateral_mode', 'n/a')}",
        "",
        "## Interpretation Notes",
        "- For large W systems, a smooth positive `tension_xx_bar` curve at 300 K is the primary baseline check.",
        "- If the curve looks inverted, verify that plotting did not use `stress_xx_bar` directly.",
        "- Do not average `xx`, `yy`, and `zz` into one tensile curve; uniaxial tensile stress is the axial component.",
        "- Very small systems or very small strains can be dominated by thermal stress fluctuations.",
        "- For custom structures, check the initial absolute stress fields before interpreting the curve.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)
