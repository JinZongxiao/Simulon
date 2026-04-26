from __future__ import annotations

import csv
import math
from pathlib import Path


def _read_rows(csv_path: str | Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _field(row: dict, *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in row and row[name] not in ("", None):
            return float(row[name])
    return float(default)


def _phase(row: dict) -> str:
    phase = row.get("phase", "loading").lower()
    if phase == "load":
        return "loading"
    if phase == "unload":
        return "unloading"
    return phase


def _trapz_work(depths: list[float], loads: list[float], sign: float = 1.0) -> float:
    if len(depths) < 2:
        return 0.0
    work = 0.0
    for i in range(1, len(depths)):
        dh = depths[i] - depths[i - 1]
        work += 0.5 * (loads[i] + loads[i - 1]) * dh
    return max(0.0, sign * work)


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 1.0e-20:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    slope = _linear_slope(xs, ys)
    if not xs:
        return 0.0, 0.0
    intercept = sum(ys) / len(ys) - slope * (sum(xs) / len(xs))
    return slope, intercept


def _detect_pop_in(depths: list[float], loads: list[float]) -> dict:
    if len(depths) < 5:
        return {"pop_in_detected": False, "pop_in_depth_A": None, "pop_in_load_nN": None}
    max_load = max(loads)
    max_depth = max(depths)
    min_event_load = max(0.5, 0.05 * max_load)
    min_event_depth = max(0.1, 0.02 * max_depth)
    load_drop_threshold = max(0.5, 0.05 * max_load)
    for i in range(1, len(loads)):
        if (
            depths[i] >= min_event_depth
            and max(loads[i - 1], loads[i]) >= min_event_load
            and loads[i - 1] - loads[i] >= load_drop_threshold
        ):
            return {
                "pop_in_detected": True,
                "pop_in_depth_A": depths[i],
                "pop_in_load_nN": loads[i],
                "pop_in_reason": "load_drop",
            }

    slopes = []
    slope_depths = []
    for i in range(1, len(depths)):
        dh = depths[i] - depths[i - 1]
        if abs(dh) <= 1.0e-12:
            continue
        slopes.append((loads[i] - loads[i - 1]) / dh)
        slope_depths.append(depths[i])
    if len(slopes) < 5:
        return {"pop_in_detected": False, "pop_in_depth_A": None, "pop_in_load_nN": None}

    for i in range(3, len(slopes)):
        previous = [value for value in slopes[max(0, i - 5) : i] if value > 0.0]
        if len(previous) < 3:
            continue
        previous_sorted = sorted(previous)
        median_prev = previous_sorted[len(previous_sorted) // 2]
        if median_prev > 1.0e-12 and slopes[i] < 0.25 * median_prev:
            depth = slope_depths[i]
            nearest = min(range(len(depths)), key=lambda j: abs(depths[j] - depth))
            if depth < min_event_depth or loads[nearest] < min_event_load:
                continue
            return {
                "pop_in_detected": True,
                "pop_in_depth_A": depth,
                "pop_in_load_nN": loads[nearest],
                "pop_in_reason": "loading_stiffness_drop",
            }

    return {"pop_in_detected": False, "pop_in_depth_A": None, "pop_in_load_nN": None}


def summarize_load_depth(csv_path: str | Path, indenter_radius_A: float | None = None) -> dict:
    rows = _read_rows(csv_path)
    if not rows:
        return {"n_points": 0, "no_nan": False}

    depths = [_field(row, "depth_A") for row in rows]
    loads = [_field(row, "load_nN") for row in rows]
    temps = [_field(row, "temp", "temperature_k") for row in rows]
    pots = [_field(row, "pot", "potential_energy_ev") for row in rows]
    kins = [_field(row, "kin", "kinetic_energy_ev") for row in rows]
    totals = [_field(row, "total", "total_energy_ev") for row in rows]
    contacts = [int(_field(row, "contact_atoms", default=0.0)) for row in rows]
    phases = [_phase(row) for row in rows]

    numeric_values = depths + loads + temps + pots + kins + totals
    no_nan = all(math.isfinite(value) for value in numeric_values)
    peak_idx = max(range(len(loads)), key=lambda i: loads[i])

    loading_idx = [i for i, phase in enumerate(phases) if phase == "loading"]
    unloading_idx = [i for i, phase in enumerate(phases) if phase == "unloading"]
    loading_depths = [depths[i] for i in loading_idx]
    loading_loads = [loads[i] for i in loading_idx]
    unloading_depths = [depths[i] for i in unloading_idx]
    unloading_loads = [loads[i] for i in unloading_idx]

    work_loading = _trapz_work(loading_depths, loading_loads, sign=1.0)
    work_unloading = _trapz_work(unloading_depths, unloading_loads, sign=-1.0)
    plastic_work_fraction = 0.0
    if work_loading > 1.0e-12:
        plastic_work_fraction = max(0.0, min(1.0, (work_loading - work_unloading) / work_loading))

    unloading_stiffness = 0.0
    unloading_intercept = 0.0
    if len(unloading_idx) >= 3:
        n_fit = min(10, len(unloading_idx))
        slope, intercept = _linear_fit(unloading_depths[:n_fit], unloading_loads[:n_fit])
        unloading_stiffness = max(0.0, slope)
        unloading_intercept = intercept

    max_depth = max(depths)
    max_load = max(loads)
    residual_depth = unloading_depths[-1] if unloading_depths else depths[-1]
    if unloading_stiffness > 1.0e-12:
        residual_depth = max(0.0, min(max_depth, -unloading_intercept / unloading_stiffness))
    contact_area = None
    hardness_gpa = None
    if indenter_radius_A is not None:
        h = max(0.0, min(max_depth, 2.0 * float(indenter_radius_A)))
        contact_area = math.pi * max(0.0, 2.0 * float(indenter_radius_A) * h - h * h)
        if contact_area > 1.0e-12:
            hardness_gpa = max_load / contact_area * 100.0

    loading_pop_in = _detect_pop_in(loading_depths, loading_loads)

    loading_trend_pass = False
    if len(loading_loads) >= 2:
        loading_slope = _linear_slope(loading_depths, loading_loads)
        loading_trend_pass = max(loading_loads) > loading_loads[0] and loading_slope > 0.0
    unloading_trend_pass = False
    if len(unloading_loads) >= 2:
        unloading_slope = _linear_slope(unloading_depths, unloading_loads)
        unloading_trend_pass = unloading_loads[-1] <= unloading_loads[0] and unloading_slope > 0.0

    return {
        "n_points": len(rows),
        "max_depth_A": max_depth,
        "max_load_nN": max_load,
        "residual_depth_A": residual_depth,
        "unloading_stiffness_nN_per_A": unloading_stiffness,
        "work_loading": work_loading,
        "work_unloading": work_unloading,
        "plastic_work_fraction": plastic_work_fraction,
        "contact_area_A2": contact_area,
        "hardness_GPa": hardness_gpa,
        "hardness_method": "geometric_spherical_contact_area",
        "pop_in_detected": bool(loading_pop_in["pop_in_detected"]),
        "pop_in_depth_A": loading_pop_in["pop_in_depth_A"],
        "pop_in_load_nN": loading_pop_in["pop_in_load_nN"],
        "pop_in_reason": loading_pop_in.get("pop_in_reason"),
        "max_temperature_K": max(temps),
        "mean_temperature_K": sum(temps) / len(temps),
        "no_nan": no_nan,
        "max_contact_atoms": max(contacts) if contacts else 0,
        "peak_load_depth_A": depths[peak_idx],
        "final_depth_A": depths[-1],
        "final_load_nN": loads[-1],
        "has_unloading": bool(unloading_idx),
        "loading_trend_pass": loading_trend_pass,
        "unloading_trend_pass": unloading_trend_pass,
        "plasticity_indicator_available": False,
        "plasticity_indicator": None,
        "plasticity_indicator_note": "not computed; no CSP/non-bcc/displacement plasticity proxy is available in this workflow yet",
    }


def plot_load_depth(
    csv_path: str | Path,
    output_path: str | Path,
    pop_in: dict | None = None,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_rows(csv_path)
    by_phase: dict[str, tuple[list[float], list[float]]] = {}
    for row in rows:
        phase = _phase(row)
        by_phase.setdefault(phase, ([], []))
        by_phase[phase][0].append(_field(row, "depth_A"))
        by_phase[phase][1].append(_field(row, "load_nN"))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    styles = {
        "loading": ("#1f77b4", "Loading"),
        "hold": ("#ff7f0e", "Hold"),
        "unloading": ("#d62728", "Unloading"),
    }
    for phase, (depths, loads) in by_phase.items():
        color, label = styles.get(phase, ("#303030", phase))
        ax.plot(depths, loads, color=color, linewidth=1.8, label=label)
    if pop_in and pop_in.get("pop_in_detected"):
        ax.scatter(
            [float(pop_in["pop_in_depth_A"])],
            [float(pop_in["pop_in_load_nN"])],
            color="#000000",
            s=45,
            zorder=5,
            label="Pop-in",
        )
        ax.annotate(
            "pop-in",
            xy=(float(pop_in["pop_in_depth_A"]), float(pop_in["pop_in_load_nN"])),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Indentation depth (A)")
    ax.set_ylabel("Load (nN)")
    ax.grid(True, alpha=0.3)
    if by_phase:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _fmt(value, digits: int = 4, unit: str = "") -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{value:.{digits}f}"
    if unit:
        text += f" {unit}"
    return text


def _yes_no(value) -> str:
    return "yes" if bool(value) else "no"


def write_indentation_report(summary: dict, output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    contact_atoms = int(summary.get("max_contact_atoms") or 0)
    loading_ok = bool(summary.get("loading_trend_pass"))
    unloading_ok = bool(summary.get("unloading_trend_pass")) if summary.get("has_unloading") else False
    final_load = float(summary.get("final_load_nN") or 0.0)
    max_load = float(summary.get("max_load_nN") or 0.0)
    final_load_ratio = final_load / max_load if max_load > 1.0e-12 else None
    residual_depth = float(summary.get("residual_depth_A") or 0.0)
    target_temperature = summary.get("temperature_K")
    mean_temperature = summary.get("mean_temperature_K")

    interpretation = []
    if loading_ok:
        interpretation.append("During loading, the load-depth response rises overall with indentation depth.")
    else:
        interpretation.append("During loading, the load-depth response is not clearly monotonic; inspect the trajectory and log before using the result as a baseline.")
    if summary.get("has_unloading"):
        if unloading_ok:
            interpretation.append("During unloading, the load decreases overall as the indenter retracts.")
        else:
            interpretation.append("During unloading, the load does not decrease cleanly; the unloading stiffness and work recovery should be treated cautiously.")
    else:
        interpretation.append("No unloading segment was run, so residual depth and unloading stiffness are limited diagnostics.")
    if residual_depth > 0.1:
        interpretation.append("A finite residual indentation depth is present after unloading, indicating irreversible deformation or incomplete elastic recovery.")
    else:
        interpretation.append("Residual depth is small in this run; plastic deformation is limited by this metric.")
    if summary.get("pop_in_detected"):
        interpretation.append("A pop-in-like event was detected from a load drop or sudden loading-stiffness drop.")
    else:
        interpretation.append("No clear pop-in event was detected by the current load-depth criterion.")
    if contact_atoms >= 30 and loading_ok and (not summary.get("has_unloading") or unloading_ok) and bool(summary.get("no_nan")):
        interpretation.append("This result is suitable as a first pure W nanoindentation production baseline, subject to the geometric hardness limitation below.")
    else:
        interpretation.append("This result is not yet a strong production baseline; the main concerns are listed in Warnings / Limitations.")

    warnings = [
        "Hardness uses the geometric spherical contact area A = pi(2Rh - h^2), not a calibrated Oliver-Pharr contact-area function.",
    ]
    if not bool(summary.get("plasticity_indicator_available")):
        warnings.append("A CSP/non-bcc/displacement plasticity proxy is not yet available, so plasticity is inferred only from the load-depth response and residual depth.")
    if contact_atoms < 30:
        warnings.append("The maximum number of contact atoms is below 30; hardness is not reliable for a physical production conclusion.")
    if final_load_ratio is not None and final_load_ratio > 0.1:
        warnings.append("The final unloading load is more than 10% of peak load; the indenter may still be in contact or residual elastic force remains.")
    if not bool(summary.get("no_nan")):
        warnings.append("The log contains NaN or Inf values and should not be used as a physical baseline.")
    if target_temperature is not None and mean_temperature is not None:
        if abs(float(mean_temperature) - float(target_temperature)) > 15.0:
            warnings.append("The reported mean temperature differs from the target by more than 15 K; fixed bottom atoms can lower the global temperature statistic, so a mobile-region temperature diagnostic is recommended.")

    lines = [
        "# Pure W Nanoindentation Report",
        "",
        "## System",
        f"- Material: {summary.get('system', 'pure W')}",
        f"- Structure: {summary.get('structure', 'not available')}",
        f"- Orientation: {summary.get('orientation', 'not available')}",
        f"- Number of atoms: {summary.get('n_atoms', 'not available')}",
        f"- Potential: {summary.get('eam', 'not available')}",
        f"- Device: {summary.get('device', 'not available')}",
        "",
        "## Indentation Protocol",
        f"- Temperature: {_fmt(summary.get('temperature_K'), 2, 'K')}",
        f"- Indenter radius: {_fmt(summary.get('indenter_radius_A'), 2, 'A')}",
        f"- Maximum target depth: {_fmt(summary.get('target_depth_A'), 2, 'A')}",
        f"- Loading steps: {summary.get('loading_steps', summary.get('steps', 'not available'))}",
        f"- Hold steps: {summary.get('hold_steps', 'not available')}",
        f"- Unloading steps: {summary.get('unload_steps', 'not available')}",
        f"- Bottom fixed thickness: {_fmt(summary.get('bottom_fixed_thickness_A'), 2, 'A')}",
        "",
        "## Main Results",
        f"- Maximum load: {_fmt(summary.get('max_load_nN'), 4, 'nN')}",
        f"- Maximum depth: {_fmt(summary.get('max_depth_A'), 4, 'A')}",
        f"- Residual depth: {_fmt(summary.get('residual_depth_A'), 4, 'A')}",
        f"- Unloading stiffness: {_fmt(summary.get('unloading_stiffness_nN_per_A'), 4, 'nN/A')}",
        f"- Estimated hardness: {_fmt(summary.get('hardness_GPa'), 4, 'GPa')}",
        f"- Contact atoms at max depth: {summary.get('max_contact_atoms', 'not available')}",
        f"- Pop-in detected: {_yes_no(summary.get('pop_in_detected'))}",
        f"- Mean measured temperature: {_fmt(summary.get('mean_temperature_K'), 2, 'K')}",
        f"- Maximum measured temperature: {_fmt(summary.get('max_temperature_K'), 2, 'K')}",
        f"- Final unloading load: {_fmt(summary.get('final_load_nN'), 4, 'nN')}",
        f"- Loading work: {_fmt(summary.get('work_loading'), 4, 'nN*A')}",
        f"- Unloading work: {_fmt(summary.get('work_unloading'), 4, 'nN*A')}",
        f"- Plastic work fraction: {_fmt(summary.get('plastic_work_fraction'), 4)}",
        "",
        "## Interpretation",
    ]
    lines.extend(f"- {item}" for item in interpretation)
    lines.extend(["", "## Warnings / Limitations"])
    lines.extend(f"- {item}" for item in warnings)
    lines.extend(
        [
            "",
            "## Output Files",
            f"- Log: {summary.get('csv', 'not available')}",
            f"- Load-depth plot: {summary.get('plot', 'not available')}",
            f"- Load-depth plot with pop-in marker: {summary.get('pop_in_plot', 'not available')}",
            f"- Trajectory: {summary.get('traj', 'not available')}",
            f"- Snapshots: {summary.get('snapshots_dir', 'snapshots/')}",
            f"- Snapshot PNGs: {summary.get('snapshots_png_dir', 'snapshots_png/')}",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output_path)
