from __future__ import annotations

import csv
import math
from pathlib import Path


_CLASSIFICATION_ORDER = ("brittle", "ductile", "opening_only", "no_crack_growth", "invalid")


def _finite_values(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def classify_crack_response(metrics: dict) -> dict:
    max_cmod = float(metrics.get("max_cmod_A", 0.0))
    crack_extension = float(metrics.get("max_crack_extension_A", 0.0))
    stress_drop = float(metrics.get("stress_drop_ratio", 0.0))
    peak_at_final = bool(metrics.get("peak_stress_at_final_step", True))
    plasticity_available = bool(metrics.get("plasticity_indicator_available", False))
    plasticity = metrics.get("plasticity_indicator")
    plasticity_high = False
    if plasticity_available and plasticity is not None:
        plasticity_high = float(plasticity) >= float(metrics.get("plasticity_high_threshold", 1.0))

    crack_opening_pass = max_cmod >= 1.0 and stress_drop >= 0.1 and not peak_at_final
    significant_propagation_pass = (
        crack_extension >= 2.0 and max_cmod >= 3.0 and stress_drop >= 0.15 and not peak_at_final
    )

    if not _finite_values([max_cmod, crack_extension, stress_drop]):
        classification = "invalid"
        reason = "non-finite crack metrics"
    elif max_cmod < 1.0 and crack_extension < 2.0 and peak_at_final:
        classification = "no_crack_growth"
        reason = "CMOD < 1 A, crack extension < 2 A, and peak stress occurs at the final step"
    elif crack_extension >= 5.0 and stress_drop >= 0.15 and not plasticity_high:
        classification = "brittle"
        if plasticity_available:
            reason = "large crack extension with post-peak drop and low plasticity proxy"
        else:
            reason = "large crack extension with post-peak drop; plasticity indicator unavailable"
    elif plasticity_high and max_cmod >= 1.0 and crack_extension < 5.0:
        classification = "ductile"
        reason = "large opening with high plasticity proxy and limited crack extension"
    elif max_cmod >= 1.0 and stress_drop >= 0.1 and crack_extension < 2.0:
        classification = "opening_only"
        reason = "post-peak opening response without significant crack extension"
    elif significant_propagation_pass:
        classification = "invalid"
        reason = "significant propagation detected but brittle/ductile mechanism cannot be assigned without plasticity"
    else:
        classification = "invalid"
        reason = "metrics do not satisfy no-growth, opening-only, brittle, or ductile criteria"

    return {
        "classification": classification,
        "classification_reason": reason,
        "crack_opening_pass": crack_opening_pass,
        "significant_crack_propagation_pass": significant_propagation_pass,
        "physics_acceptance_pass": classification in {"brittle", "ductile", "opening_only"},
        "plasticity_indicator_available": plasticity_available,
        "plasticity_indicator": plasticity,
        "plasticity_indicator_note": metrics.get(
            "plasticity_indicator_note",
            "not computed; ductile classification is conservative",
        ),
    }


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
    metrics = {
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
        "plasticity_indicator_available": False,
        "plasticity_indicator": None,
        "plasticity_indicator_note": "not computed; no DXA/CSP/non-affine plasticity proxy is available in this workflow yet",
    }
    metrics.update(classify_crack_response(metrics))
    return metrics


def _read_crack_rows(csv_path: str | Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_xyz_frames(path: str | Path) -> list[dict]:
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            n_atoms = int(line)
            comment = f.readline().strip()
            atom_types = []
            coords = []
            for _ in range(n_atoms):
                parts = f.readline().split()
                atom_types.append(parts[0])
                coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
            frames.append({"comment": comment, "atom_types": atom_types, "coords": coords})
    return frames


def _estimate_crack_length_numpy(
    coords: list[tuple[float, float, float]],
    crack_plane_y: float,
    x_min: float,
    x_max: float,
    x_center: float,
    initial_half_length: float,
    open_threshold_A: float,
    bins: int,
) -> tuple[float, float]:
    bins = max(8, int(bins))
    span = max(1.0e-12, float(x_max - x_min))
    upper_min = [math.inf] * bins
    lower_max = [-math.inf] * bins
    for x, y, _z in coords:
        idx = min(bins - 1, max(0, int(math.floor((x - x_min) / span * bins))))
        if y > crack_plane_y:
            upper_min[idx] = min(upper_min[idx], y)
        elif y < crack_plane_y:
            lower_max[idx] = max(lower_max[idx], y)

    open_bins = []
    width = span / bins
    for i in range(bins):
        finite = math.isfinite(upper_min[i]) and math.isfinite(lower_max[i])
        open_bins.append(finite and (upper_min[i] - lower_max[i]) >= open_threshold_A)
    if not any(open_bins):
        return 0.0, 0.0

    seeds = []
    for i, is_open in enumerate(open_bins):
        center = x_min + (i + 0.5) * width
        if is_open and x_center - initial_half_length <= center <= x_center + initial_half_length:
            seeds.append(i)
    if not seeds:
        return 0.0, 0.0

    left = min(seeds)
    right = max(seeds)
    while left > 0 and open_bins[left - 1]:
        left -= 1
    while right < bins - 1 and open_bins[right + 1]:
        right += 1
    crack_length = (right - left + 1) * width
    crack_extension = max(0.0, crack_length - 2.0 * initial_half_length)
    return crack_length, crack_extension


def analyze_crack_tracking_sensitivity(
    csv_path: str | Path,
    traj_path: str | Path,
    output_dir: str | Path,
    crack_half_length_A: float,
    crack_plane_y_A: float,
    x_min_A: float,
    x_max_A: float,
    x_center_A: float,
    bins: int,
    thresholds: tuple[float, ...] = (0.5, 0.8, 1.0, 1.2, 1.5),
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_crack_rows(csv_path)
    frames = _read_xyz_frames(traj_path)
    if not rows or not frames:
        return {
            "crack_tracking_reliable": False,
            "crack_tracking_note": "trajectory or crack CSV is empty; sensitivity analysis skipped",
        }

    max_cmod = max(float(row["cmod_A"]) for row in rows)
    final_cmod = float(rows[-1]["cmod_A"])
    stress_drop = summarize_crack(csv_path).get("stress_drop_ratio", 0.0)
    peak_at_final = summarize_crack(csv_path).get("peak_stress_at_final_step", True)
    sensitivity_rows = []
    for threshold in thresholds:
        lengths = []
        extensions = []
        for frame in frames:
            length, extension = _estimate_crack_length_numpy(
                frame["coords"],
                crack_plane_y=crack_plane_y_A,
                x_min=x_min_A,
                x_max=x_max_A,
                x_center=x_center_A,
                initial_half_length=crack_half_length_A,
                open_threshold_A=threshold,
                bins=bins,
            )
            lengths.append(length)
            extensions.append(extension)
        metrics = {
            "max_cmod_A": max_cmod,
            "max_crack_extension_A": max(extensions),
            "stress_drop_ratio": stress_drop,
            "peak_stress_at_final_step": peak_at_final,
            "plasticity_indicator_available": False,
            "plasticity_indicator": None,
        }
        classification = classify_crack_response(metrics)["classification"]
        sensitivity_rows.append(
            {
                "threshold_A": threshold,
                "max_CMOD_A": max_cmod,
                "final_CMOD_A": final_cmod,
                "max_crack_length_A": max(lengths),
                "final_crack_length_A": lengths[-1],
                "crack_extension_A": max(extensions),
                "classification": classification,
            }
        )

    csv_out = output_dir / "crack_tracking_sensitivity.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        fields = [
            "threshold_A",
            "max_CMOD_A",
            "final_CMOD_A",
            "max_crack_length_A",
            "final_crack_length_A",
            "crack_extension_A",
            "classification",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sensitivity_rows)

    plot_out = output_dir / "crack_tracking_sensitivity.png"
    plot_crack_tracking_sensitivity(csv_out, plot_out)
    extensions = [row["crack_extension_A"] for row in sensitivity_rows]
    classifications = {row["classification"] for row in sensitivity_rows}
    ext_range = max(extensions) - min(extensions)
    reliable = ext_range <= 2.0 and len(classifications) == 1
    note = "crack extension is stable across thresholds"
    if not reliable:
        note = "crack extension is threshold-sensitive"
    return {
        "crack_tracking_sensitivity_csv": str(csv_out),
        "crack_tracking_sensitivity_plot": str(plot_out),
        "crack_tracking_reliable": reliable,
        "crack_tracking_note": note,
        "crack_tracking_extension_range_A": ext_range,
    }


def plot_crack_tracking_sensitivity(csv_path: str | Path, output_path: str | Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_crack_rows(csv_path)
    thresholds = [float(row["threshold_A"]) for row in rows]
    extensions = [float(row["crack_extension_A"]) for row in rows]
    lengths = [float(row["max_crack_length_A"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].plot(thresholds, extensions, marker="o", linewidth=1.8)
    axes[0].set_xlabel("Open threshold (A)")
    axes[0].set_ylabel("Crack extension (A)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(thresholds, lengths, marker="o", linewidth=1.8, color="#2ca02c")
    axes[1].set_xlabel("Open threshold (A)")
    axes[1].set_ylabel("Max crack length (A)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def plot_crack(csv_path: str | Path, output_path: str | Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
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
