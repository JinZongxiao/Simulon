from __future__ import annotations

import csv
import json
from pathlib import Path

from postprocess.crack import classify_crack_response


def collect_dbtt_rows(root_dir: str | Path) -> list[dict]:
    root = Path(root_dir)
    rows = []
    for summary_path in root.rglob("summary.json"):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        stress_max = data.get("max_stress_bar", data.get("stress_max_bar"))
        if stress_max is None or "max_cmod_A" not in data or "temperature_k" not in data:
            continue
        stress_drop_ratio = float(data.get("stress_drop_ratio", 0.0))
        max_cmod_A = float(data["max_cmod_A"])
        peak_at_final = bool(data.get("peak_stress_at_final_step", True))
        if "classification" not in data:
            data.update(
                classify_crack_response(
                    {
                        "max_cmod_A": data.get("max_cmod_A", 0.0),
                        "max_crack_extension_A": data.get("max_crack_extension_A", 0.0),
                        "stress_drop_ratio": data.get("stress_drop_ratio", 0.0),
                        "peak_stress_at_final_step": data.get("peak_stress_at_final_step", True),
                        "plasticity_indicator_available": data.get("plasticity_indicator_available", False),
                        "plasticity_indicator": data.get("plasticity_indicator"),
                    }
                )
            )
        classification = str(data.get("classification", "invalid"))
        crack_opening_pass = bool(data.get("crack_opening_pass", False))
        significant_pass = bool(data.get("significant_crack_propagation_pass", False))
        physics_pass = bool(data.get("physics_acceptance_pass", False))
        rows.append(
            {
                "orientation": str(data.get("orientation", "unknown")),
                "temperature_k": float(data["temperature_k"]),
                "classification": classification,
                "max_stress_bar": float(stress_max),
                "peak_tensile_stress_bar": float(
                    data.get(
                        "peak_tensile_stress_bar",
                        data.get("peak_stress_magnitude_bar", abs(float(stress_max))),
                    )
                ),
                "peak_stress_magnitude_bar": float(
                    data.get("peak_stress_magnitude_bar", abs(float(stress_max)))
                ),
                "final_stress_bar": float(data.get("final_stress_bar", 0.0)),
                "stress_drop_ratio": stress_drop_ratio,
                "max_cmod_A": max_cmod_A,
                "final_cmod_A": float(data.get("final_cmod_A", 0.0)),
                "max_crack_length_A": float(data.get("max_crack_length_A", 0.0)),
                "max_crack_extension_A": float(data.get("max_crack_extension_A", 0.0)),
                "peak_stress_step": int(data.get("peak_stress_step", 0)),
                "peak_stress_at_final_step": peak_at_final,
                "crack_opening_pass": crack_opening_pass,
                "significant_crack_propagation_pass": significant_pass,
                "physics_acceptance_pass": physics_pass,
                "acceptance_pass": physics_pass,
                "max_applied_strain": float(data.get("max_applied_strain", 0.0)),
                "cmod_at_peak_stress_A": float(data.get("cmod_at_peak_stress_A", 0.0)),
                "plasticity_indicator_available": bool(data.get("plasticity_indicator_available", False)),
                "plasticity_indicator": data.get("plasticity_indicator"),
                "stress_retention_ratio": float(
                    data.get(
                        "stress_retention_ratio",
                        0.0 if abs(float(data.get("peak_stress_magnitude_bar", abs(float(stress_max))))) <= 1.0e-12
                        else float(data.get("final_stress_bar", 0.0)) / float(data.get("peak_stress_magnitude_bar", abs(float(stress_max)))),
                    )
                ),
                "fracture_work_proxy_bar_A": float(data.get("fracture_work_proxy_bar_A", 0.0)),
                "summary_path": str(summary_path),
            }
        )
    rows.sort(key=lambda row: (row["orientation"], row["temperature_k"]))
    return rows


def write_dbtt_csv(rows: list[dict], csv_path: str | Path) -> str:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "orientation",
        "temperature_k",
        "classification",
        "max_stress_bar",
        "peak_tensile_stress_bar",
        "peak_stress_magnitude_bar",
        "final_stress_bar",
        "stress_drop_ratio",
        "max_cmod_A",
        "final_cmod_A",
        "max_crack_length_A",
        "max_crack_extension_A",
        "peak_stress_step",
        "peak_stress_at_final_step",
        "crack_opening_pass",
        "significant_crack_propagation_pass",
        "physics_acceptance_pass",
        "acceptance_pass",
        "max_applied_strain",
        "cmod_at_peak_stress_A",
        "plasticity_indicator_available",
        "plasticity_indicator",
        "stress_retention_ratio",
        "fracture_work_proxy_bar_A",
        "summary_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(csv_path)


def summarize_dbtt(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n_runs": 0,
            "dbtt_workflow_pass": False,
            "dbtt_physics_pass": False,
            "dbtt_status": "insufficient_data",
            "dbtt_candidate_temperature_k": None,
            "classification_counts": {},
            "reason": "no per-temperature crack summaries were found",
        }
    orientations = sorted({row["orientation"] for row in rows})
    temps = [row["temperature_k"] for row in rows]
    classifications = [str(row.get("classification", "invalid")) for row in rows]
    classification_counts = {name: classifications.count(name) for name in sorted(set(classifications))}
    for name in ("brittle", "ductile", "opening_only", "no_crack_growth", "invalid"):
        classification_counts.setdefault(name, 0)
    workflow_pass = len(rows) > 0
    has_brittle = classification_counts["brittle"] > 0
    has_ductile = classification_counts["ductile"] > 0
    candidate_temp = None
    dbtt_status = "not_identified"
    dbtt_physics_pass = False
    reason = "All temperatures show similar opening-only response; no brittle-to-ductile mechanism contrast."
    if has_brittle and has_ductile:
        sorted_rows = sorted(rows, key=lambda row: row["temperature_k"])
        brittle_temps = [row["temperature_k"] for row in sorted_rows if row["classification"] == "brittle"]
        ductile_temps = [row["temperature_k"] for row in sorted_rows if row["classification"] == "ductile"]
        if brittle_temps and ductile_temps and min(ductile_temps) > min(brittle_temps):
            candidate_temp = min(ductile_temps)
            dbtt_status = "candidate_identified"
            dbtt_physics_pass = True
            reason = "Low-temperature brittle and higher-temperature ductile classifications are both present."
        else:
            reason = "Brittle and ductile labels are present but not ordered as a clear transition."
    elif len(set(classifications)) > 1:
        reason = "Multiple classifications are present, but no brittle-to-ductile transition can be assigned conservatively."
    return {
        "n_runs": len(rows),
        "orientations": orientations,
        "min_temperature_k": min(temps),
        "max_temperature_k": max(temps),
        "n_acceptance_pass": sum(1 for row in rows if row.get("acceptance_pass")),
        "dbtt_workflow_pass": workflow_pass,
        "dbtt_physics_pass": dbtt_physics_pass,
        "dbtt_status": dbtt_status,
        "dbtt_candidate_temperature_k": candidate_temp,
        "classification_counts": classification_counts,
        "reason": reason,
    }


def plot_dbtt(rows: list[dict], output_path: str | Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_orientation = {}
    for row in rows:
        by_orientation.setdefault(row["orientation"], []).append(row)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    for orientation, ori_rows in by_orientation.items():
        ori_rows = sorted(ori_rows, key=lambda row: row["temperature_k"])
        temps = [row["temperature_k"] for row in ori_rows]
        final_stress = [row["final_stress_bar"] for row in ori_rows]
        max_cmod = [row["max_cmod_A"] for row in ori_rows]
        stress_drop = [row["stress_drop_ratio"] for row in ori_rows]
        axes[0].plot(temps, final_stress, marker="o", linewidth=1.8, label=orientation)
        axes[1].plot(temps, max_cmod, marker="o", linewidth=1.8, label=orientation)
        axes[2].plot(temps, stress_drop, marker="o", linewidth=1.8, label=orientation)

    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Final opening stress, tension positive (bar)")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_ylabel("Max CMOD (A)")
    axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel("Temperature (K)")
    axes[2].set_ylabel("Stress drop ratio")
    axes[2].grid(True, alpha=0.3)
    if by_orientation:
        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def plot_dbtt_mechanism(rows: list[dict], output_path: str | Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return str(output_path)
    rows = sorted(rows, key=lambda row: row["temperature_k"])
    temps = [row["temperature_k"] for row in rows]
    crack_extensions = [row["max_crack_extension_A"] for row in rows]
    classifications = [row["classification"] for row in rows]
    class_map = {"no_crack_growth": 0, "opening_only": 1, "brittle": 2, "ductile": 3, "invalid": -1}
    class_values = [class_map.get(value, -1) for value in classifications]
    plasticity_available = all(bool(row.get("plasticity_indicator_available")) for row in rows)
    if plasticity_available:
        plasticity = [float(row["plasticity_indicator"]) for row in rows]
    else:
        plasticity = [0.0 for _ in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    axes[0].plot(temps, crack_extensions, marker="o", linewidth=1.8, color="#2ca02c")
    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Crack extension (A)")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(temps, class_values, s=70, color="#1f77b4")
    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_ylabel("Classification")
    axes[1].set_yticks([-1, 0, 1, 2, 3])
    axes[1].set_yticklabels(["invalid", "no growth", "opening", "brittle", "ductile"])
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(temps, plasticity, marker="o", linewidth=1.8, color="#d62728")
    axes[2].set_xlabel("Temperature (K)")
    axes[2].set_ylabel("Plasticity indicator")
    if not plasticity_available:
        axes[2].text(0.5, 0.5, "not available", transform=axes[2].transAxes, ha="center", va="center")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)
