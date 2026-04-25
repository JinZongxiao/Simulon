from __future__ import annotations

import csv
import json
from pathlib import Path


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
        rows.append(
            {
                "orientation": str(data.get("orientation", "unknown")),
                "temperature_k": float(data["temperature_k"]),
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
                "max_cmod_A": float(data["max_cmod_A"]),
                "final_cmod_A": float(data.get("final_cmod_A", 0.0)),
                "max_applied_strain": float(data.get("max_applied_strain", 0.0)),
                "cmod_at_peak_stress_A": float(data.get("cmod_at_peak_stress_A", 0.0)),
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
        "max_stress_bar",
        "peak_tensile_stress_bar",
        "peak_stress_magnitude_bar",
        "final_stress_bar",
        "max_cmod_A",
        "final_cmod_A",
        "max_applied_strain",
        "cmod_at_peak_stress_A",
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
        return {"n_runs": 0}
    orientations = sorted({row["orientation"] for row in rows})
    temps = [row["temperature_k"] for row in rows]
    return {
        "n_runs": len(rows),
        "orientations": orientations,
        "min_temperature_k": min(temps),
        "max_temperature_k": max(temps),
    }


def plot_dbtt(rows: list[dict], output_path: str | Path) -> str:
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
        retention = [row["stress_retention_ratio"] for row in ori_rows]
        axes[0].plot(temps, final_stress, marker="o", linewidth=1.8, label=orientation)
        axes[1].plot(temps, max_cmod, marker="o", linewidth=1.8, label=orientation)
        axes[2].plot(temps, retention, marker="o", linewidth=1.8, label=orientation)

    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Final opening stress, tension positive (bar)")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_ylabel("Max CMOD (A)")
    axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel("Temperature (K)")
    axes[2].set_ylabel("Stress retention ratio")
    axes[2].grid(True, alpha=0.3)
    if by_orientation:
        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)
