import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from postprocess.dbtt import collect_dbtt_rows, plot_dbtt, summarize_dbtt, write_dbtt_csv


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rebuild DBTT summary files from existing per-temperature outputs")
    p.add_argument("root_dir", help="DBTT root directory containing T_*K/orientation_*/summary.json")
    p.add_argument("--orientation", default="custom")
    p.add_argument("--temperatures", default="100,200,300,400,500,600")
    return p


def main():
    args = _build_parser().parse_args()
    root = Path(args.root_dir)
    rows = collect_dbtt_rows(root)
    csv_path = root / "dbtt_summary.csv"
    json_path = root / "dbtt_summary.json"
    png_path = root / "dbtt_summary.png"

    write_dbtt_csv(rows, csv_path)
    plot_dbtt(rows, png_path)

    temps = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]
    summary = summarize_dbtt(rows)
    summary.update(
        {
            "orientation": str(args.orientation),
            "temperatures_k": temps,
            "temperature_scale": 1.0,
            "csv": str(csv_path),
            "plot": str(png_path),
            "runs": [str(p.parent) for p in sorted(root.glob("T_*K/orientation_*/summary.json"))],
            "smoke": False,
        }
    )
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Rebuilt DBTT summary for {len(rows)} runs")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
