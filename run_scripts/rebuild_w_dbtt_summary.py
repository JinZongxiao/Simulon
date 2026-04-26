import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from postprocess.dbtt import collect_dbtt_rows, plot_dbtt, plot_dbtt_mechanism, summarize_dbtt, write_dbtt_csv


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rebuild a W DBTT summary from per-temperature crack summary.json files")
    p.add_argument("root_dir", help="DBTT output directory containing T_* subdirectories")
    return p


def main():
    args = _build_parser().parse_args()
    root = Path(args.root_dir)
    rows = collect_dbtt_rows(root)
    csv_path = root / "dbtt_summary.csv"
    json_path = root / "dbtt_summary.json"
    plot_path = root / "dbtt_summary.png"
    mechanism_plot_path = root / "dbtt_mechanism_summary.png"
    write_dbtt_csv(rows, csv_path)
    plot_dbtt(rows, plot_path)
    plot_dbtt_mechanism(rows, mechanism_plot_path)
    summary = summarize_dbtt(rows)
    summary.update(
        {
            "csv": str(csv_path),
            "plot": str(plot_path),
            "mechanism_plot": str(mechanism_plot_path),
        }
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"DBTT summary: {csv_path}")
    print(f"DBTT plot: {plot_path}")
    print(f"DBTT mechanism plot: {mechanism_plot_path}")
    print(f"DBTT json: {json_path}")


if __name__ == "__main__":
    main()
