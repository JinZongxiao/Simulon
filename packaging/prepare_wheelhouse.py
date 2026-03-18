from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WHEELHOUSE_ROOT = ROOT / "packaging" / "wheelhouse"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYG_WHEEL_BASE = "https://data.pyg.org/whl"
HEAVY_REQUIREMENTS = ROOT / "packaging" / "requirements-heavy.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-download heavy binary wheels used by the Simulon desktop Briefcase build.",
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["windows", "macOS"],
        help="Wheelhouse target that matches the Briefcase build target.",
    )
    parser.add_argument(
        "--python-tag",
        default=f"cp{sys.version_info.major}{sys.version_info.minor}",
        help="ABI tag suffix for the current interpreter, e.g. cp310 or cp311.",
    )
    parser.add_argument(
        "--torch-version",
        help="Explicit torch version for optional PyG operator wheels. If omitted, only base torch_geometric is vendored.",
    )
    parser.add_argument(
        "--with-pyg-ops",
        action="store_true",
        help="Download optional PyG operator wheels (pyg-lib/torch-scatter/torch-sparse/torch-cluster/torch-spline-conv).",
    )
    return parser.parse_args()


def wheelhouse_dir(target: str, python_tag: str) -> Path:
    return WHEELHOUSE_ROOT / f"{target}-{python_tag}"


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def validate_host_target(target: str) -> None:
    host = platform.system()
    if target == "macOS" and host != "Darwin":
        raise SystemExit("Prepare macOS wheelhouse on a macOS host so pip resolves the correct torch wheel.")
    if target == "windows" and host != "Windows":
        raise SystemExit("Prepare Windows wheelhouse on a Windows host so pip resolves the correct torch wheel.")


def download_base_wheels(target: str, python_tag: str) -> Path:
    destination = wheelhouse_dir(target, python_tag)
    destination.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--dest",
            str(destination),
            "--only-binary=:all:",
            "--extra-index-url",
            TORCH_CPU_INDEX,
            "-r",
            str(HEAVY_REQUIREMENTS),
        ]
    )
    return destination


def download_optional_pyg_ops(destination: Path, torch_version: str) -> None:
    torch_series = ".".join(torch_version.split(".")[:2])
    find_links = f"{PYG_WHEEL_BASE}/torch-{torch_series}.0+cpu.html"
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--dest",
            str(destination),
            "--only-binary=:all:",
            "--find-links",
            find_links,
            "pyg-lib",
            "torch-scatter",
            "torch-sparse",
            "torch-cluster",
            "torch-spline-conv",
        ]
    )


def main() -> None:
    args = parse_args()
    validate_host_target(args.target)
    destination = download_base_wheels(args.target, args.python_tag)
    if args.with_pyg_ops:
        if not args.torch_version:
            raise SystemExit("--with-pyg-ops requires --torch-version, e.g. 2.6.0 or 2.7.1.")
        download_optional_pyg_ops(destination, args.torch_version)
    print(f"Wheelhouse ready at: {destination}")


if __name__ == "__main__":
    main()
