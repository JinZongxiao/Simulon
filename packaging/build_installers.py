from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = "macOS" if sys.platform == "darwin" else "windows"
DEFAULT_FORMAT = "dmg" if sys.platform == "darwin" else "msi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a native installable Simulon desktop app with Briefcase.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        choices=["windows", "macOS"],
        help="Target platform name passed to Briefcase.",
    )
    parser.add_argument(
        "--format",
        dest="package_format",
        default=DEFAULT_FORMAT,
        help="Installer/package format such as msi, dmg, or pkg.",
    )
    parser.add_argument(
        "--app",
        default="desktop_app",
        help="Briefcase app identifier to build.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Run briefcase update before building so dependency/app changes are refreshed.",
    )
    parser.add_argument(
        "--skip-wheelhouse",
        action="store_true",
        help="Skip pre-downloading heavy binary wheels before running Briefcase.",
    )
    parser.add_argument(
        "--with-pyg-ops",
        action="store_true",
        help="Also vendor optional PyG native extension wheels from data.pyg.org.",
    )
    parser.add_argument(
        "--torch-version",
        help="Optional explicit torch version to use when downloading optional PyG native wheels.",
    )
    return parser.parse_args()


def run_briefcase(command: list[str]) -> None:
    print(f"+ {' '.join(command)}")
    subprocess.run(command, check=True, cwd=ROOT)


def validate_host_target(target: str) -> None:
    host = platform.system()
    if target == "macOS" and host != "Darwin":
        raise SystemExit("Briefcase macOS installers must be built on macOS hosts.")
    if target == "windows" and host != "Windows":
        raise SystemExit("Briefcase Windows installers must be built on Windows hosts.")


def prepare_wheelhouse(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "packaging/prepare_wheelhouse.py",
        "--target",
        args.target,
    ]
    if args.with_pyg_ops:
        command.append("--with-pyg-ops")
    if args.torch_version:
        command.extend(["--torch-version", args.torch_version])
    run_briefcase(command)


def main() -> None:
    args = parse_args()
    validate_host_target(args.target)
    print(f"Building installable desktop app for {args.target} on host {platform.system()}.")

    if not args.skip_wheelhouse:
        prepare_wheelhouse(args)

    base = [sys.executable, "-m", "briefcase"]
    run_briefcase(base + ["create", args.target, "-a", args.app])

    if args.update:
        run_briefcase(base + ["update", args.target, "-a", args.app])

    run_briefcase(base + ["build", args.target, "-a", args.app])
    run_briefcase(
        base
        + [
            "package",
            args.target,
            "-a",
            args.app,
            "-p",
            args.package_format,
        ]
    )

    print("Briefcase packaging finished.")


if __name__ == "__main__":
    main()
