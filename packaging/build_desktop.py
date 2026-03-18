from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / "Simulon.spec"
DIST = ROOT / "dist"


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        str(SPEC),
    ]
    print(f"Building desktop app for {platform.system()} using: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"Build finished. Artifacts are under: {DIST}")


if __name__ == "__main__":
    main()
