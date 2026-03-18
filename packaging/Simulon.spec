# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

repo_root = Path.cwd()

datas = [
    (str(repo_root / "run_scripts"), "run_scripts"),
    (str(repo_root / "run_data"), "run_data"),
]

hiddenimports = collect_submodules("torch_geometric") + [
    "torch",
    "numpy",
    "scipy",
    "matplotlib",
    "ase",
    "pymatgen",
]

block_cipher = None


a = Analysis(
    ["simulon_desktop.py"],
    pathex=[str(repo_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SimulonDesktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SimulonDesktop",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="SimulonDesktop.app",
        icon=None,
        bundle_identifier="com.simulon.desktop",
    )
