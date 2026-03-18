# Simulon

[English] | [中文](README_zh-CN.md)

A lightweight, PyTorch-powered molecular dynamics (MD) engine with optional custom CUDA kernels. Simulon focuses on clarity, extensibility, and practical MD workflows for research and engineering.

Key capabilities
- PyTorch-first design: all state lives in tensors; easy CPU/GPU switching.
- Optional CUDA acceleration: Lennard-Jones, EAM and neighbor search via a compiled extension (`simulon_cuda`).
- Modular force fields: Lennard-Jones (CPU/CUDA), EAM (CUDA), Born–Mayer–Huggins, and a user-defined pair-potential template.
- Neighbor search: cell/PBC handling and GPU-accelerated options.
- MD integrator: velocity-Verlet in NVT with friction (gamma), simple and robust.
- I/O and utilities: XYZ reader/writer, logging, diagnostics plots, and helpers for dataset preparation.
- ML potentials: example integration to fine-tune CHGNet-like models and run MD with learned forces.

What’s in this repo
- `core/`: MD building blocks: forces, integrators, simulators, analyzers.
- `io_utils/`: file readers/writers, loggers, parsers (ASE/pymatgen), EAM parser, structure utilities.
- `cuda source/`: C++/CUDA kernels for LJ/EAM/neighbor search and Python bindings.
- `run_scripts/`: ready-to-run examples: LJ, user-defined potentials, ML potentials, and plotting diagnostics.
- `run_data/`: small example systems (Ar, Cu, W, etc.).
- `simulation_agent/`: interactive CN/EN assistants that help generate and analyze MD runs.
- `cuda_test/`: minimal tests and examples for CUDA backends.

Requirements
- Python 3.10+ (3.11 tested)
- PyTorch (CUDA optional). See https://pytorch.org for install instructions.
- Python packages: numpy, scipy, matplotlib, ase, pymatgen, tqdm
- Optional (for ML examples): chgnet

Quick install
- Base Python deps:
  - `pip install torch` (per your CUDA/CPU environment)
  - `pip install numpy scipy matplotlib ase pymatgen tqdm`
  - Optional: `pip install chgnet`
- CUDA extension (optional but recommended for performance):
  - Prereqs: a working C++ toolchain and CUDA toolkit matching your PyTorch build.
  - Build in place: `python setup.py build_ext --inplace`
  - Windows users: make sure MSVC Build Tools and CUDA are on PATH.
  - Note: a prebuilt `simulon_cuda.cp311-win_amd64.pyd` is included for Python 3.11 on Windows; it may work if your environment matches. Otherwise, build from source.

Desktop app (Windows/macOS)
- A cross-platform Tk desktop GUI is available at `simulon_desktop.py`. It lets you load/edit the bundled JSON templates, choose an output folder, and run Lennard-Jones or user-defined pair-potential simulations without using the command line.
- Start from source:
  - `python simulon_desktop.py`
- Build a native-installable desktop application with Briefcase on the target OS:
  - `pip install -r packaging/requirements-desktop.txt`
  - Pre-download heavy binary wheels used by the installer build:
    - `python packaging/prepare_wheelhouse.py --target windows`
    - `python packaging/prepare_wheelhouse.py --target macOS`
    - Optional PyG operator wheels: `python packaging/prepare_wheelhouse.py --target windows --with-pyg-ops --torch-version 2.6.0`
  - Windows installer build: `python packaging/build_installers.py --target windows --format msi`
  - macOS installer build: `python packaging/build_installers.py --target macOS --format dmg`
  - Optional macOS PKG build: `python packaging/build_installers.py --target macOS --format pkg`
- Output artifacts:
  - Windows: an installable `.msi`
  - macOS: an installable `.dmg` or `.pkg`
- Packaging notes:
  - The Briefcase project config lives in `pyproject.toml`, so the app is described as a native installable application instead of a portable executable bundle.
  - Briefcase now looks in `packaging/wheelhouse/<target>-cpXY/` first for `torch` and `torch_geometric` wheels, then falls back to the official PyTorch CPU wheel index. This makes heavy dependency resolution much more deterministic during installer builds.
  - Per the official PyG installation docs, `torch_geometric` itself can run without the optional compiled extension packages; if you need those accelerated ops, vendor them into the wheelhouse with `--with-pyg-ops`.
  - The build bundles `run_scripts/` JSON templates and `run_data/` sample systems so the installed app can be opened and used directly after installation.
  - Build on Windows for Windows installers, and on macOS for macOS installers.

Quick start
1) Lennard-Jones MD
- Edit `run_scripts/lj_run.json` if needed (structure, box length, LJ parameters, cutoff, temperature, steps, output directory).
- Run:
  - `python run_scripts/lj_run.py --config run_scripts/lj_run.json`
- Outputs (saved to `run_data/output/` by default):
  - Energy curve PNG, trajectory `MD_traj_<timestamp>.xyz`, forces `forces_<timestamp>.xyz`, and logs.

2) User-defined pair potential
- Define your formula in `run_scripts/user_defined_run.json`, e.g. `0.5 * k * (r - 1)**2` and set per-pair parameters.
- Run:
  - `python run_scripts/user_defined_run.py --config run_scripts/user_defined_run.json`

3) ML potentials (example)
- Edit `run_scripts/mlps_run.json` to point to your AIMD position/force files and training hyperparameters.
- Requires extra deps (e.g., `chgnet`).
- Run:
  - `python run_scripts/mlps_run.py --config run_scripts/mlps_run.json`

4) Diagnostics and plots
- Produce a comprehensive set of plots (energy/forces/MSD/RDF/degree, etc.):
  - `python run_scripts/plot_md_diagnostics.py --steps 500`
- Plots are written under `run_data/output/plots_YYYYmmdd_HHMMSS/`.

Configuration highlights
- Common fields you will see in JSON configs:
  - `data_path_xyz`: input structure (.xyz)
  - `box_length`: cubic box length (Angstrom)
  - `pair_parameter`: per-pair parameters; for LJ use `epsilon` and `sigma`; for custom potentials, your symbol names (e.g., `k`).
  - `potential_formula`: for the user-defined force (e.g., `0.5 * k * (r - 1)**2`).
  - `cut_off`: neighbor cutoff (Angstrom)
  - `dt`: time step (fs or arbitrary units as chosen consistently in your setup)
  - `temperature`: NVT target, can be a scalar or per-type vector
  - `gamma`: friction coefficient for NVT
  - `num_steps`, `print_interval`, `output_save_path`

Tips
- GPU vs CPU: the code picks CUDA if available; otherwise, it runs on CPU.
- Periodic boundary conditions are handled internally; keep `box_length` consistent with your system.
- For large systems, prefer the CUDA extension for better performance.

Troubleshooting
- CUDA build errors: verify your CUDA toolkit version matches the PyTorch CUDA version and that MSVC/Clang toolchains are properly installed.
- Missing modules: check you installed all required Python packages and that your `PYTHONPATH` includes the repo root when running scripts.

Contributing
- Issues and PRs are welcome. Please include a minimal repro or a small input structure when reporting problems.
