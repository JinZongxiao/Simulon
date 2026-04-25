# Simulon

[English] | [中文](README_zh-CN.md)

A lightweight, PyTorch-powered molecular dynamics (MD) engine with optional custom CUDA kernels. Simulon targets clarity, extensibility, and practical MD workflows for research and engineering.

---

## What's new (latest update)

| Area | Changes |
|------|---------|
| **Ensembles** | NVE (micro-canonical), NVT (Langevin), **NPT (Berendsen barostat)** |
| **Triclinic box** | Full 3×3 H-matrix PBC via `core/box.py` — orthogonal and non-orthogonal cells unified |
| **Restart** | `save_checkpoint` / `load_checkpoint` — preserves coordinates, velocities, box, RNG state |
| **Force fields** | All force fields (LJ, EAM, BMH) now return `virial` for NPT pressure coupling |
| **Neighbor search** | Fixed duplicate-edge bug; Box-aware minimum image; CUDA kernel O(N²)→O(1) prefix-sum fix |
| **BMH** | Fully rewritten: edge-based analytical forces, eliminates O(N²) memory allocation |
| **EAM** | Dead-code removal; vectorized table lookup (no Python loops); virial added |
| **W tensile** | Added `run_scripts/w_tensile.py` with tensor stress output, oriented BCC-W generation, stress-strain plotting, and anisotropic lateral NPT support |
| **W indentation** | Added `run_scripts/w_indent.py` with spherical indenter loading, fixed-bottom W slabs, load-depth output, and smoke-test coverage |
| **W crack** | Added `run_scripts/w_crack.py` with center precrack generation, rigid-grip opening, stress-CMOD output, and smoke-test coverage |
| **W DBTT** | Added `run_scripts/w_dbtt_scan.py` and `postprocess/dbtt.py` for crack-based temperature scans and DBTT trend plots |
| **Performance** | ~384 steps/s on RTX 3050 for 100-atom Ar NVT |

---

## Key capabilities

- **PyTorch-first**: all state lives in tensors; CPU/GPU switching is a single flag.
- **Three ensembles**: NVE, NVT (Langevin thermostat), NPT (Berendsen barostat + Langevin).
- **Tensor stress support**: force fields now return scalar virial and virial tensor; tensile workflows use full stress tensors.
- **Triclinic PBC**: unified `Box` class handles cubic, orthorhombic, and fully triclinic cells via a 3×3 lattice matrix.
- **Verlet neighbor list**: lazy rebuild triggered by displacement threshold (skin/2); GPU-accelerated via optional CUDA extension.
- **Modular force fields**: Lennard-Jones, EAM, Born–Mayer–Huggins, and a user-defined pair-potential template.
- **W mechanical workflows**: built-in tungsten tensile, nanoindentation, and crack-opening scripts with `[100]/[110]/[111]` oriented cell generation, CSV/PNG output, and smoke-test coverage.
- **Restart**: full checkpoint/resume support — save every N steps, resume without re-equilibrating.
- **RDF analyzer**: online accumulation with correct normalization for same-species and cross-species pairs.
- **I/O & utilities**: XYZ reader/writer, CSV energy logger, trajectory dump, EAM table parser, pymatgen/ASE integration.
- **ML potentials**: example integration for CHGNet-like models.

---

## Repository layout

```
core/
  box.py                  # Unified orthogonal + triclinic PBC (H-matrix)
  barostat.py             # Isotropic Berendsen + diagonal anisotropic NPT barostats
  mechanics/loading.py    # Uniaxial tensile loader
  md_model.py             # SumBackboneInterface, BaseModel (main MD loop)
  md_simulation.py        # MDSimulator: run loop, logging, trajectory dump
  analyser.py             # RDF accumulator
  energy_minimizer.py     # Steepest-descent minimizer
  force/
    lennard_jones_force.py
    eam_force.py
    born_mayer_huggins_force.py
    template/pair_force_template.py
  integrator/integrator.py  # Velocity-Verlet (NVE / NVT / NPT)
  neighbor_search/gpu_kdtree.py

io_utils/
  reader.py               # AtomFileReader: XYZ → tensors + neighbor list
  w_bcc.py                # Oriented BCC-W structure generator
  restart.py              # save_checkpoint / load_checkpoint
  writer.py / output_logger.py / eam_parser.py / ...

postprocess/
  stress_strain.py        # Stress-strain summary + PNG plot
  indentation.py          # Load-depth summary + PNG plot
  crack.py                # Stress-CMOD summary + PNG plot
  dbtt.py                 # Temperature-scan aggregation + PNG plot

cuda source/
  neighbor_search_kernel.cu
  lj_energy_force*.cu
  eam_cuda_ext*.cu

run_scripts/
  demo_ar_nvt.py          # Quick demo: 100-atom Ar NVT
  lj_run.py               # JSON-driven LJ simulation
  user_defined_run.py
  mlps_run.py
  w_tensile.py            # Tungsten tensile workflow
  w_indent.py             # Tungsten nanoindentation workflow
  w_crack.py              # Tungsten crack-opening workflow
  w_dbtt_scan.py          # Tungsten DBTT temperature scan
  w_batch_report.py       # Combined W workflow runner + report export
  check_w_orientation.py  # Static sanity check for oriented BCC-W cells
  plot_md_diagnostics.py

run_data/                 # Example structures (Ar, Cu, W, …)
simulation_agent/         # CN/EN interactive MD assistants
```

---

## Requirements

- Python 3.10+ (3.11 tested)
- PyTorch ≥ 2.0 (CUDA optional). See https://pytorch.org
- `numpy scipy matplotlib ase pymatgen tqdm torch_geometric`
- Optional (ML examples): `chgnet`

---

## Installation

```bash
# 1. Python dependencies
pip install torch torchvision torchaudio          # adjust for your CUDA version
pip install numpy scipy matplotlib ase pymatgen tqdm
pip install torch_geometric

# 2. CUDA extension (optional, recommended for large systems)
#    Requires: MSVC Build Tools (Windows) or GCC, + matching CUDA toolkit
python setup.py build_ext --inplace
```

> **Windows note**: A prebuilt `simulon_cuda.cp311-win_amd64.pyd` is included for Python 3.11 + CUDA 12.x. If your environment differs, rebuild from source.

---

## Quick start

### 1. Instant demo — Ar NVT

```bash
python run_scripts/demo_ar_nvt.py
```

100 Ar atoms, FCC structure, LJ force field, Langevin NVT at 90 K, 500 steps. Outputs trajectory and energy CSV to `run_output/demo_ar_nvt/`.

### 2. JSON-driven LJ run

```bash
python run_scripts/lj_run.py --config run_scripts/lj_run.json
```

Edit `lj_run.json` to change structure, box, LJ parameters, ensemble, temperature, and output path.

### 3. NPT simulation (Python API)

```python
from io_utils.reader import AtomFileReader
from core.force.lennard_jones_force import LennardJonesForce
from core.md_model import SumBackboneInterface, BaseModel
from core.integrator.integrator import VerletIntegrator
from core.barostat import BerendsenBarostat
from core.md_simulation import MDSimulator

mol   = AtomFileReader('structure.xyz', box_length=30.0, cutoff=10.0,
                       parameter={"[0 0]": {"epsilon": 0.0104, "sigma": 3.4}})
ff    = LennardJonesForce(mol)
integ = VerletIntegrator(mol, dt=0.001, ensemble='NPT',
                         temperature=(300, 300), gamma=0.01)
baro  = BerendsenBarostat(mol, target_pressure=1.0, tau_p=0.5)
model = BaseModel(SumBackboneInterface([ff], mol), integ, mol, barostat=baro)

MDSimulator(model, num_steps=5000, print_interval=100).run()
```

### 4. Triclinic cell

```python
from core.box import Box
import torch

H = torch.tensor([[a, 0, 0],
                  [b*cos(gamma), b*sin(gamma), 0],
                  [...]])          # any valid lattice matrix
mol = AtomFileReader('structure.xyz', box_length=a, box_vectors=H, ...)
```

### 5. Restart / checkpoint

```python
from io_utils.restart import save_checkpoint, load_checkpoint

# save every 1000 steps
save_checkpoint(model, step=1000, path='ckpt.pt')

# resume
next_step = load_checkpoint(model, path='ckpt.pt')
for step in range(next_step, total_steps):
    model()
```

### 6. W tensile workflow

Minimal smoke test:

```bash
python run_scripts/w_tensile.py --smoke
python cuda_test/test_w_tensile_smoke.py
```

Recommended baseline for W `[100]` tensile with anisotropic lateral NPT:

```bash
python run_scripts/w_tensile.py \
  --orientation 100 \
  --replicas 4,4,3 \
  --lateral-mode stress-free \
  --steps 5000 \
  --strain-rate 0.00005 \
  --barostat-tau 0.1 \
  --barostat-gamma 1.0 \
  --gamma 2.0
```

Run all three common W tensile orientations without overwriting results:

```bash
python run_scripts/w_tensile.py --orientation 100 --replicas 4,4,3 --lateral-mode stress-free --steps 5000 --strain-rate 0.00005 --barostat-tau 0.1 --barostat-gamma 1.0 --gamma 2.0 --output-dir run_output/w_tensile
python run_scripts/w_tensile.py --orientation 110 --replicas 4,4,3 --lateral-mode stress-free --steps 5000 --strain-rate 0.00005 --barostat-tau 0.1 --barostat-gamma 1.0 --gamma 2.0 --output-dir run_output/w_tensile
python run_scripts/w_tensile.py --orientation 111 --replicas 3,3,2 --lateral-mode stress-free --steps 5000 --strain-rate 0.00005 --barostat-tau 0.1 --barostat-gamma 1.0 --gamma 2.0 --output-dir run_output/w_tensile
```

Static orientation sanity check:

```bash
python run_scripts/check_w_orientation.py --orientation all
```

Outputs are grouped by orientation:

- `run_output/w_tensile/orientation_100/`
- `run_output/w_tensile/orientation_110/`
- `run_output/w_tensile/orientation_111/`

Each tensile output directory contains:

- `stress_strain.csv`
- `summary.json`
- `stress_strain.png`
- generated oriented structure, e.g. `W_100_generated.xyz`

The CSV includes signed stress columns (`stress_xx_bar`, `stress_yy_bar`, `stress_zz_bar`), tension-positive columns (`tension_xx_bar`, `tension_yy_bar`, `tension_zz_bar`), box lengths, energy, temperature, and virial tensor diagonals.

The updated tensile workflow now:

- performs zero-pressure equilibration before loading via `--equil-steps`
- reports stress relative to the equilibrated initial state in `stress_xx_bar`
- writes tension-positive presentation columns as `tension_xx_bar`, `tension_yy_bar`, `tension_zz_bar`
- also keeps absolute stress columns as `stress_xx_abs_bar`, `stress_yy_abs_bar`, `stress_zz_abs_bar`
- stabilizes the anisotropic lateral pressure controller with `--barostat-compressibility-bar-inv` and `--barostat-pressure-tolerance-bar`
- aborts if the lateral box runs away beyond `--max-lateral-box-ratio`
- can dump an XYZ trajectory with `--traj-interval`

For large `--orientation custom` runs, still check `initial_stress_xx_abs_bar`, `initial_stress_yy_abs_bar`, and `initial_stress_zz_abs_bar` in `summary.json`. If they remain large after equilibration, increase `--equil-steps` or retune the barostat parameters before trusting the tensile curve.

Large custom-structure example on a server:

```bash
python run_scripts/w_tensile.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --steps 100000 \
  --equil-steps 1000 \
  --strain-rate 0.0004 \
  --lateral-mode stress-free \
  --barostat-tau 0.1 \
  --barostat-gamma 1.0 \
  --barostat-compressibility-bar-inv 3.2e-7 \
  --barostat-pressure-tolerance-bar 25.0 \
  --max-lateral-box-ratio 2.0 \
  --gamma 2.0 \
  --traj-interval 1000 \
  --output-dir run_output/prod_w_tensile_W31250
```

`W31250.xyz` contains a cubic BCC W box with `31250 / 2 = 15625 = 25^3` cells and lattice parameter `3.2 A`, so `--box-length 80.0` is the correct value.

### W bulk relax workflow

Use this workflow before large-structure tensile if `summary.json` still reports large `initial_stress_*_abs_bar` after equilibration.

```bash
python run_scripts/w_bulk_relax.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --steps 5000 \
  --temperature 300 \
  --gamma 2.0 \
  --target-pressure-bar 0.0 \
  --barostat-tau 0.5 \
  --barostat-compressibility-bar-inv 3.2e-7 \
  --barostat-mu-max 0.005 \
  --traj-interval 500 \
  --output-dir run_output/w_bulk_relax_W31250
```

Outputs:

- `relaxation.csv`
- `summary.json`
- relaxed XYZ structure, for example `W_custom_relaxed.xyz`
- optional `trajectory.xyz`

Key fields in `summary.json`:

- `recommended_box_length_A`
- `recommended_lattice_param_A` when the script can infer a cubic BCC cell count
- `final_pressure_bar`
- `final_box_length_x/y/z`

The intended use is:

1. relax the bulk W structure close to zero pressure
2. take `recommended_box_length_A` and the relaxed XYZ
3. use those as the starting point for the next tensile run

### 7. W nanoindentation workflow

Minimal smoke test:

```bash
python run_scripts/w_indent.py --smoke
python cuda_test/test_w_indent_smoke.py
```

Example W `[100]` spherical-indenter run:

```bash
python run_scripts/w_indent.py \
  --orientation 100 \
  --replicas 6,6,4 \
  --steps 5000 \
  --equil-steps 1000 \
  --indenter-radius-A 8.0 \
  --indenter-stiffness 5.0 \
  --initial-depth-A 0.0 \
  --target-depth-A 2.0 \
  --gamma 2.0
```

Outputs are grouped by orientation, e.g. `run_output/w_indent/orientation_100/`, and include `load_depth.csv`, `summary.json`, `load_depth.png`, and the generated slab structure.

The updated indentation workflow supports:

- loading + unloading in a single run
- `phase=load/unload` in the CSV
- optional trajectory dumping through `--traj-interval`
- approximate Oliver-Pharr analysis in `summary.json`
  - `unload_initial_stiffness_nN_per_A`
  - `oliver_pharr_contact_depth_A`
  - `projected_contact_area_A2`
  - `hardness_GPa`
  - `reduced_modulus_GPa`

`hardness_GPa` and `reduced_modulus_GPa` are workflow-level estimates. They are useful for internal comparison across runs, but they are not yet a fully calibrated experimental nanoindentation pipeline.

Large custom-structure example:

```bash
python run_scripts/w_indent.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --steps 10000 \
  --equil-steps 1000 \
  --unload-steps 5000 \
  --indenter-radius-A 8.0 \
  --indenter-stiffness 5.0 \
  --initial-depth-A 0.0 \
  --target-depth-A 4.0 \
  --final-unload-depth-A 0.5 \
  --gamma 2.0 \
  --traj-interval 500 \
  --output-dir run_output/prod_w_indent_W31250
```

For `--orientation custom`, the current implementation assumes the imported XYZ belongs to an orthogonal cubic box.

Run all three orientations:

```bash
python run_scripts/w_indent.py --orientation 100 --replicas 6,6,4 --steps 5000 --equil-steps 1000 --indenter-radius-A 8.0 --indenter-stiffness 5.0 --initial-depth-A 0.0 --target-depth-A 2.0 --gamma 2.0
python run_scripts/w_indent.py --orientation 110 --replicas 6,6,4 --steps 5000 --equil-steps 1000 --indenter-radius-A 8.0 --indenter-stiffness 5.0 --initial-depth-A 0.0 --target-depth-A 2.0 --gamma 2.0
python run_scripts/w_indent.py --orientation 111 --replicas 5,5,3 --steps 5000 --equil-steps 1000 --indenter-radius-A 8.0 --indenter-stiffness 5.0 --initial-depth-A 0.0 --target-depth-A 2.0 --gamma 2.0
```

### 8. W crack workflow

Minimal smoke test:

```bash
python run_scripts/w_crack.py --smoke
python cuda_test/test_w_crack_smoke.py
```

Example W `[100]` crack-opening run:

```bash
python run_scripts/w_crack.py \
  --orientation 100 \
  --replicas 8,8,4 \
  --steps 5000 \
  --equil-steps 500 \
  --crack-half-length-A 8.0 \
  --crack-gap-A 1.2 \
  --target-strain 0.02 \
  --gamma 2.0
```

Outputs are grouped by orientation, e.g. `run_output/w_crack/orientation_100/`, and include `crack_response.csv`, `summary.json`, `crack_response.png`, and the generated cracked structure.

The crack workflow can now dump `trajectory.xyz` with `--traj-interval`.

Large custom-structure example:

```bash
python run_scripts/w_crack.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --steps 10000 \
  --equil-steps 1000 \
  --crack-half-length-A 8.0 \
  --crack-gap-A 1.2 \
  --target-strain 0.03 \
  --gamma 2.0 \
  --traj-interval 500 \
  --output-dir run_output/prod_w_crack_W31250
```

### 9. W DBTT scan

Minimal smoke test:

```bash
python cuda_test/test_w_dbtt_smoke.py
```

Example crack-based temperature scan:

```bash
python run_scripts/w_dbtt_scan.py \
  --orientation 100 \
  --temperatures 100,200,300,400,500,600
```

Outputs are written to `run_output/w_dbtt/` and include per-temperature crack runs plus:

- `dbtt_summary.csv`
- `dbtt_summary.json`
- `dbtt_summary.png`

The DBTT summary now emphasizes crack-response trends that are more interpretable than peak stress alone:

- `final_stress_bar`
- `stress_retention_ratio`
- `max_cmod_A`

For the current W crack-based DBTT workflow, treat these three fields as the primary interpretation axis. `peak_stress_magnitude_bar` is still reported, but it should not be used alone to claim the transition temperature.

Large custom-structure example:

```bash
python run_scripts/w_dbtt_scan.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --temperatures 100,200,300,400,500,600 \
  --steps 5000 \
  --equil-steps 500 \
  --gamma 2.0 \
  --output-dir run_output/prod_w_dbtt_W31250
```

### 10. Batch report and parameter guide

Run any subset of the four workflows with a unified output root:

```bash
python run_scripts/w_batch_report.py \
  --workflows tensile,indent,crack,dbtt \
  --orientations 100,110,111 \
  --output-dir run_output/w_batch_report
```

Detailed parameter meanings and report-field definitions are documented in:

- `W_WORKFLOWS_GUIDE.md`

Large custom-structure batch example:

```bash
python run_scripts/w_batch_report.py \
  --workflows tensile,indent,crack,dbtt \
  --orientations custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --output-dir run_output/w_batch_W31250
```

For large production runs, it is still recommended to launch each workflow separately instead of batching all four on one GPU.

---

## Ensemble reference

| Ensemble | `VerletIntegrator` kwargs | Extra |
|----------|--------------------------|-------|
| NVE | `ensemble='NVE'` | — |
| NVT | `ensemble='NVT', temperature=(T_init, T_target), gamma=γ` | Langevin |
| NPT | `ensemble='NPT', temperature=(T_init, T_target), gamma=γ` | + `BerendsenBarostat` or `AnisotropicNPTBarostat` passed to `BaseModel` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA build errors | Check `nvcc --version` matches PyTorch CUDA version; MSVC ≥ 2019 on Windows |
| `ImportError: simulon_cuda` | Rebuild: `python setup.py build_ext --inplace` |
| `KeyError '[0 0]'` in parameters | Parameter dict key must match `str(np.array([type_i, type_j]))`, e.g. `"[0 0]"` for a single element type |
| Wrong temperature at start | Use `temperature=(T_init, T_target)` — first value seeds Maxwell–Boltzmann velocities |

---

## Contributing

Issues and PRs are welcome. Please include a minimal reproducible example or small input structure when reporting bugs.
