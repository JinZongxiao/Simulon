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
| **Performance** | ~384 steps/s on RTX 3050 for 100-atom Ar NVT |

---

## Key capabilities

- **PyTorch-first**: all state lives in tensors; CPU/GPU switching is a single flag.
- **Three ensembles**: NVE, NVT (Langevin thermostat), NPT (Berendsen barostat + Langevin).
- **Triclinic PBC**: unified `Box` class handles cubic, orthorhombic, and fully triclinic cells via a 3×3 lattice matrix.
- **Verlet neighbor list**: lazy rebuild triggered by displacement threshold (skin/2); GPU-accelerated via optional CUDA extension.
- **Modular force fields**: Lennard-Jones, EAM, Born–Mayer–Huggins, and a user-defined pair-potential template.
- **Restart**: full checkpoint/resume support — save every N steps, resume without re-equilibrating.
- **RDF analyzer**: online accumulation with correct normalization for same-species and cross-species pairs.
- **I/O & utilities**: XYZ reader/writer, CSV energy logger, trajectory dump, EAM table parser, pymatgen/ASE integration.
- **ML potentials**: example integration for CHGNet-like models.

---

## Repository layout

```
core/
  box.py                  # Unified orthogonal + triclinic PBC (H-matrix)
  barostat.py             # Berendsen isotropic NPT barostat
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
  restart.py              # save_checkpoint / load_checkpoint
  writer.py / output_logger.py / eam_parser.py / ...

cuda source/
  neighbor_search_kernel.cu
  lj_energy_force*.cu
  eam_cuda_ext*.cu

run_scripts/
  demo_ar_nvt.py          # Quick demo: 100-atom Ar NVT
  lj_run.py               # JSON-driven LJ simulation
  user_defined_run.py
  mlps_run.py
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

---

## Ensemble reference

| Ensemble | `VerletIntegrator` kwargs | Extra |
|----------|--------------------------|-------|
| NVE | `ensemble='NVE'` | — |
| NVT | `ensemble='NVT', temperature=(T_init, T_target), gamma=γ` | Langevin |
| NPT | `ensemble='NPT', temperature=(T_init, T_target), gamma=γ` | + `BerendsenBarostat` passed to `BaseModel` |

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
