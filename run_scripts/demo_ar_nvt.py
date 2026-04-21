"""
demo_ar_nvt.py — Argon NVT 演示
================================
系统 : 100 个 Ar 原子，FCC 结构（来自 run_data/Ar/Ar.xyz）
力场 : Lennard-Jones  ε=0.0104 eV  σ=3.4 Å  rc=8.5 Å
系综 : NVT-Langevin   T=90 K   γ=0.01 ps⁻¹
步长 : dt=0.001 ps (1 fs)   步数 : 500 步
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from io_utils.reader       import AtomFileReader
from core.force.lennard_jones_force import LennardJonesForce
from core.md_model         import SumBackboneInterface, BaseModel
from core.integrator.integrator import VerletIntegrator
from core.md_simulation    import MDSimulator

# ── 路径 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
XYZ_PATH     = os.path.join(PROJECT_ROOT, 'run_data', 'Ar', 'Ar.xyz')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'run_output', 'demo_ar_nvt')

# ── 参数 ─────────────────────────────────────────────────────────────────────
BOX_LENGTH  = 16.901   # Å（对应 100 Ar）
CUTOFF      = 8.5      # Å
SKIN        = 2.0      # Å（Verlet 皮肤）
DT          = 0.001    # ps
TEMPERATURE = (90, 90) # (T_init, T_target)  K
GAMMA       = 0.01     # ps⁻¹
NUM_STEPS   = 500
PRINT_EVERY = 50

# LJ 参数：Ar-Ar，键 "[0 0]" 为 element_id 字符串
PARAMS = {"[0 0]": {"epsilon": 0.0104, "sigma": 3.4}}

# ── 设备 ─────────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device : {device.upper()}")
if device == 'cuda':
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ── 读取结构 ─────────────────────────────────────────────────────────────────
print(f"\nLoading : {XYZ_PATH}")
mol = AtomFileReader(
    filename       = XYZ_PATH,
    box_length     = BOX_LENGTH,
    cutoff         = CUTOFF,
    device         = device,
    parameter      = PARAMS,
    skin_thickness = SKIN,
)
print(f"Atoms   : {mol.atom_count}")
print(f"Box     : {BOX_LENGTH} Å³  ({mol.box})")
print(f"Edges   : {mol.graph_data.edge_index.shape[1]} (initial Verlet list)")

# ── 力场 & 积分器 ────────────────────────────────────────────────────────────
ff    = LennardJonesForce(mol)
sb    = SumBackboneInterface([ff], mol)
integ = VerletIntegrator(mol, dt=DT, ensemble='NVT',
                         temperature=TEMPERATURE, gamma=GAMMA)
model = BaseModel(sb, integ, mol)

# ── MDSimulator ───────────────────────────────────────────────────────────────
sim = MDSimulator(
    model,
    num_steps      = NUM_STEPS,
    print_interval = PRINT_EVERY,
    output_dir     = OUTPUT_DIR,
    traj_interval  = PRINT_EVERY,    # 每 50 步写一帧轨迹
    write_energies = True,
    energies_interval = 10,          # 每 10 步写一次能量
)

print(f"\n{'='*60}")
print(f"  Ar NVT @ {TEMPERATURE[1]} K   dt={DT} ps   {NUM_STEPS} steps")
print(f"{'='*60}\n")

result = sim.run(enable_minimize_energy=False)

print(f"\nOutput  : {OUTPUT_DIR}")

# ── 性能简报 ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
sim.summarize_profile()
print(f"{'='*60}")
