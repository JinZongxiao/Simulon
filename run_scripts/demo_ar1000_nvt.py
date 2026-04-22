"""
demo_ar1000_nvt.py — Argon NVT 中等规模演示
============================================
系统 : 1000 个 Ar 原子（来自 run_data/Ar/Ar1000.xyz）
力场 : Lennard-Jones  epsilon=0.0104 eV  sigma=3.405 Ang  rc=8.5 Ang
系综 : NVT-Langevin   T=90 K   gamma=0.01 ps^-1
步长 : dt=0.001 ps (1 fs)   步数 : 500 步
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from io_utils.reader                import AtomFileReader
from core.force.lennard_jones_force import LennardJonesForce
from core.md_model                  import SumBackboneInterface, BaseModel
from core.integrator.integrator     import VerletIntegrator
from core.md_simulation             import MDSimulator

# ── 路径 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
XYZ_PATH     = os.path.join(PROJECT_ROOT, 'run_data', 'Ar', 'Ar1000.xyz')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'run_output', 'demo_ar1000_nvt')

# ── 参数 ─────────────────────────────────────────────────────────────────────
# 1000 个 Ar 原子，与 100-atom 体系同密度 → box = 16.901 * 10^(1/3) = 36.40 Ang
BOX_LENGTH  = 36.40    # Ang
CUTOFF      = 8.5      # Ang
SKIN        = 2.0      # Ang
DT          = 0.001    # ps
TEMPERATURE = (90, 90) # (T_init, T_target)  K
GAMMA       = 0.01     # ps^-1
NUM_STEPS   = 500
PRINT_EVERY = 50

PARAMS = {"[0 0]": {"epsilon": 0.0104, "sigma": 3.405}}

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
print(f"Box     : {BOX_LENGTH} Ang")

# Ar1000.xyz 坐标最大 ~62 Ang，超出 box_length=36.4，必须先 wrap 进盒再重建邻居表
with torch.no_grad():
    mol.coordinates.fmod_(BOX_LENGTH)          # 原地 wrap 到 [0, BOX)
    mol.update_coordinates(mol.coordinates)    # 重建邻居表

print(f"Edges   : {mol.graph_data.edge_index.shape[1]} (initial Verlet list, after wrap)")

# ── 力场 & 积分器 ────────────────────────────────────────────────────────────
ff    = LennardJonesForce(mol)
sb    = SumBackboneInterface([ff], mol)
integ = VerletIntegrator(mol, dt=DT, ensemble='NVT',
                         temperature=TEMPERATURE, gamma=GAMMA)
model = BaseModel(sb, integ, mol)

# ── MDSimulator ───────────────────────────────────────────────────────────────
sim = MDSimulator(
    model,
    num_steps         = NUM_STEPS,
    print_interval    = PRINT_EVERY,
    output_dir        = OUTPUT_DIR,
    traj_interval     = PRINT_EVERY,
    write_energies    = True,
    energies_interval = 10,
)

print(f"\n{'='*60}")
print(f"  Ar 1000-atom NVT @ {TEMPERATURE[1]} K   dt={DT} ps   {NUM_STEPS} steps")
print(f"{'='*60}\n")

result = sim.run(enable_minimize_energy=False)

print(f"\nOutput  : {OUTPUT_DIR}")

print(f"\n{'='*60}")
sim.summarize_profile()
print(f"{'='*60}")
