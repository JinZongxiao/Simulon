"""
demo_w250_nve.py — EAM/W NVE 中等规模演示
==========================================
系统 : 250 个 W 原子，BCC 结构（来自 run_data/W/W250.xyz）
力场 : EAM/fs  WRe_YC2.eam.fs（W-only）
系综 : NVE
步长 : dt=0.001 ps (1 fs)   步数 : 500 步
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from io_utils.reader            import AtomFileReader
from io_utils.eam_parser        import EAMParser
from core.force.eam_force       import EAMForce
from core.md_model              import SumBackboneInterface, BaseModel
from core.integrator.integrator import VerletIntegrator
from core.md_simulation         import MDSimulator

# ── 路径 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
XYZ_PATH     = os.path.join(PROJECT_ROOT, 'run_data', 'W', 'W250.xyz')
EAM_PATH     = os.path.join(PROJECT_ROOT, 'run_data', 'W', 'WRe_YC2.eam.fs')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'run_output', 'demo_w250_nve')

# ── 参数 ─────────────────────────────────────────────────────────────────────
# 250 个 BCC-W 原子，5x5x5 超胞，lattice=3.2 Ang → box=16.0 Ang
BOX_LENGTH  = 16.0     # Ang
SKIN        = 1.0      # Ang
DT          = 0.001    # ps
NUM_STEPS   = 500
PRINT_EVERY = 50

# ── 设备 ─────────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device : {device.upper()}")
if device == 'cuda':
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ── 读取 EAM 势函数 ───────────────────────────────────────────────────────────
print(f"\nLoading EAM : {EAM_PATH}")
eam_parser = EAMParser(EAM_PATH, torch.device(device))
CUTOFF = eam_parser.cutoff
print(f"EAM cutoff  : {CUTOFF:.4f} Ang   elements: {eam_parser.elements}")

# ── 读取结构 ─────────────────────────────────────────────────────────────────
print(f"Loading xyz : {XYZ_PATH}")
mol = AtomFileReader(
    filename       = XYZ_PATH,
    box_length     = BOX_LENGTH,
    cutoff         = CUTOFF,
    device         = device,
    skin_thickness = SKIN,
    is_mlp         = True,
)
print(f"Atoms   : {mol.atom_count}")
print(f"Box     : {BOX_LENGTH} Ang")
print(f"Edges   : {mol.graph_data.edge_index.shape[1]} (initial Verlet list)")

# ── 力场 & 积分器 ────────────────────────────────────────────────────────────
ff    = EAMForce(eam_parser, mol, use_tables=True)
sb    = SumBackboneInterface([ff], mol)
integ = VerletIntegrator(mol, dt=DT, ensemble='NVE')
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
print(f"  W 250-atom NVE   dt={DT} ps   {NUM_STEPS} steps")
print(f"{'='*60}\n")

result = sim.run(enable_minimize_energy=False)

print(f"\nOutput  : {OUTPUT_DIR}")

print(f"\n{'='*60}")
sim.summarize_profile()
print(f"{'='*60}")
