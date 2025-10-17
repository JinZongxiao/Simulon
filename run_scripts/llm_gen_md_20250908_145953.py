#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch, os
from io_utils.reader import AtomFileReader
from core.force.lennard_jones_force_cu import LennardJonesForce
from core.md_model import SumBackboneInterface, BaseModel
from core.integrator.integrator import VerletIntegrator
from core.md_simulation import MDSimulator

DEFAULT_XYZ = 'run_data/Ar100.xyz'
DEFAULT_EPS = 0.0104  # Argon
DEFAULT_SIG = 3.405
DEFAULT_BOX = 20.0  # Approximate for Ar100
DEFAULT_CUTOFF = 7.0
DEFAULT_STEPS = 5000
DEFAULT_TEMP = 94.4
DEFAULT_MIN = 'y'

def detect_atom_types(path: str):
    types = set()
    try:
        with open(path, 'r') as f:
            first = f.readline().strip()
            try: n = int(first)
            except: return types
            _ = f.readline()
            for _ in range(n):
                line = f.readline()
                if not line: break
                sp = line.split()
                if sp: types.add(sp[0])
    except Exception:
        pass
    return types

xyz_path = input(f'结构文件路径 (默认 {DEFAULT_XYZ}): ').strip() or DEFAULT_XYZ
box_input = input(f'盒长(Å) (默认 {DEFAULT_BOX}): ').strip()
box_length = float(box_input) if box_input else DEFAULT_BOX
cutoff_input = input(f'截断距离(Å) (默认 {DEFAULT_CUTOFF}): ').strip()
cutoff = float(cutoff_input) if cutoff_input else DEFAULT_CUTOFF
atom_types = detect_atom_types(xyz_path)
print(f'Detected atom types: {sorted(atom_types) if atom_types else "(未识别)"}')
if atom_types and atom_types == {'Ar'}:
    default_eps, default_sig = DEFAULT_EPS, DEFAULT_SIG
else:
    default_eps = DEFAULT_EPS
    default_sig = DEFAULT_SIG

eps_in = input(f'LJ epsilon (eV) 默认 {default_eps}: ').strip()
sig_in = input(f'LJ sigma (Å) 默认 {default_sig}: ').strip()
epsilon = float(eps_in) if eps_in else default_eps
sigma = float(sig_in) if sig_in else default_sig
parameters_pair = {"[0 0]": {"epsilon": epsilon, "sigma": sigma}}

steps_in = input(f'模拟步数 (默认 {DEFAULT_STEPS}): ').strip()
num_steps = int(steps_in) if steps_in else DEFAULT_STEPS
temp_in = input(f'温度(K 或 K1,K2; 默认 {DEFAULT_TEMP}): ').strip()
if temp_in:
    if ',' in temp_in:
        t1,t2 = [float(x) for x in temp_in.split(',')[:2]]
    else:
        t1 = float(temp_in); t2 = t1
else:
    t1 = t2 = DEFAULT_TEMP
min_in = input(f'是否先最小化? (y/N, 默认 {DEFAULT_MIN}): ').strip().lower() or DEFAULT_MIN
need_min = (min_in == 'y')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
molecular = AtomFileReader(filename=xyz_path, box_length=box_length, cutoff=cutoff, device=device, parameter=parameters_pair, skin_thickness=3.0)
force_field = LennardJonesForce(molecular)
bone = SumBackboneInterface([force_field], molecular)
vi = VerletIntegrator(molecular=molecular, dt=0.001, force_field=force_field, ensemble='NVT', temperature=[t1,t2], gamma=1000.0)
model = BaseModel(bone, vi, molecular)
sim = MDSimulator(model, num_steps=num_steps, print_interval=max(1, num_steps//100), save_to_graph_dataset=False)
sim.run(enable_minimize_energy=need_min)
sim.summarize_profile()
print('完成 LJ 模拟。')
