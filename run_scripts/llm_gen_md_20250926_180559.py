#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Liquid Argon NVT simulation with Lennard-Jones potential
import torch, os
from io_utils.reader import AtomFileReader
from core.force.lennard_jones_force_cu import LennardJonesForce
from core.md_model import SumBackboneInterface, BaseModel
from core.integrator.integrator import VerletIntegrator
from core.md_simulation import MDSimulator

DEFAULT_XYZ = 'run_data/Ar10000.xyz'
DEFAULT_EPS = 0.0104  # Argon
DEFAULT_SIG = 3.405

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

xyz_path = input(f'Structure file path (default {DEFAULT_XYZ}): ').strip() or DEFAULT_XYZ
box_input = input('Box length (Å) (leave blank to use 1550.0): ').strip()
box_length = float(box_input) if box_input else 1550.0
cutoff_input = input('Cutoff (Å) (default 7.0): ').strip()
cutoff = float(cutoff_input) if cutoff_input else 7.0

# Force field params - auto-detect Argon
atom_types = detect_atom_types(xyz_path)
print(f'Detected atom types: {sorted(atom_types) if atom_types else "(unknown)"}')
if atom_types and atom_types == {'Ar'}:
    default_eps, default_sig = DEFAULT_EPS, DEFAULT_SIG
    print(f"Using Argon LJ parameters: epsilon={default_eps} eV, sigma={default_sig} Å")
else:
    default_eps = DEFAULT_EPS
    default_sig = DEFAULT_SIG

eps_in = input(f'LJ epsilon (eV) default {default_eps}: ').strip()
sig_in = input(f'LJ sigma (Å) default {default_sig}: ').strip()
epsilon = float(eps_in) if eps_in else default_eps
sigma = float(sig_in) if sig_in else default_sig
parameters_pair = {"[0 0]": {"epsilon": epsilon, "sigma": sigma}}

# 1000 ps simulation with 1 fs timestep = 1,000,000 steps
steps_in = input('Number of MD steps (default 1000000 for 1000 ps): ').strip()
num_steps = int(steps_in) if steps_in else 1000000

# Temperature set to 94.4 K for liquid Argon
temp_in = input('Temperature (K or K1,K2; default 94.4): ').strip()
if temp_in:
    if ',' in temp_in:
        t1,t2 = [float(x) for x in temp_in.split(',')[:2]]
    else:
        t1 = float(temp_in); t2 = t1
else:
    t1 = t2 = 94.4

min_in = input('Energy minimization before run? (y/N): ').strip().lower()
need_min = (min_in == 'y')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
molecular = AtomFileReader(filename=xyz_path, box_length=box_length, cutoff=cutoff, device=device, parameter=parameters_pair, skin_thickness=3.0)
force_field = LennardJonesForce(molecular)
bone = SumBackboneInterface([force_field], molecular)
vi = VerletIntegrator(molecular=molecular, dt=0.001, force_field=force_field, ensemble='NVT', temperature=[t1,t2], gamma=1000.0)
model = BaseModel(bone, vi, molecular)
sim = MDSimulator(model, num_steps=num_steps, print_interval=max(1, num_steps//100), save_to_graph_dataset=False)

print(f"\nSimulation Summary:")
print(f"Structure: {xyz_path}")
print(f"Box length: {box_length} Å")
print(f"Cutoff: {cutoff} Å")
print(f"Temperature: {t1} K")
print(f"Steps: {num_steps} ({(num_steps * 0.001):.1f} ps)")
print(f"LJ parameters: epsilon={epsilon} eV, sigma={sigma} Å")
print(f"Minimization: {'Yes' if need_min else 'No'}")
print(f"Device: {device}")

sim.run(enable_minimize_energy=need_min)
sim.summarize_profile()
print('LJ simulation finished.')
