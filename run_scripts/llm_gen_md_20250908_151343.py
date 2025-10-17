#!/usr/bin/env python
# -*- coding: utf-8 -*-
# EAM 模拟脚本 for Tungsten (W) with interactive inputs
import torch
from io_utils.reader import AtomFileReader
from io_utils.eam_parser import EAMParser
from core.force.eam_force_cu import EAMForceCUDAExt as EAMForce
from core.md_model import SumBackboneInterface, BaseModel
from core.integrator.integrator import VerletIntegrator
from core.md_simulation import MDSimulator

DEFAULT_POT = 'run_data/WRe_YC2.eam.fs'
DEFAULT_XYZ = 'run_data/W_bcc_stable.xyz'

eam_path = input(f'EAM 势文件路径 (默认 {DEFAULT_POT}): ').strip() or DEFAULT_POT
xyz_path = input(f'结构文件路径 (默认 {DEFAULT_XYZ}): ').strip() or DEFAULT_XYZ
box_in = input('盒长(Å) (默认 9.6): ').strip()
box_length = float(box_in) if box_in else 9.6
cut_in = input('截断距离(Å) (默认 6.0): ').strip()
cutoff = float(cut_in) if cut_in else 6.0
steps_in = input('模拟步数 (默认 2000): ').strip()
num_steps = int(steps_in) if steps_in else 2000
temp_in = input('温度(K 或 K1,K2; 默认 300): ').strip()
if temp_in:
    if ',' in temp_in:
        t1, t2 = [float(x) for x in temp_in.split(',')[:2]]
    else:
        t1 = t2 = float(temp_in)
else:
    t1 = t2 = 300.0
min_in = input('是否先最小化? (y/N): ').strip().lower()
need_min = (min_in == 'y')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = EAMParser(filepath=eam_path, device=device)
molecular = AtomFileReader(filename=xyz_path, box_length=box_length, cutoff=cutoff, device=device, is_fs=True, skin_thickness=3.0)
force_field = EAMForce(eam_parser=parser, molecular=molecular)
bone = SumBackboneInterface([force_field], molecular)
vi = VerletIntegrator(molecular=molecular, dt=0.001, force_field=force_field, ensemble='NVT', temperature=[t1, t2], gamma=1000.0)
model = BaseModel(bone, vi, molecular)
sim = MDSimulator(model, num_steps=num_steps, print_interval=max(1, num_steps//100), save_to_graph_dataset=False)
sim.run(enable_minimize_energy=need_min)
sim.summarize_profile()
print('完成 EAM 模拟。')
