import json
import sys
import argparse
import os

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
# Resolve relative data paths against project root
os.chdir(project_root)

import torch
from io_utils.reader import AtomFileReader
from core.force.lennard_jones_force import LennardJonesForce
from core.md_model import SumBackboneInterface, BaseModel
from core.integrator.integrator import VerletIntegrator
from core.md_simulation import MDSimulator


def main():
    parser = argparse.ArgumentParser(description='JSON-driven LJ simulation')
    parser.add_argument('--config', type=str, default='run_scripts/lj_run.json',
                        help='Path to JSON configuration file')
    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path_xyz    = config['data_path_xyz']
    box_length       = config['box_length']
    parameters_pair  = config['pair_parameter']
    cut_off          = config['cut_off']
    dt               = config['dt']            # unit: ps  (0.001 = 1 fs)
    temperature      = config['temperature']   # [T_init, T_target] in K
    gamma            = config['gamma']         # Langevin damping, unit: 1/ps
    num_steps        = config['num_steps']
    print_interval   = config['print_interval']
    output_save_path = config['output_save_path']

    os.makedirs(output_save_path, exist_ok=True)

    mol = AtomFileReader(
        data_path_xyz,
        box_length     = box_length,
        cutoff         = cut_off,
        device         = device,
        parameter      = parameters_pair,
        skin_thickness = 2.0,
    )
    print(f"Loaded: {mol.atom_count} atoms  box={box_length} A  device={device}")

    ff   = LennardJonesForce(mol)
    sb   = SumBackboneInterface([ff], mol)

    # NOTE: VerletIntegrator keyword-argument form — do NOT use positional after dt
    vi = VerletIntegrator(
        mol,
        dt          = dt,
        ensemble    = 'NVT',
        temperature = temperature,
        gamma       = gamma,
    )

    model = BaseModel(sb, vi, mol)

    sim = MDSimulator(
        model,
        num_steps      = num_steps,
        print_interval = print_interval,
        output_dir     = output_save_path,
        write_energies = True,
        energies_interval = print_interval,
        traj_interval  = print_interval,
    )

    sim.run(enable_minimize_energy=True)

    # Save energy curve (matplotlib PNG)
    import time
    now_time    = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    energy_path = os.path.join(output_save_path, f"MD_energy_curve_{now_time}.png")
    sim.save_energy_curve(energy_path)
    print(f"Energy curve saved to: {energy_path}")
    print(f"Trajectory / energies written to: {output_save_path}/")


if __name__ == "__main__":
    main()
