import json
import sys
import argparse
import os

script_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)


from io_utils.reader import AtomFileReader
import torch
import time
from io_utils.output_logger import Logger
from core.force.template.pair_force_template import PairForce
from core.md_model import SumBackboneInterface
from core.integrator.integrator import VerletIntegrator
from core.md_model import BaseModel
from core.md_simulation import MDSimulator


def main():
    parser = argparse.ArgumentParser(description='confing.json paramed')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to JSON configuration file')
    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"the file {args.config} is not exist")
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path_xyz = config['data_path_xyz']
    box_length = config['box_length']
    parameters_pair = config['pair_parameter']
    potential_formula = config['potential_formula']
    cut_off = config['cut_off']
    dt = config['dt']
    temperature = config['temperature']
    gamma = config['gamma']
    num_steps = config['num_steps']
    print_interval = config['print_interval']
    output_save_path = config['output_save_path']

    sys.stdout = Logger(sys.stdout, log_dir=output_save_path)
    sys.stderr = Logger(sys.stderr, log_dir=output_save_path)

    atom_file_reader = AtomFileReader(data_path_xyz,
                                      box_length=box_length,
                                      cutoff= cut_off,
                                      device=device,
                                      parameter=parameters_pair,
                                      skin_thickness=3.0)

    force_filed = PairForce(atom_file_reader, ['k'], potential_formula)
    bone_force_filed = SumBackboneInterface([force_filed], atom_file_reader)
    vi = VerletIntegrator(atom_file_reader,
                          dt,
                          force_filed,
                          'NVT',
                          temperature,
                          gamma)
    MDModel = BaseModel(bone_force_filed, vi, atom_file_reader)
    Simulation = MDSimulator(MDModel, num_steps, print_interval,
                             save_to_graph_dataset=False)
    Simulation.run()
    now_time = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    energy_path = os.path.join(output_save_path, f"MD_energy_curve_{now_time}.png")
    Simulation.save_energy_curve(energy_path)
    traj_path = os.path.join(output_save_path, f"MD_traj_{now_time}.xyz")
    Simulation.save_xyz_trajectory(traj_path,
                                   atom_types=atom_file_reader.atom_types)
    force_path = os.path.join(output_save_path, f"forces_{now_time}.xyz")
    Simulation.save_forces_grad(force_path,
                                with_no_ele=True,
                                atom_types=atom_file_reader.atom_types)


if __name__ == "__main__":
    main()
