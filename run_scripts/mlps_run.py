import json
import os
import argparse
import sys

script_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)


from io_utils.reader import AtomFileReader
from io_utils.output_logger import Logger
import torch
import time
from machine_learning_potentials.machine_learning_force import MachineLearningForce
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
    cut_off = config['cut_off']
    dt = config['dt']
    temperature = config['temperature']
    gamma = config['gamma']
    num_steps = config['num_steps']
    print_interval = config['print_interval']
    output_save_path = config['output_save_path']
    aimd_pos_file = config['aimd_pos_file']
    aimd_force_file = config['aimd_force_file']
    mlps_finetune_params = config['mlps_finetune_params']

    sys.stdout = Logger(sys.stdout, log_dir=output_save_path)
    sys.stderr = Logger(sys.stderr, log_dir=output_save_path)
    
    atom_file_reader = AtomFileReader(data_path_xyz,
                                      box_length=box_length,
                                      cutoff= cut_off,
                                      device=device,
                                      is_mlp=True)
    machine_learning_force = MachineLearningForce(
        molecular=atom_file_reader,
        aimd_pos_file=aimd_pos_file,
        aimd_force_file=aimd_force_file,
        mlp_model_name='chgnet',
        mlps_finetune_params = mlps_finetune_params,
        mlps_model_path= None
    )
    ForceField = SumBackboneInterface([machine_learning_force], atom_file_reader)
    vi = VerletIntegrator(atom_file_reader, 
                          dt, 
                          ForceField, 
                          'NVT',
                          temperature, 
                          gamma)
    MDModel = BaseModel(ForceField, vi, atom_file_reader)
    Simulation = MDSimulator(MDModel, num_steps, print_interval)
    Simulation.run(enable_minimize_energy=False)
    now_time = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    energy_path = os.path.join(output_save_path, f"MD_energy_curve_{now_time}.png")
    Simulation.save_energy_curve(energy_path)
    traj_path = os.path.join(output_save_path, f"MD_traj_{now_time}.xyz")
    Simulation.save_xyz_trajectory(traj_path,
                                   atom_types=atom_file_reader.atom_types)
    force_path = os.path.join(output_save_path, f"forces_{now_time}.xyz")
    Simulation.save_forces_grad(force_path,
                                with_no_ele= True,
                                atom_types=atom_file_reader.atom_types)


if __name__ == "__main__":
    main()