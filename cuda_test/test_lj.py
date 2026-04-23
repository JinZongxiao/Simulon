import torch
from io_utils.reader import AtomFileReader
from core.force.lennard_jones_force_cu import LennardJonesForce
# from core.force.lennard_jones_force import LennardJonesForce
from core.md_model import SumBackboneInterface
from core.integrator.integrator import VerletIntegrator
from core.md_model import BaseModel
from core.md_simulation import MDSimulator
from core.analyser import RDFAccumulator

import sys
from io_utils.output_logger import Logger

# sys.stdout = Logger(sys.stdout, log_dir="/public/home/normal_bgd/J1N/Simulon/run_data/logs")

NA = 6.02214076e23
AR_MOLAR_MASS = 39.948
def box_length_for_density(n_atoms: int, rho_gcm3: float, molar_mass: float = AR_MOLAR_MASS) -> float:
    mass_g = n_atoms * molar_mass / NA
    volume_cm3 = mass_g / rho_gcm3
    volume_A3 = volume_cm3 * 1e24
    return volume_A3 ** (1.0/3.0)

# 0.0018 气态
# 1.374 液态

# xyz_path = "/public/home/normal_bgd/J1N/Simulon/run_data/Ar100000.xyz"
xyz_path = "C:/Users/Administrator/Desktop/phd_workspace/code_repo/Simulon/run_data/Ar10000.xyz"


n_atoms = 10000 # _read_xyz_count(xyz_path)
box_length = box_length_for_density(n_atoms, 1.374)
print(f"Box length for density 1.374 g/cm^3: {box_length:.3f} Å")

ok = True
# ok = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parameters_pair = {
    "[0 0]": {
      "epsilon": 0.0104,
      "sigma"  : 3.405
    }
}
atom_file_reader = AtomFileReader(filename=xyz_path,
                                  # "/public/home/normal_bgd/J1N/Simulon/run_data/Ar.xyz"
                                  # ../../run_data/Ar1000.xyz
                                  # C:/Users/Administrator/Desktop/phd_workspace/code_repo/Simulon/run_data/Ar1000.xyz
                                  box_length=box_length, # 155 335 720 1550
                                  cutoff= 7,
                                  device=device,
                                  parameter=parameters_pair,
                                  skin_thickness=4.5
                                  ,is_switch=True, switch_ratio=0.9
                                  )
lj_force = LennardJonesForce(atom_file_reader)
# initial_result = lj_force()
# print(f"初始势能: {initial_result['energy'].item():.2f} eV")
# print(f"每原子势能: {initial_result['energy'].item()/10000:.4f} eV/atom")
spread_mode = 'scale'  # none|scale|random


if ok:
  bone_force_filed = SumBackboneInterface([lj_force], atom_file_reader)
  vi = VerletIntegrator(molecular=atom_file_reader,
                        dt=0.0005,  # 1 fs
                        force_field=lj_force,
                        ensemble='NVT',
                        temperature=[94.4,94.4],
                        gamma=5
                        )
  rdf = RDFAccumulator( molecular=atom_file_reader, 
                       nbins=200, cutoff=7.0, nevery=1000, 
                       nrepeat=50, outfile='rdf_11.out', 
                       type_pair=None )
  MDModel = BaseModel(bone_force_filed, vi, atom_file_reader)
  Simulation = MDSimulator(MDModel, num_steps=10000, print_interval=1,save_to_graph_dataset=False,spread_mode=spread_mode,
                           output_dir= r'C:\Users\Administrator\Desktop\phd_workspace\code_repo\Simulon\output\Ar_LJ',
                           dump_interval=1,write_forces=False, write_energies=True,
                           traj_interval=100,
                           forces_interval= 100,
                           energies_interval= 1,
                           rdf_accumulator=rdf 
                           )
  
  Simulation.run(enable_minimize_energy=True)  # 先进行能量最小化
  Simulation.summarize_profile()

