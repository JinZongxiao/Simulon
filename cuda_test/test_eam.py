from io_utils.reader import AtomFileReader
from io_utils.eam_parser import EAMParser
# from core.force.eam_force import EAMForce
from core.force.eam_force_cu import EAMForceCUDAExt  as EAMForce
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print("正在解析EAM势文件...")
# eamfs = 'C:/Users/Administrator/Desktop/phd_workspace/code_repo/Simulon/run_data/WRe_YC2.eam.fs'
eamfs = '/public/home/normal_bgd/J1N/Simulon/run_data/WRe_YC2.eam.fs'
eam_parser = EAMParser(filepath=eamfs, device=device)
# print(f"解析成功！元素: {eam_parser.elements}")
# print(f"截断距离: {eam_parser.cutoff}")

xyz_path = "/public/home/normal_bgd/J1N/Simulon/run_data/W31250.xyz"
# xyz_path = 'C:/Users/Administrator/Desktop/phd_workspace/code_repo/Simulon/run_data/W_bcc_stable.xyz'

# print("正在读取原子结构文件...")
molecular = AtomFileReader( filename=xyz_path,
                            box_length=80,# 9.6,
                            cutoff=6.0,
                            device=device,
                            is_fs = True)
# print(f"读取成功！原子数: {len(molecular.atom_types)}")

#print("正在初始化EAM力场...")
eam_force_field = EAMForce(eam_parser=eam_parser, molecular=molecular)

# print("正在测试力场计算...")
result = eam_force_field()
# print(f"总能量: {result['energy'].item():.6f} eV")
# print(f"力的范围: [{result['forces'].min().item():.6f}, {result['forces'].max().item():.6f}] eV/Å")
# print("EAM力场测试成功！")

# 如果需要运行完整模拟，可以取消下面的注释
from core.md_model import SumBackboneInterface
from core.integrator.integrator import VerletIntegrator
from core.md_model import BaseModel
from core.md_simulation import MDSimulator

bone_force_filed = SumBackboneInterface([eam_force_field], molecular)
integrator = VerletIntegrator(
    molecular=molecular,
    force_field=eam_force_field,
    dt=0.001, # 1 fs
    ensemble='NVT',
    temperature=[800, 800], # K
    gamma=1/0.001 # ps^-1 
)

MDModel = BaseModel(bone_force_filed, integrator, molecular)
Simulation = MDSimulator(MDModel, num_steps=100000, print_interval=1,save_to_graph_dataset=False,
                         output_dir="/public/home/normal_bgd/J1N/Simulon/output",
                         dump_interval=1, write_forces=False, write_energies=True)
Simulation.run(enable_minimize_energy=False)
Simulation.summarize_profile()