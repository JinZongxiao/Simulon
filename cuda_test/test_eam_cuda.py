from io_utils.reader import AtomFileReader
from io_utils.eam_parser import EAMParser
from core.force.eam_force import EAMForce
from core.force.eam_force_cu import EAMForceCUDA
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=== EAM Force 性能对比测试 ===")

# 解析EAM势
eam_parser = EAMParser(filepath='C:/Users/Administrator/Desktop/phd_workspace/code_repo/Simulon/run_data/WRe_YC2.eam.fs', device=device)

# 读取结构
molecular = AtomFileReader(
    filename='C:/Users/Administrator/Desktop/phd_workspace/code_repo/Simulon/run_data/W250.xyz',
    box_length=9.6,
    cutoff=6.0,
    device=device,
    is_fs=True
)

print(f"测试系统: {len(molecular.atom_types)} 个原子")
print(f"设备: {device}")

# 创建两个版本的力场
print("\n=== 初始化力场 ===")
eam_force_standard = EAMForce(eam_parser=eam_parser, molecular=molecular)
eam_force_cuda = EAMForceCUDA(eam_parser=eam_parser, molecular=molecular)

# 测试标准版本
print("\n=== 测试标准版本 ===")
torch.cuda.synchronize() if device.type == 'cuda' else None
start_time = time.time()

result_standard = eam_force_standard()
torch.cuda.synchronize() if device.type == 'cuda' else None
standard_time = time.time() - start_time

print(f"标准版本耗时: {standard_time:.4f} 秒")
print(f"总能量: {result_standard['energy'].item():.2f} eV")
print(f"每原子能量: {result_standard['energy'].item()/len(molecular.atom_types):.2f} eV/atom")

# 测试CUDA版本
print("\n=== 测试CUDA优化版本 ===")
torch.cuda.synchronize() if device.type == 'cuda' else None
start_time = time.time()

result_cuda = eam_force_cuda()
torch.cuda.synchronize() if device.type == 'cuda' else None
cuda_time = time.time() - start_time

print(f"CUDA版本耗时: {cuda_time:.4f} 秒")
print(f"总能量: {result_cuda['energy'].item():.2f} eV")
print(f"每原子能量: {result_cuda['energy'].item()/len(molecular.atom_types):.2f} eV/atom")

# 对比结果
print(f"\n=== 结果对比 ===")
energy_diff = abs(result_standard['energy'].item() - result_cuda['energy'].item())
print(f"能量差异: {energy_diff:.6f} eV")

if cuda_time > 0:
    speedup = standard_time / cuda_time
    print(f"加速比: {speedup:.2f}x")

# 检查力的差异
force_diff = torch.norm(result_standard['forces'] - result_cuda['forces']).item()
print(f"力的差异(L2范数): {force_diff:.6f} eV/Å")

if energy_diff < 1e-3 and force_diff < 1e-2:
    print("✓ CUDA版本验证通过！")
else:
    print("⚠️ CUDA版本结果有差异，需要检查")

# 多次运行测试性能
print(f"\n=== 性能基准测试 (10次运行) ===")
n_runs = 10

# 标准版本基准
torch.cuda.synchronize() if device.type == 'cuda' else None
start_time = time.time()
for _ in range(n_runs):
    result = eam_force_standard()
torch.cuda.synchronize() if device.type == 'cuda' else None
standard_total_time = time.time() - start_time

# CUDA版本基准
torch.cuda.synchronize() if device.type == 'cuda' else None
start_time = time.time()
for _ in range(n_runs):
    result = eam_force_cuda()
torch.cuda.synchronize() if device.type == 'cuda' else None
cuda_total_time = time.time() - start_time

print(f"标准版本平均耗时: {standard_total_time/n_runs:.4f} 秒")
print(f"CUDA版本平均耗时: {cuda_total_time/n_runs:.4f} 秒")
if cuda_total_time > 0:
    avg_speedup = standard_total_time / cuda_total_time
    print(f"平均加速比: {avg_speedup:.2f}x")
