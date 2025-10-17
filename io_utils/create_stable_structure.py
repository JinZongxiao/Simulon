import numpy as np
import torch

def create_bcc_structure(n_cells, lattice_param, filename):

    bcc_base = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ]) * lattice_param
    
    atoms = []
    
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                cell_origin = np.array([i, j, k]) * lattice_param
                for base_pos in bcc_base:
                    atom_pos = cell_origin + base_pos
                    atoms.append(atom_pos)
    
    atoms = np.array(atoms)
    n_atoms = len(atoms)
    
    box_size = n_cells * lattice_param
    
    # print(f"创建了 {n_atoms} 个W原子")
    print(f"Creat {n_atoms} W atoms")
    # print(f"盒子尺寸: {box_size:.3f} Å")
    print(f"Box size: {box_size:.3f} Å")
    # print(f"密度: {n_atoms / box_size**3:.4f} atoms/Å³")
    print(f"Density: {n_atoms / box_size**3:.4f} atoms/Å³")
    
    with open(filename, 'w') as f:
        f.write(f"{n_atoms}\n")
        f.write(f"BCC W structure with lattice parameter {lattice_param:.3f} A\n")
        for atom in atoms:
            f.write(f"W {atom[0]:.6f} {atom[1]:.6f} {atom[2]:.6f}\n")
    
    return n_atoms, box_size

def create_bcc_structure_and_verify(n_cells,
                                    filename,
                                    lattice_param=3.2):


    n_atoms, box_size = create_bcc_structure(
        n_cells=n_cells, 
        lattice_param=lattice_param,
        filename=filename
    )
    print(f"Theoretical nearest neighbor distance: {lattice_param * np.sqrt(3)/2:.3f} Å")
    print(f"Theoretical second nearest neighbor distance: {lattice_param:.3f} Å")
    print(f"Theoretical nearest neighbor distance: {lattice_param * np.sqrt(3)/2:.3f} Å")
    print(f"Theoretical second nearest neighbor distance: {lattice_param:.3f} Å")
    print(f"BCC structure created with {n_atoms} atoms and box size {box_size:.3f} Å")
    print(f"File saved to {filename}")
    


lattice_param = 3.2  # Å

n_cells = 25
n_atoms, box_size = create_bcc_structure(
    n_cells=n_cells, 
    lattice_param=lattice_param,
    filename='C:/Users/Administrator/Desktop/phd_workspace/code_repo/Simulon/run_data/W31250.xyz'
)

# print(f"\n理论最近邻距离: {lattice_param * np.sqrt(3)/2:.3f} Å")
print(f"Theoretical nearest neighbor distance: {lattice_param * np.sqrt(3)/2:.3f} Å")
# print(f"理论次近邻距离: {lattice_param:.3f} Å")
print(f"Theoretical second nearest neighbor distance: {lattice_param:.3f} Å")

from io_utils.reader import AtomFileReader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

molecular = AtomFileReader(
    filename='C:/Users/Administrator/Desktop/phd_workspace/code_repo/Simulon/run_data/W31250.xyz',
    box_length=box_size,
    cutoff=6.0,
    device=device,
    is_fs=True
)

coords = molecular.coordinates
n_atoms = len(molecular.atom_types)

min_distances = []
for i in range(min(10, n_atoms)):  
    distances = []
    for j in range(n_atoms):
        if i != j:
            dist_vec = coords[i] - coords[j]
            dist_vec -= torch.round(dist_vec / molecular.box_length) * molecular.box_length
            dist = torch.norm(dist_vec).item()
            if dist > 0.1:
                distances.append(dist)
    if distances:
        min_distances.append(min(distances))

if min_distances:
    avg_min_dist = sum(min_distances) / len(min_distances)
    # print(f"实际平均最近邻距离: {avg_min_dist:.3f} Å")
    print(f"Actual average nearest neighbor distance: {avg_min_dist:.3f} Å")

from io_utils.eam_parser import EAMParser
from core.force.eam_force import EAMForce

eam_parser = EAMParser(filepath='C:/Users/Administrator/Desktop/phd_workspace/code_repo/Simulon/run_data/WRe_YC2.eam.fs', device=device)
eam_force_field = EAMForce(eam_parser=eam_parser, molecular=molecular)

result = eam_force_field()
forces = result['forces']
force_magnitudes = torch.norm(forces, dim=1)

# print(f"\n=== 新结构的力分析 ===")
# print(f"力的范围: {force_magnitudes.min().item():.2f} - {force_magnitudes.max().item():.2f} eV/Å")
print(f"Force range: {force_magnitudes.min().item():.2f} - {force_magnitudes.max().item():.2f} eV/Å")
# print(f"平均力大小: {force_magnitudes.mean().item():.2f} eV/Å")
print(f"Average force magnitude: {force_magnitudes.mean().item():.2f} eV/Å")
# print(f"总能量: {result['energy'].item():.2f} eV")
print(f"Total energy: {result['energy'].item():.2f} eV")
# print(f"每原子能量: {result['energy'].item()/n_atoms:.2f} eV/atom")
print(f"Energy per atom: {result['energy'].item()/n_atoms:.2f} eV/atom")
