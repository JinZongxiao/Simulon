from collections import defaultdict
import time  # added for timing

import torch

from core.element_info import get_element_mass, get_element_iron_num

from pymatgen.core import Structure, Lattice

from torch_geometric.data import Data

from scipy.spatial import cKDTree
from core.neighbor_search.gpu_kdtree import find_neighbors_gpu_pbc
from core.box import Box


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class AtomFileReader:

    def __init__(self,
                 filename,
                 box_length,
                 cutoff,
                 device=DEVICE,
                 parameter = None,
                 skin_thickness = 5.0,
                 is_mlp = False,
                 is_fs = False,
                 is_switch = False,
                 switch_ratio: float = 0.9,
                 box_vectors = None,         # [3,3] 三斜格矢（可选），覆盖 box_length
                 ):
        # normalize device
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # ── Box 对象（正交或三斜） ──────────────────────────────────────────
        if box_vectors is not None:
            self.box = Box(box_vectors, device=device)
        else:
            self.box = Box(box_length, device=device)

        # 兼容旧代码的 box_length 属性（正交盒子）
        self.box_length_cpu = box_length
        self.box_length = self.box.diag[0] if self.box.is_orthogonal \
                          else torch.tensor(box_length, device=device, dtype=torch.float64)
        self.cutoff = torch.tensor(cutoff,device=self.device)
        self.parameter = parameter
        self.is_mlp = is_mlp
        self.is_fs = is_fs
        self.is_switch = is_switch
        self.switch_ratio = float(switch_ratio)

        self.atom_count = 0
        self.atom_types = []
        self.coordinates = []
        self.atom_mass = []
        self.atom_iron_num = []

        self.read_file(filename)

        self.atom_set = self.get_atom_set()

        self.atom_velocities = torch.zeros(self.atom_count, 3,device=self.device)

        self.element_to_id = []
        element_to_id_result_temp = {}
        for element in self.atom_types:
            if element not in element_to_id_result_temp:
                element_to_id_result_temp[element] = len(element_to_id_result_temp)
            self.element_to_id.append(element_to_id_result_temp[element])
        self.element_ids = torch.tensor(self.element_to_id, device=self.device)

        self.skin_thickness = torch.tensor(skin_thickness,device=device)
        self.verlet_cutoff = self.cutoff + self.skin_thickness
        self.last_positions = None
        self.needs_update = True


        self.graph_data = self.initialize_pyg_data(self.verlet_cutoff)
        if not is_mlp and not is_fs:
            self.pair_params = self.initial_parameters()

        self.profile = {'neighbor_time': 0.0, 'neighbor_builds': 0, 'effective_edges_last': 0}

    def get_atom_set(self):
        count_dict = {elem: self.atom_types.count(elem) for elem in self.atom_types}
        duplicated_set = {elem for elem, count in count_dict.items() if count > 1}
        return list(duplicated_set)

    def get_atom_mass(self):
        return self.atom_mass

    def get_atom_num(self):
        return self.atom_count

    def get_atom_type_array(self):
        return self.atom_types

    def get_atom_coordinates(self):
        return self.coordinates

    def get_parameter(self, param_name: str):
        num_edges = self.graph_data.index_pairs.shape[0]
        
        if num_edges == 0:
            return torch.empty(0, device=self.device)
        
        # 缓存CPU数据，避免重复转换
        if not hasattr(self, '_index_pairs_cpu_cache'):
            unique_pairs = torch.unique(self.graph_data.index_pairs, dim=0)
            if len(unique_pairs) == 1:
                self._index_pairs_cpu_cache = {'single': str(unique_pairs[0].cpu().numpy())}
            else:
                self._index_pairs_cpu_cache = {'multiple': self.graph_data.index_pairs.cpu().numpy()}
        
        if 'single' in self._index_pairs_cpu_cache:
            # All pairs are the same, optimized path
            pair_key = self._index_pairs_cpu_cache['single']
            param_value = self.parameter[pair_key][param_name]
            return torch.full((num_edges,), param_value, 
                            dtype=torch.float32, device=self.device)
        else:
            # Multiple pair types, use cached CPU data
            index_pairs_cpu = self._index_pairs_cpu_cache['multiple']
            param_values = []
            for i in range(num_edges):
                pair_key = str(index_pairs_cpu[i])
                param_values.append(self.parameter[pair_key][param_name])
            
            return torch.tensor(param_values, device=self.device)

    def update_coordinates(self, coordinates):
        if self.last_positions is not None:
            displacement = self.box.minimum_image(coordinates - self.last_positions)
            max_displacement = torch.max(torch.norm(displacement, dim=1))

            if max_displacement > self.skin_thickness / 2:
                self.needs_update = True
                self.last_positions = coordinates.detach().clone()
            else:
                self.needs_update = False
        else:
            self.needs_update = True
            self.last_positions = coordinates.detach().clone()

        self.coordinates = coordinates
        self.graph_data.pos = coordinates

        if self.needs_update:
            self.update_neighbor_list()
            if not self.is_mlp and not self.is_fs:
                self.pair_params = self.initial_parameters()
        else:
            self.graph_data.edge_attr = self.calculate_edge_attr(
                coordinates,
                self.graph_data.edge_index,
                self.box,
            )

    def initial_parameters(self):
        first_key = next(iter(self.parameter))
        first_value = self.parameter[first_key]
        param_list = []
        
        # Get the number of edges
        num_edges = self.graph_data.index_pairs.shape[0]
        
        if num_edges == 0:
            # No edges, return empty tensors
            for key in first_value:
                param_list.append(torch.empty(0, device=self.device))
            return param_list
        
        # 缓存CPU数据，避免重复转换
        if not hasattr(self, '_index_pairs_cpu_cache'):
            unique_pairs = torch.unique(self.graph_data.index_pairs, dim=0)
            if len(unique_pairs) == 1:
                self._index_pairs_cpu_cache = {'single': str(unique_pairs[0].cpu().numpy())}
            else:
                self._index_pairs_cpu_cache = {'multiple': self.graph_data.index_pairs.cpu().numpy()}
        
        if 'single' in self._index_pairs_cpu_cache:
            # All pairs are the same, optimized path
            pair_key = self._index_pairs_cpu_cache['single']
            # Keep the original order of parameters from the dictionary
            for key in first_value.keys():  # This preserves the order
                param_value = self.parameter[pair_key][key]
                param_tensor = torch.full((num_edges,), param_value, 
                                        dtype=torch.float32, device=self.device)
                param_list.append(param_tensor)
        else:
            # Multiple pair types, use cached CPU data
            index_pairs_cpu = self._index_pairs_cpu_cache['multiple']
            # Keep the original order of parameters from the dictionary
            for key in first_value.keys():  # This preserves the order
                # Create parameter values for all edges
                param_values = []
                for i in range(num_edges):
                    pair_key = str(index_pairs_cpu[i])
                    param_values.append(self.parameter[pair_key][key])
                
                param_tensor = torch.tensor(param_values, dtype=torch.float32, device=self.device)
                param_list.append(param_tensor)
                
        return param_list

    def update_neighbor_list(self):
        t0 = time.perf_counter()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        # edge_index, edge_attr = self.find_neighbors(self.coordinates, self.verlet_cutoff)
        edge_index, edge_attr = self.find_neighbors_kdtree(self.coordinates, self.verlet_cutoff)
        self.graph_data.edge_index = edge_index
        self.graph_data.edge_attr = edge_attr
        element_ids = torch.tensor(self.element_to_id, device=self.device)
        self.graph_data.element_edge_0 = element_ids[edge_index[0]]
        self.graph_data.element_edge_1 = element_ids[edge_index[1]]
        self.graph_data.index_pairs = torch.stack(
            [self.graph_data.element_edge_0, self.graph_data.element_edge_1],
            dim=1
        )
        # invalidate cached pair mapping if any
        if hasattr(self, '_index_pairs_cpu_cache'):
            delattr(self, '_index_pairs_cpu_cache')
        self.needs_update = False
        # record effective edges (<= cutoff) for diagnostics
        try:
            mask = self.graph_data.edge_attr <= self.cutoff
            self.profile['effective_edges_last'] = int(mask.sum().item())
        except Exception:
            self.profile['effective_edges_last'] = 0
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.profile['neighbor_time'] += dt
        self.profile['neighbor_builds'] += 1
        # debug: print(f"Neighbor list updated: {self.graph_data.edge_index.shape[1]} edges")

    # Helper methods for cutoff filtering
    def get_cutoff_mask(self):
        if self.graph_data.edge_attr is None:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        return self.graph_data.edge_attr <= self.cutoff

    def effective_edge_count(self):
        return int(self.get_cutoff_mask().sum().item())

    def update_velocities(self, velocities):
        self.atom_velocities = velocities

    def read_file(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            if not lines:
                raise ValueError(f"Error: The file {filename} is empty.")
            skip_lines = 0
            if filename.lower().endswith('.xyz'):
                skip_lines = 1
            try:
                self.atom_count = int(lines[0].strip())
                for line in lines[1 + skip_lines:]:
                    parts = line.split()
                    if len(parts) >= 4:
                        atom_type = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        self.atom_types.append(atom_type)
                        self.atom_mass.append(get_element_mass(atom_type))
                        self.atom_iron_num.append(get_element_iron_num(atom_type))
                        self.coordinates.append([x,y,z])
            except Exception as e:
                raise ValueError(f"Error: {e}")
            self.atom_mass = torch.tensor(self.atom_mass, device=self.device)
            self.atom_iron_num = torch.tensor(self.atom_iron_num, device=self.device)
            self.coordinates = torch.tensor(self.coordinates, device=self.device)

    def to_pymatgen_structure(self):
        coords = self.coordinates.detach().cpu().numpy()
        lattice = Lattice([[self.box_length_cpu, 0, 0],
                           [0, self.box_length_cpu, 0],
                           [0, 0, self.box_length_cpu]])
        structure = Structure(lattice,
                              self.atom_types,
                              coords,
                              coords_are_cartesian=True)
        return structure

    def create_velocity_gaussian(self, temperature, seed):
        torch.manual_seed(seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(seed)

        k_b_ev = 8.617333262e-5  # eV/(K·atom)
        ev_to_gA2ps2 = 1.60218e-20  # 1 eV -> g·Å²/ps²

        #g/atom
        # 1 amu = 1 g/mol
        # /N_A
        # mass_g_per_atom = self.atom_mass / 6.02214076e23
        mass_g_per_atom = self.atom_mass


        # A/ps
        sigma_squared = (k_b_ev * temperature * ev_to_gA2ps2) / mass_g_per_atom
        sigma = torch.sqrt(sigma_squared)

        velocities = torch.randn((self.atom_count, 3), device=self.device) * sigma.view(-1, 1)

        total_momentum = torch.sum(velocities * mass_g_per_atom.view(-1, 1), dim=0)
        total_mass = torch.sum(mass_g_per_atom)
        velocities -= total_momentum / total_mass

        self.atom_velocities = velocities

    def set_maxwell_boltzmann_velocity(self,temperature):
        """Sample initial velocities from Maxwell-Boltzmann at temperature (K).
        Units: eV, Å, ps, amu. Mass is converted to internal M=eV·ps^2/Å^2 so that
        v ~ N(0, kB T / M) in each Cartesian component. Removes COM momentum.
        """
        natoms = self.atom_count
        device = self.device
        kB_eV = 8.617333262e-5  # eV/K
        MASS_CONV = 1.036427e-4 # 1 amu*(Å/ps)^2 = 1.036427e-4 eV

        # Per-atom mass in internal units (N,1)
        m_internal = (self.atom_mass * MASS_CONV).unsqueeze(-1)  # (N,1)
        sigma = torch.sqrt(kB_eV * temperature / m_internal)     # (N,1)
        velocities = torch.randn((natoms, 3), device=device, dtype=self.coordinates.dtype) * sigma

        # Remove center-of-mass momentum (mass-weighted)
        total_p = (velocities * m_internal).sum(dim=0)   # (3,)
        total_m = m_internal.sum()                       # scalar
        velocities = velocities - total_p / total_m

        self.atom_velocities = velocities

    def initialize_pyg_data(self, cutoff):
        atom_types_index = torch.tensor(list(range(0, len(self.atom_types))), device=self.device)
        pos = self.coordinates
        pos.requires_grad_(True)
        # edge_index, edge_attr = self.find_neighbors(pos, cutoff)
        edge_index, edge_attr = self.find_neighbors_kdtree(pos, cutoff)
        element_ids = torch.tensor(self.element_to_id, device=self.device)
        element_edge_0 = element_ids[edge_index[0]]  # 直接索引映射
        element_edge_1 = element_ids[edge_index[1]]
        index_pairs = torch.stack([element_edge_0,element_edge_1], dim=1)
        # edge_attr.requires_grad_(True)
        return Data(
            x=torch.stack([atom_types_index,self.atom_iron_num, self.atom_mass], dim=1).float().to(self.device),
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            # box_size=torch.tensor([box_length] * 3, device=self.device),
            index_pairs = index_pairs,
            element_edge_0 = element_edge_0,
            element_edge_1 = element_edge_1,
            energy = torch.zeros(0, device=self.device),
            forces = torch.empty((self.atom_count, 3), device=self.device)
        )

    @staticmethod
    def calculate_edge_attr(pos, edge_index, box):
        """
        box: Box 对象 或 标量/张量 box_length（向后兼容）。
        """
        row, col = edge_index[0], edge_index[1]
        rij = pos[row] - pos[col]
        if isinstance(box, Box):
            rij = box.minimum_image(rij)
        else:
            bl = float(box) if not torch.is_tensor(box) else box.float()
            rij = rij - bl * torch.round(rij / bl)
        return torch.norm(rij, dim=1)

    @staticmethod
    def build_cell_list(pos, box_length, cutoff):
        cell_size = cutoff  
        n_cells = int(box_length // cell_size)
        cell_indices = (pos / cell_size).floor().long() % n_cells
        return cell_indices, n_cells, cell_size

    @staticmethod
    def assign_particles_to_cells(cell_indices):
        cell_dict = defaultdict(list)
        for particle_idx, idx in enumerate(cell_indices):
            cell_dict[tuple(idx.tolist())].append(particle_idx)
        return cell_dict

    def local_neighbor_search(self,pos, cell_dict, n_cells, cutoff, box_length):
        neighbors = []
        edge_attr_list  = []
        neighbor_offsets = []
        for dz in [0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == -1: continue
                    if dz == 0 and dy == -1: continue
                    if dx == 0 and dy == 0 and dz == 0: continue
                    neighbor_offsets.append((dx, dy, dz))
        for cell_idx, particles_in_cell in cell_dict.items():
            # 1. 计算单元格内部的原子对
            for i_idx, i in enumerate(particles_in_cell):
                for j in particles_in_cell[i_idx + 1:]:
                    rij = pos[j] - pos[i]
                    rij -= box_length * torch.round(rij / box_length)
                    dist = torch.norm(rij)
                    if dist < cutoff:
                        neighbors.append((i, j))
                        edge_attr_list.append(dist)

            # 2. 计算与13个邻居单元格之间的原子对
            for offset in neighbor_offsets:
                neighbor_cell_idx = (
                    (cell_idx[0] + offset[0]) % n_cells,
                    (cell_idx[1] + offset[1]) % n_cells,
                    (cell_idx[2] + offset[2]) % n_cells
                )
                if neighbor_cell_idx in cell_dict:
                    for i in particles_in_cell:
                        for j in cell_dict[neighbor_cell_idx]:
                            rij = pos[j] - pos[i]
                            rij -= box_length * torch.round(rij / box_length)
                            dist = torch.norm(rij)
                            if dist < cutoff:
                                neighbors.append((i, j))
                                edge_attr_list.append(dist)
        # for i in range(len(pos)):
        #     cell_idx = tuple((pos[i] / (box_length / n_cells)).floor().long().tolist())
        #     for dx in [-1, 0, 1]:
        #         for dy in [-1, 0, 1]:
        #             for dz in [-1, 0, 1]:
        #                 neighbor_cell = (
        #                     (cell_idx[0] + dx) % n_cells,
        #                     (cell_idx[1] + dy) % n_cells,
        #                     (cell_idx[2] + dz) % n_cells
        #                 )
        #                 for j in cell_dict.get(neighbor_cell, []):
        #                     if j <= i: continue
        #                     rij = pos[j] - pos[i]
        #                     # rij -= torch.round(rij / box_length) * box_length
        #                     rij = rij - box_length * torch.round(rij / box_length)
        #                     dist = torch.norm(rij)
        #                     if dist < cutoff:
        #                         neighbors.append((i, j))
        #                         edge_attr_list.append(dist)
        edge_index = torch.tensor(neighbors,device=self.device).t().contiguous()
        edge_attr  = torch.stack(edge_attr_list)
        return edge_index, edge_attr

    def find_neighbors(self, pos, cutoff):
        cell_indices, n_cells, cell_size = self.build_cell_list(pos, self.box_length, cutoff)

        cell_dict = self.assign_particles_to_cells(cell_indices)

        edge_index,edge_attr = self.local_neighbor_search(pos, cell_dict, n_cells, cutoff,self.box_length)
        return edge_index,edge_attr

    def expand_pos_pbc(self,pos):
        expanded_pos = []
        expanded_indices = []
        expanded_shifts = [] 
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    shift = torch.tensor([dx, dy, dz], device=self.device) * self.box_length
                    expanded_pos.append(pos + shift)
                    expanded_indices.extend(range(len(pos)))
                    expanded_shifts.extend([[dx, dy, dz]] * len(pos))
        expanded_pos = torch.cat(expanded_pos, dim=0)
        expanded_indices = torch.tensor(expanded_indices, device=self.device)
        expanded_shifts = torch.tensor(expanded_shifts, device=self.device)
        return expanded_pos, expanded_indices, expanded_shifts

    def find_neighbors_kdtree(self, pos, cutoff):
        edge_index, edge_attr = find_neighbors_gpu_pbc(pos, cutoff, self.box)
        # 确保 i < j（上三角，消除重复边）
        mask = edge_index[0] < edge_index[1]
        edge_index = edge_index[:, mask]
        # 用 box.minimum_image 重新精确计算距离
        edge_attr = self.calculate_edge_attr(pos, edge_index, self.box)
        return edge_index, edge_attr
        
    def find_neighbors_kdtree_cpu(self, pos, cutoff):
        expanded_pos, expanded_indices, expanded_shifts = self.expand_pos_pbc(pos)
        tree = cKDTree(expanded_pos.detach().cpu().numpy())
        pairs = tree.query_pairs(cutoff.cpu().numpy())
        pair_list = []
        for pair in pairs:
            if (99 >= pair[0] >= 0) or (99 >= pair[1] >= 0):
                pair_list.append(pair)
        edge_index = torch.tensor(pair_list, device=self.device).t().contiguous()
        edge_index = expanded_indices[edge_index]
        edge_attr = self.calculate_edge_attr(expanded_pos, edge_index, self.box_length)
        return edge_index, edge_attr

