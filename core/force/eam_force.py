import torch
import torch.nn as nn
from io_utils.eam_parser import EAMParser

class EAMForce(nn.Module):

    def __init__(self, eam_parser: EAMParser, molecular, use_tables: bool = True,
                 n_r: int = 8192, n_rho: int = 4096):
        super().__init__()
        self.parser = eam_parser
        self.molecular = molecular
        self.device = molecular.device
        self.cutoff = self.parser.cutoff
        self.element_map = self.parser.element_map
        self.use_tables = use_tables
        self.n_r = n_r
        self.n_rho = n_rho

        # A mapping from the atom's global index to its element type index (0, 1, 2...)
        self.atom_type_indices = self._map_atom_types_to_indices()

        self._prepare_splines_for_torch()
        if self.use_tables:
            self._build_uniform_tables()

    def _map_atom_types_to_indices(self):
        """Creates a tensor mapping each atom index to its EAM element type index."""
        type_indices = [self.element_map[atom_type] for atom_type in self.molecular.atom_types]
        return torch.tensor(type_indices, device=self.device, dtype=torch.long)

    def _prepare_splines_for_torch(self):
        """
        Converts numpy-based spline coefficients into torch tensors for all element types.
        """
        self.spline_rho_x = {}
        self.spline_rho_y_embed = {}
        self.spline_rho_y_deriv_embed = {}
        
        self.spline_r_x = {}
        self.spline_r_y_density = {}
        self.spline_r_y_deriv_density = {}
        
        self.spline_r_y_pair = {}
        self.spline_r_y_deriv_pair = {}

        for i, element in enumerate(self.parser.elements):
            # Embedding function F(rho) for element i
            self.spline_rho_x[i] = torch.from_numpy(self.parser.embedding_splines[i].x).to(self.device, dtype=torch.float32)
            self.spline_rho_y_embed[i] = torch.from_numpy(self.parser.embedding_splines[i].c).to(self.device, dtype=torch.float32)
            self.spline_rho_y_deriv_embed[i] = torch.from_numpy(self.parser.embedding_deriv_splines[i].c).to(self.device, dtype=torch.float32)

            # Electron density function f(r) for element i
            self.spline_r_x[i] = torch.from_numpy(self.parser.density_splines[i].x).to(self.device, dtype=torch.float32)
            self.spline_r_y_density[i] = torch.from_numpy(self.parser.density_splines[i].c).to(self.device, dtype=torch.float32)
            self.spline_r_y_deriv_density[i] = torch.from_numpy(self.parser.density_deriv_splines[i].c).to(self.device, dtype=torch.float32)

            # Pair potentials phi_ij(r)
            self.spline_r_y_pair[i] = {}
            self.spline_r_y_deriv_pair[i] = {}
            for j, _ in enumerate(self.parser.elements):
                if j in self.parser.pair_potential_splines[i]:
                    self.spline_r_y_pair[i][j] = torch.from_numpy(self.parser.pair_potential_splines[i][j].c).to(self.device, dtype=torch.float32)
                    self.spline_r_y_deriv_pair[i][j] = torch.from_numpy(self.parser.pair_potential_deriv_splines[i][j].c).to(self.device, dtype=torch.float32)

    def _torch_spline_eval(self, x, x_knots, y_coeffs):

        x = torch.clamp(x, x_knots[0], x_knots[-1])
        indices = torch.searchsorted(x_knots, x, right=True) - 1
        indices = torch.clamp(indices, 0, len(x_knots) - 2)
        x_i = x_knots[indices]
        dx = x - x_i
        if y_coeffs.shape[0] == 4:  # cubic
            c0 = y_coeffs[0, indices]
            c1 = y_coeffs[1, indices]
            c2 = y_coeffs[2, indices]
            c3 = y_coeffs[3, indices]
            val = c3 + dx * (c2 + dx * (c1 + dx * c0))
        elif y_coeffs.shape[0] == 3:  # quadratic
            c0 = y_coeffs[0, indices]
            c1 = y_coeffs[1, indices]
            c2 = y_coeffs[2, indices]
            val = c2 + dx * (c1 + dx * c0)
        elif y_coeffs.shape[0] == 2:  # linear
            c0 = y_coeffs[0, indices]
            c1 = y_coeffs[1, indices]
            val = c1 + dx * c0
        else:
            raise ValueError(f"Unexpected y_coeffs shape: {y_coeffs.shape}. 只支持线性、二次和三次样条")
        return val

    def _build_uniform_tables(self):

        with torch.no_grad():
            self.r_max = float(self.cutoff)
            self.dr = self.r_max / (self.n_r - 1)
            self.inv_dr = 1.0 / self.dr
            r_grid = torch.linspace(0.0, self.r_max, self.n_r, device=self.device)
            self.r_grid = r_grid

            self.density_table = {}
            self.density_deriv_table = {}
            self.pair_table = {}
            self.pair_deriv_table = {}

            self.embed_table = {}
            self.embed_deriv_table = {}
            self.embed_rho_min = {}
            self.embed_drho = {}
            self.embed_inv_drho = {}
            self.embed_n_rho = {}
            self.embed_rho_grid = {}

            for i in range(len(self.parser.elements)):
                # f_i(r) & f'_i(r)
                self.density_table[i] = self._torch_spline_eval(r_grid, self.spline_r_x[i], self.spline_r_y_density[i])
                self.density_deriv_table[i] = self._torch_spline_eval(r_grid, self.spline_r_x[i], self.spline_r_y_deriv_density[i])
                # pair φ_ij(r) & φ'_ij(r)
                self.pair_table[i] = {}
                self.pair_deriv_table[i] = {}
                for j in range(len(self.parser.elements)):
                    if j in self.spline_r_y_pair[i]:
                        self.pair_table[i][j] = self._torch_spline_eval(r_grid, self.spline_r_x[i], self.spline_r_y_pair[i][j])
                        self.pair_deriv_table[i][j] = self._torch_spline_eval(r_grid, self.spline_r_x[i], self.spline_r_y_deriv_pair[i][j])
                # Embedding F_i(rho) & F'_i(rho)
                rho_knots = self.spline_rho_x[i]
                rho_min = float(rho_knots[0])
                rho_max = float(rho_knots[-1])
                n_rho_i = self.n_rho
                drho = (rho_max - rho_min) / (n_rho_i - 1)
                inv_drho = 1.0 / drho
                rho_grid = torch.linspace(rho_min, rho_max, n_rho_i, device=self.device)
                self.embed_table[i] = self._torch_spline_eval(rho_grid, rho_knots, self.spline_rho_y_embed[i])
                self.embed_deriv_table[i] = self._torch_spline_eval(rho_grid, rho_knots, self.spline_rho_y_deriv_embed[i])
                self.embed_rho_min[i] = rho_min
                self.embed_drho[i] = drho
                self.embed_inv_drho[i] = inv_drho
                self.embed_n_rho[i] = n_rho_i
                self.embed_rho_grid[i] = rho_grid

    @torch.jit.ignore
    def _interp_linear_table(self, r, table):
        r_clamped = torch.clamp(r, 0.0, self.r_max * (1 - 1e-7))
        idx_f = r_clamped * self.inv_dr
        idx = idx_f.long()
        frac = idx_f - idx
        next_idx = torch.clamp(idx + 1, max=self.n_r - 1)
        val0 = table[idx]
        val1 = table[next_idx]
        return val0 + frac * (val1 - val0)

    @torch.jit.ignore
    def _interp_linear_table_rho(self, rho, i_type):
        rho_min = self.embed_rho_min[i_type]
        drho = self.embed_drho[i_type]
        inv_drho = self.embed_inv_drho[i_type]
        n_rho = self.embed_n_rho[i_type]
        rho_clamped = torch.clamp(rho, rho_min, rho_min + drho * (n_rho - 1) * (1 - 1e-7))
        idx_f = (rho_clamped - rho_min) * inv_drho
        idx = idx_f.long()
        frac = idx_f - idx
        next_idx = torch.clamp(idx + 1, max=n_rho - 1)
        val0 = self.embed_table[i_type][idx]
        val1 = self.embed_table[i_type][next_idx]
        return val0 + frac * (val1 - val0)

    @torch.jit.ignore
    def _interp_linear_table_rho_deriv(self, rho, i_type):
        rho_min = self.embed_rho_min[i_type]
        drho = self.embed_drho[i_type]
        inv_drho = self.embed_inv_drho[i_type]
        n_rho = self.embed_n_rho[i_type]
        rho_clamped = torch.clamp(rho, rho_min, rho_min + drho * (n_rho - 1) * (1 - 1e-7))
        idx_f = (rho_clamped - rho_min) * inv_drho
        idx = idx_f.long()
        frac = idx_f - idx
        next_idx = torch.clamp(idx + 1, max=n_rho - 1)
        val0 = self.embed_deriv_table[i_type][idx]
        val1 = self.embed_deriv_table[i_type][next_idx]
        return val0 + frac * (val1 - val0)

    def forward(self):
        if not self.use_tables:
            return self._forward_original()
        return self._forward_table()

    def _forward_original(self):
        coords = self.molecular.coordinates
        edge_index = self.molecular.graph_data.edge_index
        box_length = self.molecular.box_length
        num_atoms = coords.shape[0]
        row, col = edge_index
        dist_vec = coords[row] - coords[col]
        dist_vec -= torch.round(dist_vec / box_length) * box_length
        distances = torch.norm(dist_vec, dim=1)
        in_cutoff = distances < self.cutoff
        edge_index_cutoff = edge_index[:, in_cutoff]
        distances_cutoff = distances[in_cutoff]
        dist_vec_cutoff = dist_vec[in_cutoff]
        row_cutoff, col_cutoff = edge_index_cutoff
        row_types = self.atom_type_indices[row_cutoff]
        col_types = self.atom_type_indices[col_cutoff]
        electron_density_pairs = torch.zeros_like(distances_cutoff)
        for j_type in range(len(self.parser.elements)):
            mask = col_types == j_type
            if mask.any():
                r_vals = distances_cutoff[mask]
                density_vals = self._torch_spline_eval(r_vals, self.spline_r_x[j_type], self.spline_r_y_density[j_type])
                electron_density_pairs[mask] = density_vals
        rho = torch.zeros(num_atoms, device=self.device, dtype=torch.float32)
        rho.scatter_add_(0, row_cutoff, electron_density_pairs)
        embedding_energy_per_atom = torch.zeros_like(rho)
        for i_type in range(len(self.parser.elements)):
            mask = self.atom_type_indices == i_type
            if mask.any():
                rho_vals = rho[mask]
                embed_vals = self._torch_spline_eval(rho_vals, self.spline_rho_x[i_type], self.spline_rho_y_embed[i_type])
                embedding_energy_per_atom[mask] = embed_vals
        total_embedding_energy = torch.sum(embedding_energy_per_atom)
        pair_potential_values = torch.zeros_like(distances_cutoff)
        for i_type in range(len(self.parser.elements)):
            for j_type in range(len(self.parser.elements)):
                mask = (row_types == i_type) & (col_types == j_type)
                if mask.any():
                    r_vals = distances_cutoff[mask]
                    pair_vals = self._torch_spline_eval(r_vals, self.spline_r_x[i_type], self.spline_r_y_pair[i_type][j_type])
                    pair_potential_values[mask] = pair_vals
        total_pair_potential = 0.5 * torch.sum(pair_potential_values)
        total_energy = total_embedding_energy + total_pair_potential
        dF_drho = torch.zeros_like(rho)
        for i_type in range(len(self.parser.elements)):
            mask = self.atom_type_indices == i_type
            if mask.any():
                rho_vals = rho[mask]
                deriv_vals = self._torch_spline_eval(rho_vals, self.spline_rho_x[i_type], self.spline_rho_y_deriv_embed[i_type])
                dF_drho[mask] = deriv_vals
        d_density_dr = torch.zeros_like(distances_cutoff)
        d_pair_dr = torch.zeros_like(distances_cutoff)
        for i_type in range(len(self.parser.elements)):
            for j_type in range(len(self.parser.elements)):
                mask = (row_types == i_type) & (col_types == j_type)
                if mask.any():
                    r_vals = distances_cutoff[mask]
                    d_density_dr[mask] = self._torch_spline_eval(r_vals, self.spline_r_x[j_type], self.spline_r_y_deriv_density[j_type])
                    d_pair_dr[mask] = self._torch_spline_eval(r_vals, self.spline_r_x[i_type], self.spline_r_y_deriv_pair[i_type][j_type])
        dF_sum = dF_drho[row_cutoff] + dF_drho[col_cutoff]
        force_scalar_term = (dF_sum * d_density_dr) + d_pair_dr
        force_vectors = -force_scalar_term.unsqueeze(1) * (dist_vec_cutoff / (distances_cutoff.unsqueeze(1) + 1e-8))
        forces = torch.zeros_like(coords)
        forces.scatter_add_(0, row_cutoff.unsqueeze(1).expand_as(force_vectors), force_vectors)
        forces.scatter_add_(0, col_cutoff.unsqueeze(1).expand_as(force_vectors), -force_vectors)
        return {
            'energy': total_energy,
            'forces': forces
        }

    def _forward_table(self):
        coords = self.molecular.coordinates
        edge_index = self.molecular.graph_data.edge_index
        distances_all = self.molecular.graph_data.edge_attr
        box_length = self.molecular.box_length
        num_atoms = coords.shape[0]
        row, col = edge_index
        in_cutoff = distances_all < self.cutoff
        if not in_cutoff.any():
            return {'energy': torch.tensor(0.0, device=self.device), 'forces': torch.zeros_like(coords)}
        distances = distances_all[in_cutoff]
        row_cutoff = row[in_cutoff]
        col_cutoff = col[in_cutoff]
        rij_vec = coords[row_cutoff] - coords[col_cutoff]
        rij_vec -= torch.round(rij_vec / box_length) * box_length
        row_types = self.atom_type_indices[row_cutoff]
        col_types = self.atom_type_indices[col_cutoff]
        E = len(self.parser.elements)
        electron_density_pairs = torch.zeros_like(distances)
        for j_type in range(E):
            mask = (col_types == j_type)
            if mask.any():
                r_vals = distances[mask]
                electron_density_pairs[mask] = self._interp_linear_table(r_vals, self.density_table[j_type])
        rho = torch.zeros(num_atoms, device=self.device)
        rho.scatter_add_(0, row_cutoff, electron_density_pairs)
        embedding_energy_per_atom = torch.zeros_like(rho)
        for i_type in range(E):
            mask = (self.atom_type_indices == i_type)
            if mask.any():
                embedding_energy_per_atom[mask] = self._interp_linear_table_rho(rho[mask], i_type)
        total_embedding_energy = embedding_energy_per_atom.sum()
        pair_potential_values = torch.zeros_like(distances)
        for i_type in range(E):
            for j_type in range(E):
                mask = (row_types == i_type) & (col_types == j_type)
                if mask.any() and j_type in self.pair_table[i_type]:
                    r_vals = distances[mask]
                    pair_potential_values[mask] = self._interp_linear_table(r_vals, self.pair_table[i_type][j_type])
        total_pair_potential = 0.5 * pair_potential_values.sum()
        total_energy = total_embedding_energy + total_pair_potential
        # dF/drho
        dF_drho = torch.zeros_like(rho)
        for i_type in range(E):
            mask = (self.atom_type_indices == i_type)
            if mask.any():
                dF_drho[mask] = self._interp_linear_table_rho_deriv(rho[mask], i_type)
        # df/dr, dphi/dr
        d_density_dr = torch.zeros_like(distances)
        d_pair_dr = torch.zeros_like(distances)
        for i_type in range(E):
            for j_type in range(E):
                mask = (row_types == i_type) & (col_types == j_type)
                if mask.any() and j_type in self.pair_deriv_table[i_type]:
                    r_vals = distances[mask]
                    d_density_dr[mask] = self._interp_linear_table(r_vals, self.density_deriv_table[j_type])
                    d_pair_dr[mask] = self._interp_linear_table(r_vals, self.pair_deriv_table[i_type][j_type])
        dF_sum = dF_drho[row_cutoff] + dF_drho[col_cutoff]
        force_scalar = (dF_sum * d_density_dr) + d_pair_dr
        force_vec = -force_scalar.unsqueeze(1) * (rij_vec / (distances.unsqueeze(1) + 1e-8))
        forces = torch.zeros_like(coords)
        forces.scatter_add_(0, row_cutoff.unsqueeze(1).expand_as(force_vec), force_vec)
        forces.scatter_add_(0, col_cutoff.unsqueeze(1).expand_as(force_vec), -force_vec)
        return {'energy': total_energy, 'forces': forces}
