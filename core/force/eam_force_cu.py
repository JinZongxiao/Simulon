import torch
import torch.nn as nn
import numpy as np

class EAMForceCUDA(nn.Module):

    def __init__(self, eam_parser, molecular):
        super().__init__()
        self.parser = eam_parser
        self.molecular = molecular
        self.device = molecular.device
        self.cutoff = eam_parser.cutoff
        
        self.atom_type_indices = torch.zeros(len(molecular.atom_types), dtype=torch.long, device=self.device)
        for i, atom_type in enumerate(molecular.atom_types):
            if atom_type in eam_parser.element_map:
                self.atom_type_indices[i] = eam_parser.element_map[atom_type]
        
        self._prepare_cuda_splines()
        
        self._load_cuda_kernels()
    
    def _prepare_cuda_splines(self):
        n_elements = len(self.parser.elements)
        
        self.spline_r_x = torch.from_numpy(
            self.parser.density_splines[0].x
        ).to(self.device, dtype=torch.float32)
        
        self.n_spline_points = len(self.spline_r_x)
        self.dr = self.spline_r_x[1] - self.spline_r_x[0]
        
        density_spline_coeffs = []
        for i in range(n_elements):
            coeffs = torch.from_numpy(self.parser.density_splines[i].c).to(self.device, dtype=torch.float32)
            density_spline_coeffs.append(coeffs)
        self.density_spline_coeffs = torch.stack(density_spline_coeffs)  # [n_elements, 4, n_points-1]
        
        self.embed_x = []
        self.embed_spline_coeffs = []
        for i in range(n_elements):
            embed_x = torch.from_numpy(self.parser.embedding_splines[i].x).to(self.device, dtype=torch.float32)
            embed_spline_coeffs = torch.from_numpy(self.parser.embedding_splines[i].c).to(self.device, dtype=torch.float32)
            self.embed_x.append(embed_x)
            self.embed_spline_coeffs.append(embed_spline_coeffs)
        
        self.embed_deriv_coeffs = []
        for i in range(n_elements):
            deriv_coeffs = torch.from_numpy(self.parser.embedding_deriv_splines[i].c).to(self.device, dtype=torch.float32)
            self.embed_deriv_coeffs.append(deriv_coeffs)
        
        density_deriv_coeffs = []
        for i in range(n_elements):
            coeffs = torch.from_numpy(self.parser.density_deriv_splines[i].c).to(self.device, dtype=torch.float32)
            density_deriv_coeffs.append(coeffs)
        self.density_deriv_coeffs = torch.stack(density_deriv_coeffs)
        
        pair_spline_coeffs = []
        for i in range(n_elements):
            pair_i = []
            for j in range(n_elements):
                if j in self.parser.pair_potential_splines[i]:
                    coeffs = torch.from_numpy(self.parser.pair_potential_splines[i][j].c).to(self.device, dtype=torch.float32)
                else:
                    coeffs = torch.zeros((4, self.n_spline_points-1), device=self.device, dtype=torch.float32)
                pair_i.append(coeffs)
            pair_spline_coeffs.append(torch.stack(pair_i))
        self.pair_spline_coeffs = torch.stack(pair_spline_coeffs)
        
        pair_deriv_coeffs = []
        for i in range(n_elements):
            pair_deriv_i = []
            for j in range(n_elements):
                if j in self.parser.pair_potential_deriv_splines[i]:
                    coeffs = torch.from_numpy(self.parser.pair_potential_deriv_splines[i][j].c).to(self.device, dtype=torch.float32)
                else:
                    coeffs = torch.zeros((3, self.n_spline_points-1), device=self.device, dtype=torch.float32)
                pair_deriv_i.append(coeffs)
            pair_deriv_coeffs.append(torch.stack(pair_deriv_i))
        self.pair_deriv_coeffs = torch.stack(pair_deriv_coeffs)
    
    def _load_cuda_kernels(self):
        self.use_cuda_kernels = True
        
        try:

            raise ImportError("CUDA kernels not compiled yet")
        except ImportError:
            print("→ 使用PyTorch优化实现")
            self.use_cuda_kernels = False
    
    def _compute_density_cuda(self, distances, row_indices, col_indices):
        if self.use_cuda_kernels:
            pass
        else:
            return self._compute_density_pytorch(distances, row_indices, col_indices)
    
    def _compute_density_pytorch(self, distances, row_indices, col_indices):
        n_pairs = len(distances)
        density_contributions = torch.zeros_like(distances)
        
        for j_type in range(len(self.parser.elements)):
            mask = self.atom_type_indices[col_indices] == j_type
            if mask.any():
                r_vals = distances[mask]
                
                r_clamped = torch.clamp(r_vals, self.spline_r_x[0], self.spline_r_x[-1])
                indices = torch.searchsorted(self.spline_r_x, r_clamped, right=True) - 1
                indices = torch.clamp(indices, 0, len(self.spline_r_x) - 2)
                
                x_i = self.spline_r_x[indices]
                dx = r_clamped - x_i
                
                c0 = self.density_spline_coeffs[j_type, 0, indices]
                c1 = self.density_spline_coeffs[j_type, 1, indices]
                c2 = self.density_spline_coeffs[j_type, 2, indices]
                c3 = self.density_spline_coeffs[j_type, 3, indices]
                
                vals = c3 + dx * (c2 + dx * (c1 + dx * c0))
                density_contributions[mask] = vals
        
        return density_contributions
    
    def _compute_embedding_energy(self, rho):
        n_atoms = len(rho)
        embedding_energies = torch.zeros_like(rho)
        
        for i_type in range(len(self.parser.elements)):
            mask = self.atom_type_indices == i_type
            if mask.any():
                rho_vals = rho[mask]
                
                embed_x = self.embed_x[i_type]
                embed_spline_coeffs = self.embed_spline_coeffs[i_type]
                
                rho_clamped = torch.clamp(rho_vals, embed_x[0], embed_x[-1])
                indices = torch.searchsorted(embed_x, rho_clamped, right=True) - 1
                indices = torch.clamp(indices, 0, len(embed_x) - 2)
                
                x_i = embed_x[indices]
                dx = rho_clamped - x_i
                
                c0 = embed_spline_coeffs[0, indices]
                c1 = embed_spline_coeffs[1, indices]
                c2 = embed_spline_coeffs[2, indices]
                c3 = embed_spline_coeffs[3, indices]
                
                vals = c3 + dx * (c2 + dx * (c1 + dx * c0))
                embedding_energies[mask] = vals
        
        return embedding_energies
    
    def _compute_pair_potential(self, distances, row_indices, col_indices):
        n_pairs = len(distances)
        pair_energies = torch.zeros_like(distances)
        
        for i_type in range(len(self.parser.elements)):
            for j_type in range(len(self.parser.elements)):
                mask = (self.atom_type_indices[row_indices] == i_type) & \
                       (self.atom_type_indices[col_indices] == j_type)
                
                if mask.any():
                    r_vals = distances[mask]
                    
                    r_clamped = torch.clamp(r_vals, self.spline_r_x[0], self.spline_r_x[-1])
                    indices = torch.searchsorted(self.spline_r_x, r_clamped, right=True) - 1
                    indices = torch.clamp(indices, 0, len(self.spline_r_x) - 2)
                    
                    x_i = self.spline_r_x[indices]
                    dx = r_clamped - x_i
                    
                    c0 = self.pair_spline_coeffs[i_type, j_type, 0, indices]
                    c1 = self.pair_spline_coeffs[i_type, j_type, 1, indices]
                    c2 = self.pair_spline_coeffs[i_type, j_type, 2, indices]
                    c3 = self.pair_spline_coeffs[i_type, j_type, 3, indices]
                    
                    vals = c3 + dx * (c2 + dx * (c1 + dx * c0))
                    pair_energies[mask] = vals
        
        return pair_energies
    
    def forward(self):
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
        
        density_contributions = self._compute_density_cuda(
            distances_cutoff, row_cutoff, col_cutoff
        )
        
        rho = torch.zeros(num_atoms, device=self.device, dtype=torch.float32)
        rho.scatter_add_(0, row_cutoff, density_contributions)
        
        embedding_energies = self._compute_embedding_energy(rho)
        total_embedding_energy = torch.sum(embedding_energies)
        
        pair_energies = self._compute_pair_potential(
            distances_cutoff, row_cutoff, col_cutoff
        )
        total_pair_potential = 0.5 * torch.sum(pair_energies)
        
        total_energy = total_embedding_energy + total_pair_potential
        
        forces = self._compute_forces(
            coords, edge_index_cutoff, distances_cutoff, dist_vec_cutoff,
            rho, embedding_energies
        )
        
        return {
            'energy': total_energy,
            'forces': forces,
            'embedding_energy': total_embedding_energy,
            'pair_potential': total_pair_potential
        }
    
    def _compute_forces(self, coords, edge_index_cutoff, distances_cutoff, dist_vec_cutoff, rho, embedding_energies):
        num_atoms = coords.shape[0]
        forces = torch.zeros_like(coords)
        
        row_cutoff, col_cutoff = edge_index_cutoff
        
        dF_drho = torch.zeros_like(rho)
        for i_type in range(len(self.parser.elements)):
            mask = self.atom_type_indices == i_type
            if mask.any():
                rho_vals = rho[mask]
                
                embed_x = self.embed_x[i_type]
                embed_deriv_coeffs = self.embed_deriv_coeffs[i_type]
                
                rho_clamped = torch.clamp(rho_vals, embed_x[0], embed_x[-1])
                indices = torch.searchsorted(embed_x, rho_clamped, right=True) - 1
                indices = torch.clamp(indices, 0, len(embed_x) - 2)
                
                x_i = embed_x[indices]
                dx = rho_clamped - x_i
                
                c0 = embed_deriv_coeffs[0, indices]  
                c1 = embed_deriv_coeffs[1, indices]  
                c2 = embed_deriv_coeffs[2, indices]
                
                deriv_vals = c2 + dx * (c1 + dx * c0)
                dF_drho[mask] = deriv_vals
        
        d_density_dr = torch.zeros_like(distances_cutoff)
        for i_type in range(len(self.parser.elements)):
            for j_type in range(len(self.parser.elements)):
                mask = (self.atom_type_indices[row_cutoff] == i_type) & \
                       (self.atom_type_indices[col_cutoff] == j_type)
                
                if mask.any():
                    r_vals = distances_cutoff[mask]
                    
                    r_clamped = torch.clamp(r_vals, self.spline_r_x[0], self.spline_r_x[-1])
                    indices = torch.searchsorted(self.spline_r_x, r_clamped, right=True) - 1
                    indices = torch.clamp(indices, 0, len(self.spline_r_x) - 2)
                    
                    x_i = self.spline_r_x[indices]
                    dx = r_clamped - x_i
                    
                    c0 = self.density_deriv_coeffs[j_type, 0, indices]  
                    c1 = self.density_deriv_coeffs[j_type, 1, indices]  
                    c2 = self.density_deriv_coeffs[j_type, 2, indices]
                    
                    deriv_vals = c2 + dx * (c1 + dx * c0)
                    d_density_dr[mask] = deriv_vals
        
        d_pair_dr = torch.zeros_like(distances_cutoff)
        for i_type in range(len(self.parser.elements)):
            for j_type in range(len(self.parser.elements)):
                mask = (self.atom_type_indices[row_cutoff] == i_type) & \
                       (self.atom_type_indices[col_cutoff] == j_type)
                
                if mask.any():
                    r_vals = distances_cutoff[mask]
                    
                    r_clamped = torch.clamp(r_vals, self.spline_r_x[0], self.spline_r_x[-1])
                    indices = torch.searchsorted(self.spline_r_x, r_clamped, right=True) - 1
                    indices = torch.clamp(indices, 0, len(self.spline_r_x) - 2)
                    
                    x_i = self.spline_r_x[indices]
                    dx = r_clamped - x_i
                    
                    c0 = self.pair_deriv_coeffs[i_type, j_type, 0, indices]  
                    c1 = self.pair_deriv_coeffs[i_type, j_type, 1, indices]  
                    c2 = self.pair_deriv_coeffs[i_type, j_type, 2, indices]  
                    
                    deriv_vals = c2 + dx * (c1 + dx * c0)
                    d_pair_dr[mask] = deriv_vals
        
        dF_sum = dF_drho[row_cutoff] + dF_drho[col_cutoff]
        force_scalar_term = (dF_sum * d_density_dr) + d_pair_dr
        force_vectors = -force_scalar_term.unsqueeze(1) * (dist_vec_cutoff / (distances_cutoff.unsqueeze(1) + 1e-8))
        
        forces.scatter_add_(0, row_cutoff.unsqueeze(1).expand_as(force_vectors), force_vectors)
        forces.scatter_add_(0, col_cutoff.unsqueeze(1).expand_as(force_vectors), -force_vectors)
        
        return forces

class EAMForceCUDAExt(nn.Module):
   
    def __init__(self, eam_parser, molecular, n_r: int = 8192, n_rho: int = 4096, use_extension: bool = True):
        super().__init__()
        self.parser = eam_parser
        self.molecular = molecular
        self.device = molecular.device
        self.cutoff = float(eam_parser.cutoff)
        self.n_r = n_r
        self.n_rho = n_rho
        self.use_extension = use_extension and torch.cuda.is_available()
        self.atom_type_indices = torch.tensor([
            eam_parser.element_map[a] for a in molecular.atom_types
        ], device=self.device, dtype=torch.long)
        self.E = len(self.parser.elements)
        self._build_uniform_tables()
        self._maybe_load_extension()

    def _build_uniform_tables(self):
        with torch.no_grad():
            self.r_max = self.cutoff
            self.dr = self.r_max / (self.n_r - 1)
            self.inv_dr = 1.0 / self.dr
            r_grid = torch.linspace(0.0, self.r_max, self.n_r, device=self.device)
            density_table = []
            density_deriv_table = []
            pair_table = []
            pair_deriv_table = []
            self.embed_rho_min = {}
            self.embed_drho = {}
            self.embed_inv_drho = {}
            self.embed_n = {}
            self.embed_table = {}
            self.embed_deriv_table = {}
            for i in range(self.E):
                # f_i(r)
                f_i = self._spline_eval(self.parser.density_splines[i], r_grid)
                df_i = self._spline_eval(self.parser.density_deriv_splines[i], r_grid)
                density_table.append(f_i)
                density_deriv_table.append(df_i)
            for i in range(self.E):
                row_phi = []
                row_dphi = []
                for j in range(self.E):
                    if j in self.parser.pair_potential_splines[i]:
                        phi_ij = self._spline_eval(self.parser.pair_potential_splines[i][j], r_grid)
                        dphi_ij = self._spline_eval(self.parser.pair_potential_deriv_splines[i][j], r_grid)
                    else:
                        phi_ij = torch.zeros_like(r_grid)
                        dphi_ij = torch.zeros_like(r_grid)
                    row_phi.append(phi_ij)
                    row_dphi.append(dphi_ij)
                pair_table.append(torch.stack(row_phi))  # [E, n_r]
                pair_deriv_table.append(torch.stack(row_dphi))
            self.r_grid = r_grid
            self.density_table = torch.stack(density_table)            # [E, n_r]
            self.density_deriv_table = torch.stack(density_deriv_table) # [E, n_r]
            self.pair_table = torch.stack(pair_table)                   # [E, E, n_r]
            self.pair_deriv_table = torch.stack(pair_deriv_table)       # [E, E, n_r]
            for i in range(self.E):
                x = torch.from_numpy(self.parser.embedding_splines[i].x).to(self.device, dtype=torch.float32)
                rho_min = float(x[0]); rho_max = float(x[-1])
                n_rho_i = self.n_rho
                drho = (rho_max - rho_min) / (n_rho_i - 1)
                inv_drho = 1.0 / drho
                rho_grid = torch.linspace(rho_min, rho_max, n_rho_i, device=self.device)
                F_i = self._spline_eval(self.parser.embedding_splines[i], rho_grid)
                Fp_i = self._spline_eval(self.parser.embedding_deriv_splines[i], rho_grid)
                self.embed_rho_min[i] = rho_min
                self.embed_drho[i] = drho
                self.embed_inv_drho[i] = inv_drho
                self.embed_n[i] = n_rho_i
                self.embed_table[i] = F_i
                self.embed_deriv_table[i] = Fp_i
            for t in [self.density_table, self.density_deriv_table, self.pair_table, self.pair_deriv_table]:
                assert t.dtype == torch.float32

    def _spline_eval(self, spline_obj, x_tensor: torch.Tensor):
        knots = torch.from_numpy(spline_obj.x).to(self.device, dtype=torch.float32)
        coeffs = torch.from_numpy(spline_obj.c).to(self.device, dtype=torch.float32)
        x = torch.clamp(x_tensor, knots[0], knots[-1])
        idx = torch.searchsorted(knots, x, right=True) - 1
        idx = torch.clamp(idx, 0, len(knots) - 2)
        dx = x - knots[idx]
        if coeffs.shape[0] == 4:  # cubic
            c0, c1, c2, c3 = coeffs[0, idx], coeffs[1, idx], coeffs[2, idx], coeffs[3, idx]
            val = c3 + dx * (c2 + dx * (c1 + dx * c0))
        elif coeffs.shape[0] == 3:  # quadratic
            c0, c1, c2 = coeffs[0, idx], coeffs[1, idx], coeffs[2, idx]
            val = c2 + dx * (c1 + dx * c0)
        elif coeffs.shape[0] == 2:
            c0, c1 = coeffs[0, idx], coeffs[1, idx]
            val = c1 + dx * c0
        else:
            raise RuntimeError("Unsupported spline order")
        return val

    def _maybe_load_extension(self):
        if not self.use_extension:
            self.ext_mod = None
            print("[EAMForceCUDAExt] Using pure PyTorch fallback (extension disabled).")
            return
        try:
            import simulon_cuda
            has_plain = hasattr(simulon_cuda, 'density_pass') and hasattr(simulon_cuda, 'force_pass')
            has_cuda_suffix = hasattr(simulon_cuda, 'density_pass_cuda') and hasattr(simulon_cuda, 'force_pass_cuda')
            if not (has_plain or has_cuda_suffix):
                raise AttributeError("simulon_cuda missing required EAM symbols (density_pass / force_pass)")
            self.ext_mod = simulon_cuda
            self._density_fn_name = 'density_pass_cuda' if has_cuda_suffix else 'density_pass'
            self._force_fn_name = 'force_pass_cuda' if has_cuda_suffix else 'force_pass'
            print(f"[EAMForceCUDAExt] Unified extension 'simulon_cuda' loaded (using {self._density_fn_name}, {self._force_fn_name}).")
        except Exception as e:
            print(f"[EAMForceCUDAExt] Failed to import simulon_cuda, fallback to PyTorch. Reason: {e}")
            self.ext_mod = None
            self.use_extension = False

    def _interp_r(self, r, table):
        # table shape [..., n_r]; r in [0, r_max]
        r_clamped = torch.clamp(r, 0.0, self.r_max * (1 - 1e-7))
        idx_f = r_clamped * self.inv_dr
        idx = idx_f.long()
        frac = idx_f - idx
        next_idx = torch.clamp(idx + 1, max=self.n_r - 1)
        v0 = table[..., idx]
        v1 = table[..., next_idx]
        return v0 + frac * (v1 - v0)

    def _interp_rho(self, rho, i_type, deriv=False):
        rho_min = self.embed_rho_min[i_type]; drho = self.embed_drho[i_type]
        inv_drho = self.embed_inv_drho[i_type]; n = self.embed_n[i_type]
        rho_clamped = torch.clamp(rho, rho_min, rho_min + drho * (n - 1) * (1 - 1e-7))
        idx_f = (rho_clamped - rho_min) * inv_drho
        idx = idx_f.long()
        frac = idx_f - idx
        next_idx = torch.clamp(idx + 1, max=n - 1)
        table = self.embed_deriv_table[i_type] if deriv else self.embed_table[i_type]
        v0 = table[idx]
        v1 = table[next_idx]
        return v0 + frac * (v1 - v0)

    def forward(self):
        coords = self.molecular.coordinates
        edge_index = self.molecular.graph_data.edge_index
        if edge_index.numel() == 0:
            zero = torch.zeros((), device=self.device, dtype=coords.dtype)
            return {'energy': zero, 'forces': torch.zeros_like(coords), 'virial': zero}
        row, col = edge_index
        if getattr(self.molecular.graph_data, 'edge_attr', None) is not None:
            distances_all = self.molecular.graph_data.edge_attr
        else:
            rij = coords[row] - coords[col]
            if hasattr(self.molecular, 'box'):
                rij = self.molecular.box.minimum_image(rij)
            else:
                L = self.molecular.box_length
                rij -= torch.round(rij / L) * L
            distances_all = torch.norm(rij, dim=1)
        mask = distances_all < self.cutoff
        if not mask.any():
            zero = torch.zeros((), device=self.device, dtype=coords.dtype)
            return {'energy': zero, 'forces': torch.zeros_like(coords), 'virial': zero}
        distances = distances_all[mask].contiguous().to(torch.float32)
        row_m = row[mask].contiguous()
        col_m = col[mask].contiguous()
        rij_vec = coords[row_m] - coords[col_m]
        if hasattr(self.molecular, 'box'):
            rij_vec = self.molecular.box.minimum_image(rij_vec)
        else:
            L = self.molecular.box_length
            rij_vec -= torch.round(rij_vec / L) * L
        rij_vec = rij_vec.contiguous().to(torch.float32)
        N = coords.shape[0]
        rho = torch.zeros(N, device=self.device, dtype=torch.float32)
        row_types = self.atom_type_indices[row_m]
        col_types = self.atom_type_indices[col_m]
        if self.use_extension and self.ext_mod is not None:
            getattr(self.ext_mod, self._density_fn_name)(
                distances, row_m, col_m,
                self.atom_type_indices,
                self.density_table, # [E,n_r]
                self.inv_dr, self.n_r,
                rho
            )
        else:
            for atom_type in range(self.E):
                mask_col = (col_types == atom_type)
                if mask_col.any():
                    vals = self._interp_r(distances[mask_col], self.density_table[atom_type])
                    rho.scatter_add_(0, row_m[mask_col], vals)
                mask_row = (row_types == atom_type)
                if mask_row.any():
                    vals = self._interp_r(distances[mask_row], self.density_table[atom_type])
                    rho.scatter_add_(0, col_m[mask_row], vals)
        embedding_energy = torch.zeros_like(rho)
        dF_drho = torch.zeros_like(rho)
        for i_type in range(self.E):
            atom_mask = (self.atom_type_indices == i_type)
            if atom_mask.any():
                rho_i = rho[atom_mask]
                embedding_energy[atom_mask] = self._interp_rho(rho_i, i_type, deriv=False)
                dF_drho[atom_mask] = self._interp_rho(rho_i, i_type, deriv=True)
        total_embedding_energy = embedding_energy.sum()
        pair_energy_edges = torch.zeros_like(distances)
        for i_type in range(self.E):
            for j_type in range(self.E):
                sel = (row_types == i_type) & (col_types == j_type)
                if sel.any():
                    pair_energy_edges[sel] = self._interp_r(distances[sel], self.pair_table[i_type, j_type])
        total_pair_energy = pair_energy_edges.sum()
        total_energy = total_embedding_energy + total_pair_energy
        forces = torch.zeros_like(coords)
        d_density_dr_row = torch.zeros_like(distances)
        d_density_dr_col = torch.zeros_like(distances)
        d_pair_dr = torch.zeros_like(distances)
        for i_type in range(self.E):
            for j_type in range(self.E):
                sel = (row_types == i_type) & (col_types == j_type)
                if sel.any():
                    r_sel = distances[sel]
                    d_density_dr_row[sel] = self._interp_r(r_sel, self.density_deriv_table[i_type])
                    d_density_dr_col[sel] = self._interp_r(r_sel, self.density_deriv_table[j_type])
                    d_pair_dr[sel] = self._interp_r(r_sel, self.pair_deriv_table[i_type, j_type])
        force_scalar = (
            dF_drho[row_m] * d_density_dr_col
            + dF_drho[col_m] * d_density_dr_row
            + d_pair_dr
        )
        if self.use_extension and self.ext_mod is not None:
            getattr(self.ext_mod, self._force_fn_name)(
                distances,
                rij_vec,
                row_m,
                col_m,
                self.atom_type_indices,
                self.density_deriv_table,
                self.pair_deriv_table,
                dF_drho,
                self.inv_dr,
                self.n_r,
                forces
            )
        else:
            force_vec = -force_scalar.unsqueeze(1) * (rij_vec / (distances.unsqueeze(1) + 1e-8))
            forces.scatter_add_(0, row_m.unsqueeze(1).expand_as(force_vec), force_vec)
            forces.scatter_add_(0, col_m.unsqueeze(1).expand_as(force_vec), -force_vec)
        virial = (force_scalar * distances).sum().to(coords.dtype)
        return {
            'energy': total_embedding_energy + total_pair_energy,
            'forces': forces,
            'virial': virial,
            'embedding_energy': total_embedding_energy,
            'pair_potential': total_pair_energy,
        }

EAMForceCUDA = EAMForceCUDAExt


def load_eam_force_cuda():
    return EAMForceCUDAExt
