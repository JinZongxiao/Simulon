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

            # ----------------------------------------------------------------
            # 堆叠张量：将dict-of-tensors转换为连续的多维张量，
            # 使 forward 中的 for 循环全部替换为单次 gather 操作。
            # ----------------------------------------------------------------
            E = len(self.parser.elements)

            # density / pair: [E, n_r]
            self.density_table_stack = torch.stack(
                [self.density_table[i] for i in range(E)])
            self.density_deriv_table_stack = torch.stack(
                [self.density_deriv_table[i] for i in range(E)])

            # pair potential: [E, E, n_r]  (缺少的对填0)
            pair_rows, pair_deriv_rows = [], []
            for i in range(E):
                row, row_d = [], []
                for j in range(E):
                    if j in self.pair_table[i]:
                        row.append(self.pair_table[i][j])
                        row_d.append(self.pair_deriv_table[i][j])
                    else:
                        row.append(torch.zeros(self.n_r, device=self.device))
                        row_d.append(torch.zeros(self.n_r, device=self.device))
                pair_rows.append(torch.stack(row))
                pair_deriv_rows.append(torch.stack(row_d))
            self.pair_table_stack = torch.stack(pair_rows)        # [E, E, n_r]
            self.pair_deriv_table_stack = torch.stack(pair_deriv_rows)

            # embedding: [E, n_rho]（所有元素共享同一 n_rho）
            self.embed_table_stack = torch.stack(
                [self.embed_table[i] for i in range(E)])
            self.embed_deriv_table_stack = torch.stack(
                [self.embed_deriv_table[i] for i in range(E)])

            # 每种元素的 rho 网格参数（标量→张量，供向量化查表）
            self.embed_rho_min_t = torch.tensor(
                [self.embed_rho_min[i] for i in range(E)],
                device=self.device, dtype=torch.float32)
            self.embed_inv_drho_t = torch.tensor(
                [self.embed_inv_drho[i] for i in range(E)],
                device=self.device, dtype=torch.float32)

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

    # ------------------------------------------------------------------
    # 向量化辅助：一次对所有原子做 embedding 查表（无element-type循环）
    # ------------------------------------------------------------------
    def _interp_embed_all(self, rho):
        """
        rho: [N] float32
        返回 (embed_E [N], embed_dE [N])
        利用 embed_table_stack [E, n_rho] 和 embed_deriv_table_stack [E, n_rho]
        通过 atom_type_indices 一次 gather，替代原来的 for i_type 循环。
        """
        atom_types = self.atom_type_indices            # [N] long
        rho_min = self.embed_rho_min_t[atom_types]     # [N]
        inv_drho = self.embed_inv_drho_t[atom_types]   # [N]

        # clamp 到各元素的有效 rho 范围
        rho_max_idx = self.n_rho - 1
        rho_clamped = torch.clamp(
            rho,
            rho_min,
            rho_min + rho_max_idx / inv_drho * (1.0 - 1e-7)
        )
        idx_f = (rho_clamped - rho_min) * inv_drho
        idx = idx_f.long().clamp(0, rho_max_idx - 1)
        frac = idx_f - idx.float()
        next_idx = (idx + 1).clamp(max=rho_max_idx)

        # 双线性 gather：[N]
        e0 = self.embed_table_stack[atom_types, idx]
        e1 = self.embed_table_stack[atom_types, next_idx]
        d0 = self.embed_deriv_table_stack[atom_types, idx]
        d1 = self.embed_deriv_table_stack[atom_types, next_idx]

        embed_E  = e0 + frac * (e1 - e0)
        embed_dE = d0 + frac * (d1 - d0)
        return embed_E, embed_dE

    def forward(self):
        return self._forward_table_fast()

    def _forward_table_fast(self):
        """
        向量化版 EAM forward。

        原版 _forward_table 中存在 O(E²) 个 Python for 循环，每次都创建 bool
        mask 并做小批量 scatter，在 GPU 上效率极低。

        优化策略：
          - 密度/对势查表：利用 density_table_stack[E, n_r] 和
            pair_table_stack[E, E, n_r]，通过 col_types / (row_types, col_types)
            直接 gather，整个 edge list 一次完成，无 Python 循环。
          - Embedding 查表：利用 _interp_embed_all，对全部 N 个原子一次完成。
          - 所有中间张量保持在 GPU 上，避免 CPU-GPU 同步。
        """
        coords = self.molecular.coordinates
        edge_index = self.molecular.graph_data.edge_index
        distances_all = self.molecular.graph_data.edge_attr
        num_atoms = coords.shape[0]
        row, col = edge_index

        in_cutoff = distances_all < self.cutoff
        if not in_cutoff.any():
            return {
                'energy': torch.tensor(0.0, device=self.device),
                'forces': torch.zeros_like(coords),
                'virial': torch.tensor(0.0, device=self.device),
            }

        distances = distances_all[in_cutoff]
        row_c = row[in_cutoff]
        col_c = col[in_cutoff]
        # Box-aware minimum image（支持正交与三斜盒子）
        if hasattr(self.molecular, 'box'):
            rij_vec = self.molecular.box.minimum_image(coords[row_c] - coords[col_c])
        else:
            bl = self.molecular.box_length
            rij_vec = coords[row_c] - coords[col_c]
            rij_vec = rij_vec - torch.round(rij_vec / bl) * bl

        row_types = self.atom_type_indices[row_c]  # [E_edges]
        col_types = self.atom_type_indices[col_c]  # [E_edges]

        # ── 共用线性查表索引（对 r 网格，所有元素共享） ──────────────────
        r_cl = distances.clamp(0.0, self.r_max * (1.0 - 1e-7))
        idx_f = r_cl * self.inv_dr
        idx   = idx_f.long()
        frac  = idx_f - idx.float()
        nidx  = (idx + 1).clamp(max=self.n_r - 1)

        # ── Phase 1: 电子密度累加 ────────────────────────────────────────
        # density_table_stack[col_type, idx]: j 原子贡献给 i 的密度
        f0 = self.density_table_stack[col_types, idx]
        f1 = self.density_table_stack[col_types, nidx]
        dens_pairs = f0 + frac * (f1 - f0)

        rho = torch.zeros(num_atoms, device=self.device, dtype=distances.dtype)
        rho.scatter_add_(0, row_c, dens_pairs)

        # ── Phase 2: Embedding 能量与 dF/drho ───────────────────────────
        embed_E, dF_drho = self._interp_embed_all(rho)
        total_embedding_energy = embed_E.sum()

        # ── Phase 3: 对势能 ──────────────────────────────────────────────
        # pair_table_stack[row_type, col_type, idx]
        p0 = self.pair_table_stack[row_types, col_types, idx]
        p1 = self.pair_table_stack[row_types, col_types, nidx]
        pair_vals = p0 + frac * (p1 - p0)
        total_pair_potential = 0.5 * pair_vals.sum()

        total_energy = total_embedding_energy + total_pair_potential

        # ── Phase 4: 力 ─────────────────────────────────────────────────
        # df/dr (密度导数): 按 col_type 查表
        fd0 = self.density_deriv_table_stack[col_types, idx]
        fd1 = self.density_deriv_table_stack[col_types, nidx]
        d_density_dr = fd0 + frac * (fd1 - fd0)

        # dphi/dr (对势导数): 按 (row_type, col_type) 查表
        pd0 = self.pair_deriv_table_stack[row_types, col_types, idx]
        pd1 = self.pair_deriv_table_stack[row_types, col_types, nidx]
        d_pair_dr = pd0 + frac * (pd1 - pd0)

        dF_sum = dF_drho[row_c] + dF_drho[col_c]
        force_scalar = dF_sum * d_density_dr + d_pair_dr
        unit_vec = rij_vec / (distances.unsqueeze(1) + 1e-8)
        force_vec = -force_scalar.unsqueeze(1) * unit_vec

        forces = torch.zeros_like(coords)
        forces.scatter_add_(0, row_c.unsqueeze(1).expand_as(force_vec),  force_vec)
        forces.scatter_add_(0, col_c.unsqueeze(1).expand_as(force_vec), -force_vec)

        # 维里（用于 NPT 压力）：W = Σ_edges force_scalar * r
        virial = (force_scalar * distances).sum()

        return {'energy': total_energy, 'forces': forces, 'virial': virial}
