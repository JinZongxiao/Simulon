"""
Born-Mayer-Huggins 力场（边列表版本）

原版问题：
  - 构建完整的 N×N 距离矩阵 → O(N²) 内存，N>3000 时 GPU OOM
  - 通过 autograd 反向传播求力 → 保留计算图，速度慢
  - forward() 签名与 BackboneInterface 不一致（含多余参数）

优化后：
  - 基于已有邻居列表（graph_data.edge_index）的边列表计算
  - 解析求力公式，无 autograd
  - O(E) 内存（E = 边数，通常 << N²）
"""
import torch
import torch.nn as nn

from core.md_model import BackboneInterface
from core.parameter_manager import ElementParameterManager


class BornMayerHugginsForce(BackboneInterface, nn.Module):

    def __init__(self, molecular, parameters):
        super(BornMayerHugginsForce, self).__init__()
        self.molecular = molecular

        # 用 ElementParameterManager 预计算每个元素对类型的参数向量
        element_list = list(dict.fromkeys(molecular.atom_types))  # 去重保序
        self._param_mgr = ElementParameterManager(
            element_list=element_list,
            parameter_dict=parameters
        )

        # 记录元素种类数及原子→类型索引
        self._n_types = len(element_list)
        elem_to_idx = {e: i for i, e in enumerate(element_list)}
        type_indices = [elem_to_idx[t] for t in molecular.atom_types]
        self._atom_type_idx = torch.tensor(
            type_indices, dtype=torch.long, device=molecular.device
        )

        # 将每个有序对（i_type, j_type）的参数预取到张量
        # torch_pair_parameters 已按 itertools.product(element_list, repeat=2) 排列
        pm = self._param_mgr.torch_pair_parameters
        n = self._n_types
        self._A   = pm['A'].reshape(n, n)    if 'A'   in pm else None
        self._C   = pm['C'].reshape(n, n)    if 'C'   in pm else None
        self._D   = pm['D'].reshape(n, n)    if 'D'   in pm else None
        self._rho = pm['rho'].reshape(n, n)  if 'rho' in pm else None
        self._sig = pm['sigma'].reshape(n, n)if 'sigma' in pm else None

        # 预分配力缓冲区
        self._forces_buf = torch.zeros_like(
            molecular.coordinates, device=molecular.device
        )

    def forward(self):
        mol        = self.molecular
        pos        = mol.coordinates
        edge_index = mol.graph_data.edge_index
        rc         = float(mol.cutoff)

        # 使用 Box 对象做最小镜像（支持斜方/三斜盒子）
        if hasattr(mol, 'box'):
            min_image = mol.box.minimum_image
        else:
            bl = float(mol.box_length)
            def min_image(r): return r - bl * torch.round(r / bl)

        i = edge_index[0]
        j = edge_index[1]
        rij = min_image(pos[i] - pos[j])                # [E, 3]
        r   = torch.norm(rij, dim=1).clamp(min=1e-6)   # [E]

        # 截断掩码
        mask = (r < rc).float()

        # 按 (i_type, j_type) 查表，得到每条边的参数 [E]
        it = self._atom_type_idx[i]  # [E]
        jt = self._atom_type_idx[j]  # [E]

        A   = self._A  [it, jt] if self._A   is not None else r.new_zeros(r.shape)
        C   = self._C  [it, jt] if self._C   is not None else r.new_zeros(r.shape)
        D   = self._D  [it, jt] if self._D   is not None else r.new_zeros(r.shape)
        rho = self._rho[it, jt] if self._rho is not None else r.new_ones(r.shape)
        sig = self._sig[it, jt] if self._sig is not None else r.new_zeros(r.shape)

        inv_r  = 1.0 / r
        inv_r6 = inv_r ** 6
        inv_r8 = inv_r6 * inv_r ** 2

        # --- 能量 ---
        U_pair = A * torch.exp((sig - r) / rho) - C * inv_r6 + D * inv_r8
        total_energy = 0.5 * (U_pair * mask).sum()

        # --- 解析力（对 r 求导后沿 rij 方向） ---
        # dU/dr = -A/rho * exp(...) + 6C/r^7 - 8D/r^9
        dU_dr = (
            -A / rho * torch.exp((sig - r) / rho)
            + 6.0 * C * inv_r6 * inv_r
            - 8.0 * D * inv_r8 * inv_r
        )
        # F_i = -dU/dr * (rij/r)；F_j = -F_i
        Fmag = -dU_dr * mask              # [E]
        fij  = Fmag.unsqueeze(1) * (rij * inv_r.unsqueeze(1))  # [E, 3]

        # 力散射到原子
        self._forces_buf.zero_()
        self._forces_buf.index_add_(0, i,  fij)
        self._forces_buf.index_add_(0, j, -fij)

        # 维里（用于 NPT 压力计算）
        virial = (Fmag * r).sum()

        return {
            'energy': total_energy,
            'forces': self._forces_buf,
            'virial': virial,
        }
