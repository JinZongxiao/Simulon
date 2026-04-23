import torch
import torch.nn as nn

from core.md_model import BackboneInterface


class LennardJonesForce(BackboneInterface, nn.Module):
    def __init__(self, molecular):
        super(LennardJonesForce, self).__init__()
        self.molecular = molecular
        # 预分配力缓冲区，避免每步 forward 内 torch.zeros_like 分配
        self._forces_buf = torch.zeros_like(self.molecular.coordinates,
                                            device=self.molecular.device)

    def forward(self):
        pos        = self.molecular.graph_data.pos
        edge_index = self.molecular.graph_data.edge_index
        epsilon    = self.molecular.pair_params[0].contiguous()
        sigma      = self.molecular.pair_params[1].contiguous()
        rc        = float(self.molecular.cutoff)
        is_fs     = getattr(self.molecular, 'is_fs', False)
        is_switch = getattr(self.molecular, 'is_switch', False)
        rs        = float(self.molecular.switch_ratio) * rc if is_switch else None

        # 最小镜像：优先使用 Box 对象（支持三斜），回退到 box_length
        if hasattr(self.molecular, 'box'):
            _min_img = self.molecular.box.minimum_image
        else:
            bl = float(self.molecular.box_length)
            def _min_img(r): return r - bl * torch.round(r / bl)

        i = edge_index[0]
        j = edge_index[1]

        rij = _min_img(pos[i] - pos[j])
        r2    = (rij * rij).sum(dim=1).clamp_min(1e-24)
        r     = torch.sqrt(r2)
        inv_r = 1.0 / r
        e_ij  = rij * inv_r.unsqueeze(1)

        # LJ 原始能量与力幅值
        sr    = sigma * inv_r
        sr6   = sr ** 6
        sr12  = sr6 ** 2
        U_raw    = 4.0 * epsilon * (sr12 - sr6)
        Fmag_raw = 24.0 * epsilon * inv_r * (2.0 * sr12 - sr6)

        if is_switch:
            # 平滑切换 S(r) on [rs, rc]：避免两次 clone，改用 torch.where 无分支写法
            x_raw  = (r - rs) / (rc - rs)
            x_c    = x_raw.clamp(0.0, 1.0)
            S_poly  = 1.0 - 10.0*x_c**3 + 15.0*x_c**4 - 6.0*x_c**5
            dS_poly = (-30.0*x_c**2 + 60.0*x_c**3 - 30.0*x_c**4) / (rc - rs)

            # 在 r<=rs 时 S=1, dS=0；r>=rc 时 S=0, dS=0
            in_mid = (r > rs) & (r < rc)
            S  = torch.where(r <= rs, torch.ones_like(r),
                 torch.where(in_mid, S_poly, torch.zeros_like(r)))
            dS = torch.where(in_mid, dS_poly, torch.zeros_like(r))

            # F = -dU_eff/dr = Fmag_raw·S - U_raw·dS/dr  (因 Fmag_raw ≡ -dU_raw/dr)
            U_eff = U_raw * S
            Fmag  = Fmag_raw * S - U_raw * dS

        elif is_fs:
            # Force-shift：在截断处使力连续
            sr_rc   = sigma / rc
            sr6_rc  = sr_rc ** 6
            sr12_rc = sr6_rc ** 2
            U_rc    = 4.0 * epsilon * (sr12_rc - sr6_rc)
            # dU/dr|_rc = 24ε(2σ¹²/rc¹³ - σ⁶/rc⁷)，即 Fmag_rc / rc
            # U_eff(r) = U(r) - U(rc) - (r - rc)·dU/dr|_rc
            Fmag_rc = 24.0 * epsilon * (1.0 / rc) * (2.0 * sr12_rc - sr6_rc)
            U_eff   = U_raw - U_rc + (r - rc) * Fmag_rc   # dU/dr|_rc = -Fmag_rc
            Fmag    = Fmag_raw - Fmag_rc

        else:
            U_eff = U_raw
            Fmag  = Fmag_raw

        # 截断外置零（乘 mask 比 masked_fill 更快）
        mask  = (r < rc).float()
        Fmag  = Fmag  * mask
        U_eff = U_eff * mask

        # 力累加（使用预分配缓冲区，zero_ 比 zeros_like 节省一次内存分配）
        self._forces_buf.zero_()
        fij = Fmag.unsqueeze(1) * e_ij
        self._forces_buf.index_add_(0, i,  fij)
        self._forces_buf.index_add_(0, j, -fij)

        total_energy = U_eff.sum()
        virial = (Fmag * r * mask).sum()   # Σ_edges Fmag·r，用于 NPT 压力
        virial_tensor = torch.einsum('ei,ej->ij', fij, rij)
        return {'energy': total_energy, 'forces': self._forces_buf, 'virial': virial, 'virial_tensor': virial_tensor}
