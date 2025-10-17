import torch.nn as nn
import torch

from core.md_model import BackboneInterface

class LennardJonesForce(BackboneInterface, nn.Module):
    def __init__(self, molecular):
        super(LennardJonesForce, self).__init__()
        self.molecular = molecular
        self.pair_force = torch.empty_like(self.molecular.coordinates, device=self.molecular.device)

    def forward(self):
        pos = self.molecular.graph_data.pos
        edge_index = self.molecular.graph_data.edge_index
        epsilon = self.molecular.pair_params[0].contiguous()
        sigma = self.molecular.pair_params[1].contiguous()
        box_length = float(self.molecular.box_length)
        rc = float(self.molecular.cutoff)
        is_fs = getattr(self.molecular, 'is_fs', False)
        is_switch = getattr(self.molecular, 'is_switch', False)
        rs = float(self.molecular.switch_ratio) * rc if is_switch else None

        # Build pair vectors with minimum image convention
        i = edge_index[0]
        j = edge_index[1]
        rij = pos[i] - pos[j]
        rij = rij - box_length * torch.round(rij / box_length)
        r2 = (rij * rij).sum(dim=1).clamp_min(1e-24)
        r = torch.sqrt(r2)
        inv_r = 1.0 / r
        e_ij = rij * inv_r.unsqueeze(1)

        # LJ raw energy and force magnitude
        sr = sigma * inv_r  # sigma/r
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        U_raw = 4.0 * epsilon * (sr12 - sr6)
        Fmag_raw = 24.0 * epsilon * inv_r * (2.0 * sr12 - sr6)  # magnitude along e_ij

        if is_switch:
            # Smooth switch S(r) on [rs, rc]
            one = torch.ones_like(r)
            zero = torch.zeros_like(r)
            S = torch.where(r <= rs, one, torch.where(r >= rc, zero, one))
            dS = torch.zeros_like(r)
            mid = (r > rs) & (r < rc)
            if mid.any():
                x = (r[mid] - rs) / (rc - rs)
                S_mid = 1 - 10*x**3 + 15*x**4 - 6*x**5
                dS_mid = (-30*x**2 + 60*x**3 - 30*x**4) / (rc - rs)
                S = S.clone(); dS = dS.clone()
                S[mid] = S_mid
                dS[mid] = dS_mid
            U_eff = U_raw * S
            # F = S*F_raw + U_raw * dS/dr in radial direction
            Fmag = Fmag_raw * S + U_raw * dS
        elif is_fs:
            # Force-shift: Ufs = U - U(rc) - (r-rc)U'(rc); Ffs = F_raw - F(rc)
            sr_rc = sigma / rc
            sr6_rc = sr_rc ** 6
            sr12_rc = sr6_rc ** 2
            U_rc = 4.0 * epsilon * (sr12_rc - sr6_rc)
            dUdr_rc = 24.0 * epsilon * (-2.0 * (sigma ** 12) / (rc ** 13) + (sigma ** 6) / (rc ** 7))
            U_eff = U_raw - U_rc - (r - rc) * dUdr_rc
            Fmag_rc = 24.0 * epsilon * (1.0/rc) * (2.0 * sr12_rc - sr6_rc)
            Fmag = Fmag_raw - Fmag_rc
        else:
            U_eff = U_raw
            Fmag = Fmag_raw

        # Zero out beyond cutoff for safety
        mask = r < rc
        Fmag = Fmag * mask
        U_eff = U_eff * mask

        # Accumulate atomic forces
        forces = torch.zeros_like(pos)
        fij = Fmag.unsqueeze(1) * e_ij
        forces.index_add_(0, i, fij)
        forces.index_add_(0, j, -fij)

        total_energy = U_eff.sum()
        self.pair_force = forces
        return {'energy': total_energy, 'forces': self.pair_force}