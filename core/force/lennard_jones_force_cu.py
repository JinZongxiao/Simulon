import torch.nn as nn
import torch
from simulon_cuda import lj_energy_force_cuda as lj_cuda
from core.md_model import BackboneInterface



class LennardJonesForce(BackboneInterface, nn.Module):
    def __init__(self, molecular):
        super(LennardJonesForce, self).__init__()
        self.molecular = molecular
        self.pair_force = torch.empty_like(self.molecular.coordinates, device=self.molecular.device)
        self._cfg_last = None
        self._cfg_warned = False

    def forward(self):
        pos = self.molecular.graph_data.pos
        edge_index_full = self.molecular.graph_data.edge_index
        epsilon_full = self.molecular.pair_params[0].contiguous()
        sigma_full = self.molecular.pair_params[1].contiguous()
        # Runtime cutoff filtering (Verlet list constructed with cutoff+skin)
        mask = self.molecular.get_cutoff_mask()
        if mask.shape[0] == edge_index_full.shape[1] and not mask.all():
            edge_index = edge_index_full[:, mask]
            epsilon = epsilon_full[mask]
            sigma = sigma_full[mask]
        else:
            edge_index = edge_index_full
            epsilon = epsilon_full
            sigma = sigma_full
        if edge_index is None or edge_index.numel() == 0 or edge_index.shape[1] == 0:
            zero_energy = torch.zeros((), device=pos.device, dtype=pos.dtype)
            zero_forces = torch.zeros_like(pos)
            zero_virial = torch.zeros((), device=pos.device, dtype=pos.dtype)
            zero_virial_tensor = torch.zeros((3, 3), device=pos.device, dtype=pos.dtype)
            return {'energy': zero_energy, 'forces': zero_forces, 'virial': zero_virial, 'virial_tensor': zero_virial_tensor}
        rc_t = getattr(self.molecular, 'cutoff', None)
        rc = float(rc_t.item()) if torch.is_tensor(rc_t) else float(rc_t)
        is_switch = getattr(self.molecular, 'is_switch', False)
        is_fs = getattr(self.molecular, 'is_fs', False)
        if is_switch:
            mode = 2
            rs = float(getattr(self.molecular, 'switch_ratio', 0.9)) * rc
        elif is_fs:
            mode = 1
            rs = 0.0
        else:
            mode = 0
            rs = 0.0
        curr_cfg = (int(mode), float(rc), float(rs))
        if curr_cfg != self._cfg_last:
            try:
                from simulon_cuda import configure_lj_smoothing
                configure_lj_smoothing(*curr_cfg)
                print(f"[LJ CUDA] smoothing configured: mode={mode}, rc={rc:.3f}, rs={rs:.3f}")
                self._cfg_last = curr_cfg
            except Exception as e:
                if not self._cfg_warned:
                    print(f"[warn] configure_lj_smoothing not available: {e}")
                    self._cfg_warned = True
        dtype0 = pos.dtype
        pos32 = pos.contiguous().float()
        edge_index = edge_index.contiguous()
        epsilon32 = epsilon.contiguous().float()
        sigma32 = sigma.contiguous().float()
        use_box = getattr(self.molecular, 'box', None)
        if use_box is not None and not use_box.is_orthogonal:
            raise NotImplementedError(
                "LJ CUDA kernel currently supports only orthogonal boxes; "
                "triclinic boxes should use the PyTorch LJ implementation."
            )
        box_len_t = getattr(self.molecular, 'box_length', None)
        box_length = float(box_len_t.item()) if torch.is_tensor(box_len_t) else float(box_len_t)
        energy, pair_force32 = lj_cuda(pos32, edge_index, sigma32, epsilon32, float(box_length))
        total_energy = energy[0].to(dtype0)
        self.pair_force = pair_force32.to(dtype0)

        i = edge_index[0]
        j = edge_index[1]
        rij = pos[i] - pos[j]
        if use_box is not None:
            rij = use_box.minimum_image(rij)
        else:
            rij = rij - box_length * torch.round(rij / box_length)
        r2 = (rij * rij).sum(dim=1).clamp_min(1e-24)
        r = torch.sqrt(r2)
        inv_r = 1.0 / r

        sigma_d = sigma.to(dtype0)
        epsilon_d = epsilon.to(dtype0)
        sr = sigma_d * inv_r
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        u_raw = 4.0 * epsilon_d * (sr12 - sr6)
        fmag_raw = 24.0 * epsilon_d * inv_r * (2.0 * sr12 - sr6)

        if is_switch:
            x = ((r - rs) / (rc - rs)).clamp(0.0, 1.0)
            s_mid = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
            ds_mid = (-30.0 * x**2 + 60.0 * x**3 - 30.0 * x**4) / (rc - rs)
            in_mid = (r > rs) & (r < rc)
            s = torch.where(r <= rs, torch.ones_like(r),
                torch.where(in_mid, s_mid, torch.zeros_like(r)))
            ds = torch.where(in_mid, ds_mid, torch.zeros_like(r))
            fmag = fmag_raw * s - u_raw * ds
        elif is_fs:
            sr_rc = sigma_d / rc
            sr6_rc = sr_rc ** 6
            sr12_rc = sr6_rc ** 2
            fmag_rc = 24.0 * epsilon_d * (1.0 / rc) * (2.0 * sr12_rc - sr6_rc)
            fmag = fmag_raw - fmag_rc
        else:
            fmag = fmag_raw

        virial = (fmag * r * (r < rc).to(dtype0)).sum()
        fij = fmag.unsqueeze(1) * (rij / r.unsqueeze(1))
        virial_tensor = torch.einsum('ei,ej->ij', fij, rij)
        return {'energy': total_energy, 'forces': self.pair_force, 'virial': virial, 'virial_tensor': virial_tensor}
