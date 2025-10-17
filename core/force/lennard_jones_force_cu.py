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
        box_len_t = getattr(self.molecular, 'box_length', None)
        box_length = float(box_len_t.item()) if torch.is_tensor(box_len_t) else float(box_len_t)
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
            return {'energy': zero_energy, 'forces': zero_forces}
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
        energy, pair_force32 = lj_cuda(pos32, edge_index, sigma32, epsilon32, float(box_length))
        total_energy = energy[0].to(dtype0)
        self.pair_force = pair_force32.to(dtype0)
        
            
        return {'energy': total_energy, 'forces': self.pair_force}