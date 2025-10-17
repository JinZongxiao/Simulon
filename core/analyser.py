import math
import torch
from pathlib import Path
from typing import Optional, Sequence

class RDFAccumulator:

    def __init__(self,
                 molecular,
                 nbins: int,
                 cutoff: float,
                 nevery: int,
                 nrepeat: int,
                 outfile: str,
                 type_pair: Optional[Sequence[int]] = None):
        # ...existing code before _h_acc...
        self.mol = molecular
        self.device = molecular.device if hasattr(molecular, 'device') else torch.device('cpu')
        self.nbins = int(nbins)
        self.cutoff = float(cutoff)
        self.nevery = int(nevery)
        self.nrepeat = int(nrepeat)
        self.outfile = Path(outfile)
        self.type_pair = type_pair
        self.dr = self.cutoff / self.nbins
        self._h_acc = torch.zeros(self.nbins, dtype=torch.float64, device=self.device)
        self._samples = 0
        L = float(molecular.box_length) if torch.is_tensor(molecular.box_length) else float(molecular.box_length)
        self.volume = L**3
        self.box_length = L
        self._write_header()

    def _write_header(self):
        if not self.outfile.parent.exists():
            self.outfile.parent.mkdir(parents=True, exist_ok=True)
        with open(self.outfile, 'w', encoding='utf-8') as f:
            f.write(f"# RDF output (LAMMPS-like average blocks)\n")
            f.write(f"# nbins={self.nbins} cutoff={self.cutoff} nevery={self.nevery} nrepeat={self.nrepeat}\n")
            f.write("# Columns: r  g(r)  coord(r)\n")

    def _select_indices(self):
        if not hasattr(self.mol, 'element_ids'):
            return None, None
        if self.type_pair is None:
            return None, None
        t1, t2 = self.type_pair
        elem_ids = self.mol.element_ids
        t1 -= 1; t2 -= 1
        mask1 = (elem_ids == t1)
        mask2 = (elem_ids == t2)
        return mask1, mask2

    def _compute_instant_hist(self):
        edge_index = self.mol.graph_data.edge_index
        d_all = self.mol.graph_data.edge_attr
        if edge_index is None or d_all is None or edge_index.numel()==0:
            return torch.zeros(self.nbins, dtype=torch.float64, device=self.device)
        mask_c = d_all <= self.cutoff
        if mask_c.sum()==0:
            return torch.zeros(self.nbins, dtype=torch.float64, device=self.device)
        d = d_all[mask_c]
        mask1, mask2 = self._select_indices()
        if mask1 is not None and mask2 is not None:
            i = edge_index[0][mask_c]
            j = edge_index[1][mask_c]
            pair_mask = (mask1[i] & mask2[j]) | (mask2[i] & mask1[j])
            if pair_mask.sum()==0:
                return torch.zeros(self.nbins, dtype=torch.float64, device=self.device)
            d = d[pair_mask]
        bin_idx = torch.clamp((d / self.dr).long(), max=self.nbins-1)
        hist = torch.bincount(bin_idx, minlength=self.nbins)
        hist = hist.to(dtype=torch.float64, device=self.device)
        return hist

    def _normalize(self, h: torch.Tensor):
        N = self.mol.atom_count
        r_centers = (torch.arange(self.nbins, dtype=torch.float64, device=self.device) + 0.5) * self.dr
        shell = 4.0 * math.pi * (r_centers**2) * self.dr
        if self.type_pair is None:
            denom = (N * N) * shell
            g = torch.zeros_like(r_centers)
            valid = shell > 0
            g[valid] = (2.0 * self.volume * h[valid]) / denom[valid]
        else:
            t1_mask, t2_mask = self._select_indices()
            N1 = int(t1_mask.sum().item())
            N2 = int(t2_mask.sum().item())
            denom = (N1 * N2) * shell / self.volume
            g = torch.zeros_like(r_centers)
            valid = (denom > 0)
            g[valid] = h[valid] / denom[valid]
        rho_all = N / self.volume
        coord = torch.cumsum(g * shell * rho_all, dim=0)
        return r_centers, g, coord

    def update(self, step: int, molecular=None):
        if molecular is not None:
            self.mol = molecular
        if ((step + 1) % self.nevery) != 0:
            return
        h = self._compute_instant_hist()
        self._h_acc += h
        self._samples += 1
        if self._samples >= self.nrepeat:
            avg_h = self._h_acc / self._samples
            r, g, coord = self._normalize(avg_h)
            self._write_block(r, g, coord)
            # reset for next window
            self._h_acc.zero_()
            self._samples = 0

    def _write_block(self, r, g, coord):
        r_cpu = r.detach().cpu().tolist()
        g_cpu = g.detach().cpu().tolist()
        coord_cpu = coord.detach().cpu().tolist()
        with open(self.outfile, 'a', encoding='utf-8') as f:
            f.write(f"# ---- block average ----\n")
            for ri, gi, ci in zip(r_cpu, g_cpu, coord_cpu):
                f.write(f"{ri:.6f} {gi:.8f} {ci:.8f}\n")
            f.write("\n")

    def finalize(self):
        if self._samples > 0:
            avg_h = self._h_acc / self._samples
            r, g, coord = self._normalize(avg_h)
            self._write_block(r, g, coord)
            self._h_acc.zero_(); self._samples = 0

__all__ = ['RDFAccumulator']
