import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from typing import Dict, List

main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BackboneInterface(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self) -> Dict[str, torch.Tensor]:
        pass

class IntegratorInterface(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, force: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

class SumBackboneInterface(nn.Module):
    def __init__(self, backbones,molecular):
        super(SumBackboneInterface, self).__init__()
        self.backbones = nn.ModuleList(backbones)
        self.molecular = molecular
        self.device = self.molecular.device
        self.atom_num = self.molecular.atom_count

    def forward(self):
        total_forces = torch.zeros(self.atom_num, 3, device=self.device)
        total_energy = torch.tensor(0.0, device=self.device)
        for backbone in self.backbones:
            output = backbone()
            total_forces.add_(output['forces'])
            total_energy.add_(output['energy'])
        return {'forces': total_forces,'energy': total_energy}


class BaseModel(nn.Module):
    def __init__(self, sum_bone, integrator: IntegratorInterface, molecular):
        super(BaseModel, self).__init__()
        self.sum_bone = sum_bone
        self.Integrator = integrator
        self.molecular = molecular
        self.force_cache = torch.empty_like(molecular.coordinates, device=main_device)
        self.energy_cache = torch.empty(1, device=main_device)
        self._first_call = True
        self.profile = {
            'pair_time': 0.0,
            'integrate_time': 0.0,
            'pair_calls': 0,
            'integrate_calls': 0,
        }
        if torch.cuda.is_available():
            self._evt_first_start = torch.cuda.Event(enable_timing=True)
            self._evt_first_end = torch.cuda.Event(enable_timing=True)
            self._evt_pair_start = torch.cuda.Event(enable_timing=True)
            self._evt_pair_end = torch.cuda.Event(enable_timing=True)
            self._evt_second_end = torch.cuda.Event(enable_timing=True)
        else:
            self._evt_first_start = None

    def forward(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self._first_call:
            if torch.cuda.is_available():
                self._evt_pair_start.record()
                out0 = self.sum_bone()
                self._evt_pair_end.record()
                torch.cuda.synchronize()
                self.profile['pair_time'] += self._evt_pair_start.elapsed_time(self._evt_pair_end)/1000.0
            else:
                import time; t0=time.perf_counter(); out0=self.sum_bone(); t1=time.perf_counter(); self.profile['pair_time'] += (t1-t0)
            self.force_cache = out0['forces']
            self.energy_cache = out0['energy']
            self._first_call = False
        if torch.cuda.is_available():
            self._evt_first_start.record()
        else:
            import time; t_first0 = time.perf_counter()
        self.Integrator.first_half(self.force_cache)
        if torch.cuda.is_available():
            self._evt_first_end.record()
        else:
            import time; t_first1 = time.perf_counter()
        if torch.cuda.is_available():
            self._evt_pair_start.record()
            out = self.sum_bone()
            self._evt_pair_end.record()
            torch.cuda.synchronize()
            pair_ms = self._evt_pair_start.elapsed_time(self._evt_pair_end)
            self.profile['pair_time'] += pair_ms/1000.0
        else:
            import time; t_pair0=time.perf_counter(); out=self.sum_bone(); t_pair1=time.perf_counter(); self.profile['pair_time'] += (t_pair1-t_pair0)
        self.force_cache = out['forces']
        self.energy_cache = out['energy']
        if torch.cuda.is_available():
            self.Integrator.second_half(self.force_cache)
            self._evt_second_end.record()
            torch.cuda.synchronize()
            first_ms = self._evt_first_start.elapsed_time(self._evt_first_end)
            second_ms = self._evt_pair_end.elapsed_time(self._evt_second_end)
            self.profile['integrate_time'] += (first_ms + second_ms)/1000.0
            integ_out = {'update_coordinates': self.Integrator.new_coords,
                         'kinetic_energy': (0.5 * self.Integrator.atom_mass * self.molecular.atom_velocities.pow(2)).sum(),
                         'temperature': (2.0/3.0) * (0.5 * self.Integrator.atom_mass * self.molecular.atom_velocities.pow(2)).sum() / (self.molecular.atom_count * self.Integrator.BOLTZMAN)}
        else:
            import time; t_second0=time.perf_counter(); integ_out = self.Integrator.second_half(self.force_cache); t_second1=time.perf_counter(); self.profile['integrate_time'] += (t_first1 - t_first0) + (t_second1 - t_second0)
        self.profile['pair_calls'] += 1
        self.profile['integrate_calls'] += 1
        return {
            'forces': self.force_cache,
            'energy': self.energy_cache,
            'updated_coordinates': integ_out['update_coordinates'],
            'kinetic_energy': integ_out['kinetic_energy'],
            'temperature': integ_out['temperature']
        }
