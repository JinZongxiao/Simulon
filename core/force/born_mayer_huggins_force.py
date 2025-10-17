import torch.nn as nn
import torch
from core.md_model import BackboneInterface
from core.parameter_manager import ElementParameterManager


class BornMayerHugginsForce(BackboneInterface, nn.Module):
    def __init__(self, molecular, parameters):
        super(BornMayerHugginsForce, self).__init__()
        self.molecular = molecular
        self.parameters = parameters
        self.parameter_manager = ElementParameterManager(
            element_list=self.molecular.atom_set,
            parameter_dict=parameters
        )

    def forward(self, atom_coordinates=None, cutoff=10, min_distance=1):
        molecular = self.molecular
        atom_types = molecular.atom_types
        box_length = torch.tensor(molecular.box_length,device=self.molecular.device)

        parameters = self.parameter_manager.get_parameters_for_pairs(atom_types)

        if atom_coordinates is not None:
            atom_coordinates = atom_coordinates
        else:
            atom_coordinates = molecular.get_atom_coordinates()
        atom_coordinates.requires_grad = True

        rij = atom_coordinates.unsqueeze(1) - atom_coordinates.unsqueeze(0)
        rij = rij - torch.round(rij / box_length) * box_length
        rij2 = torch.sum(rij  ** 2, dim = 2)
        rij = torch.sqrt(torch.clamp(rij2, min=min_distance ** 2))
        mask = (rij < cutoff) & (~torch.eye(rij.size(0), dtype=torch.bool, device=rij.device))

        A = parameters['A']
        C = parameters['C']
        D = parameters['D']
        rho = parameters['rho']
        sigma = parameters['sigma']

        pair_energy = A * torch.exp((sigma - rij) / rho) - C / (rij ** 6) + D / (rij ** 8)
        pair_energy = pair_energy.masked_fill(~mask, 0)
        total_energy = (pair_energy.sum()) / 2
        atom_force = -torch.autograd.grad(total_energy, atom_coordinates, create_graph=False,retain_graph=False)[0]

        del rij, rij2, mask, pair_energy,A ,C ,D ,rho, sigma
        
        return {'energy': total_energy, 'forces': atom_force}
