import torch.nn as nn
import torch
from core.md_model import BackboneInterface
from core.parameter_manager import ElementParameterManager
from typing import Dict, Tuple


class BornMayerHugginsForce(BackboneInterface, nn.Module):

    def __init__(self, molecular, parameters, block_size=512):
        super().__init__()
        self.molecular = molecular
        self.parameters = parameters
        self.block_size = block_size
        self.box_length = molecular.box_length
        self.parameter_manager = ElementParameterManager(
            element_list=self.molecular.atom_set,
            parameter_dict=parameters
        )

    @staticmethod
    def _compute_block_interaction(coords_i, coords_j, q_i, q_j, params, cutoff, box_length, min_distance=1):
        qiqj = q_i.unsqueeze(0) * q_j.unsqueeze(1)
        rij = coords_i.unsqueeze(1) - coords_j.unsqueeze(0)
        rij = rij - torch.round(rij / box_length) * box_length
        rij2 = torch.sum(rij ** 2, dim=-1)
        dij = torch.sqrt(torch.clamp(rij2, min=min_distance))

        mask = (dij < cutoff) & (~torch.eye(dij.size(0), dtype=torch.bool, device=dij.device))
        if coords_i is coords_j:
            mask = torch.triu(mask, diagonal=1)

        valid_indices = torch.where(mask)
        dij_valid = dij[valid_indices]
        qiqj_valid = qiqj[valid_indices]

        A = params['A'][valid_indices]
        C = params['C'][valid_indices]
        D = params['D'][valid_indices]
        rho = params['rho'][valid_indices]
        sigma = params['sigma'][valid_indices]

        pair_energy = (
                #(qiqj_valid / dij_valid)
                + A * torch.exp((sigma - dij_valid) / rho)
                - C / dij_valid ** 6
                + D / dij_valid ** 8
        )
        return pair_energy.sum(), rij[valid_indices[0], valid_indices[1]], pair_energy

    def forward(self, atom_coordinates=None, cutoff=17.5):
        molecular = self.molecular
        atom_types = molecular.atom_types

        atom_coords = atom_coordinates if atom_coordinates is not None \
            else molecular.get_atom_coordinates()
        atom_coords.requires_grad_(True)
        qiqj = molecular.atom_iron_num

        full_params = self.parameter_manager.get_parameters_for_pairs(atom_types)

        block_size = self.block_size
        blocks = torch.split(atom_coords, block_size, dim=0)
        num_blocks = len(blocks)

        blocks_iron_num = torch.split(qiqj, block_size, dim=0)

        total_energy = torch.tensor(0.0, device=atom_coords.device)
        atom_coords.grad = torch.zeros_like(atom_coords)

        for i in range(num_blocks):
            for j in range(i, num_blocks):
                block_params = {}
                for param_name in ['A', 'C', 'D', 'rho', 'sigma']:
                    param = full_params[param_name]

                    row_split = torch.split(param, block_size, dim=0)[i]  # [block_size, N]
                    block = torch.split(row_split, block_size, dim=1)[j]  # [block_size, block_size]
                    block_params[param_name] = block
                energy, _, pair_energy = self._compute_block_interaction(
                    blocks[i],
                    blocks[j],
                    blocks_iron_num[i],
                    blocks_iron_num[j],
                    block_params,
                    cutoff,
                    self.box_length
                )

                scale = 2.0 if i != j else 1.0
                total_energy += energy * scale

                if pair_energy.requires_grad:
                    grad_i, grad_j = torch.autograd.grad(
                        pair_energy.sum(),
                        (blocks[i], blocks[j]),
                        retain_graph=True,
                        allow_unused=True
                    )

                    if grad_i is not None:
                        start_i = i * block_size
                        end_i = start_i + grad_i.size(0)
                        atom_coords.grad[start_i:end_i] -= grad_i * scale

                    if j != i and grad_j is not None:
                        start_j = j * block_size
                        end_j = start_j + grad_j.size(0)
                        atom_coords.grad[start_j:end_j] -= grad_j * scale

        return {
            'energy': total_energy,
            'forces': -atom_coords.grad.detach()
        }
