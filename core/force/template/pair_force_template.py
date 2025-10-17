import torch.nn as nn
from core.md_model import BackboneInterface
from core.auto_force_calculator import auto_force_calculator as afc


class PairForce(BackboneInterface, nn.Module):
    def __init__(self, molecular,param_names,potential_formula):
        super(PairForce, self).__init__()
        self.molecular = molecular
        self.param_names = param_names
        self.potential_formula = potential_formula

    def forward(self):
        r = self.molecular.graph_data.edge_attr
        pos = self.molecular.graph_data.pos
        params = {}
        for name in self.param_names:
            params[name] = self.molecular.get_parameter(name)

        local_env = {'r': r}
        local_env.update(params)
        try:
            v_r = eval(self.potential_formula, {}, local_env)
        except Exception as e:
            raise RuntimeError(f"Error evaluating potential formula: {str(e)}")


        pair_force = afc(v_r, pos, self.molecular.atom_count, self.molecular.device, self.molecular.graph_data.edge_index[0])
        total_energy = v_r.sum()
        atom_force = -pair_force/2
        return {'energy': total_energy, 'forces': atom_force}