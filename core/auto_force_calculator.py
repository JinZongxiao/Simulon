import torch

def auto_force_calculator(v_r,r,atom_count,device,index):
    autograd = torch.autograd.grad(v_r.sum(), r, create_graph=True)[0]
    return autograd