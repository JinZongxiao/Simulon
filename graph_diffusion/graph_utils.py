import torch


def load_graph(filename: str):

    return torch.load(filename)

# dataset = load_graph("C:\\Users\\Thinkstation2\\Desktop\\MD_graph_dataset.pt")
# print(dataset)


def calc_rho(data, box_length):

    NA = 6.02214076e23

    if torch.is_tensor(box_length):
        L = float(box_length.detach().cpu().item())
    else:
        L = float(box_length)

    V_cm3 = (L ** 3) * 1e-24

    total_g_per_mol = float(data.x[:, 2].sum().detach().cpu().item())
    M_g = total_g_per_mol / NA

    rho = M_g / V_cm3
    return rho