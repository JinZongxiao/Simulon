import torch

# mss : amu=g/mol
# radius in angstrom (A)
element_info = {
    'H': {'number': 1, 'mass': 1.008, 'atomic_radius': 0.32,                      'iron_num': 1},
    'He': {'number': 2, 'mass': 4.003},
    'Li': {'number': 3, 'mass': 6.940, 'atomic_radius': 1.45, 'iron_radius': 0.6, 'iron_num': 1},
    'Be': {'number': 4, 'mass': 9.0120},
    'B': {'number': 5, 'mass': 10.810},
    'C': {'number': 6, 'mass': 12.011},
    'N': {'number': 7, 'mass': 14.007},
    'O': {'number': 8, 'mass': 15.999, 'atomic_radius': 0, 'iron_radius': 0, 'iron_num': -2},
    'F': {'number': 9, 'mass': 18.998},
    'Ne': {'number': 10, 'mass': 20.180},
    'Na': {'number': 11, 'mass': 22.990, 'atomic_radius': 1.54, 'iron_radius': 0.95, 'iron_num': 1},
    'Mg': {'number': 12, 'mass': 24.305, 'atomic_radius': 1.36, 'iron_radius': 0.65},
    'Al': {'number': 13, 'mass': 26.982},
    'Si': {'number': 14, 'mass': 28.085},
    'P': {'number': 15, 'mass': 30.974},
    'S': {'number': 16, 'mass': 32.060},
    'Cl': {'number': 17, 'mass': 35.450, 'atomic_radius': 0.99, 'iron_radius': 1.81, 'iron_num': -1},
    'Ar': {'number': 18, 'mass': 39.948, 'atomic_radius': 0.71, 'iron_radius': 0, 'iron_num': 0},
    # more...
    'W': {'number': 74, 'mass': 183.84, 'atomic_radius': 1.30, 'iron_radius': 0.62, 'iron_num': 6},
    # more...
}


def get_element_radius(symbol):
    info = element_info.get(symbol)
    if info:
        radius_tensor = torch.tensor(info['atomic_radius'])
        return radius_tensor
    return None


def get_element_iron_radius(symbol):
    info = element_info.get(symbol)
    if info:
        radius_tensor = torch.tensor(info['iron_radius'])
        return radius_tensor
    return None


def get_element_iron_num(symbol):
    info = element_info.get(symbol)
    if info:
        number_tensor = torch.tensor(info['iron_num'])
        return number_tensor
    return None


def get_element_mass(symbol):
    info = element_info.get(symbol)
    if info:
        mass_tensor = torch.tensor(info['mass'])
        return mass_tensor
    return None


def get_element_number(symbol):
    info = element_info.get(symbol)
    if info:
        number_tensor = torch.tensor(info['number'])
        return number_tensor
    return None
