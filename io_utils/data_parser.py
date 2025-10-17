from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


def read_cp2k_to_structure_dict(force_xyz_file: str, pos_xyz_file: str, cell_param=[15, 15, 15], output_file=None):
    positions_frames = read(pos_xyz_file, index=":")
    forces_frames = read(force_xyz_file, index=":")
    structures = []
    forces = []
    energies = []
    for pos_atoms, force_atoms in tqdm(zip(positions_frames, forces_frames)):
        pos_atoms.set_cell(cell_param)
        pos_atoms.set_pbc(True)
        structure = AseAtomsAdaptor().get_structure(pos_atoms)
        structures.append(structure)

        force_atom = force_atoms.positions
        forces.append(force_atom)

        energy = pos_atoms.info['E']
        energies.append(energy)

    return {"structures": structures, "forces": forces, "energies": energies}
