import os
from core.element_info import element_info

class Converter:
    def __init__(self):
        self.read_methods = {
            ".txyz": self.read_txyz,
            ".xyz": self.read_xyz,
            ".data": self.read_lammps
        }
        self.write_methods = {
            "lammps": self.write_lammps,
            "xyz": self.write_xyz,
        }

    def read_txyz(self, file_path):
        atom_data = []
        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines[1:]:

            parts = line.split()

            if len(parts) >= 5:  # Format: Type X Y Z
                atom_type, x, y, z = parts[1], *map(float, parts[2:5])
                atom_data.append((atom_type, x, y, z))
        return atom_data

    def read_xyz(self, file_path):
        atom_data = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines[2:]:  # Skip first two header lines
                parts = line.split()
                if len(parts) == 4:  # Format: Type X Y Z
                    atom_type, x, y, z = parts[0], *map(float, parts[1:])
                    atom_data.append((atom_type, x, y, z))
        return atom_data

    def write_lammps(self, atom_data, output_file, box_size=35.0):
        atom_types = sorted(set([atom[0] for atom in atom_data]))
        atom_type_map = {atom: idx + 1 for idx, atom in enumerate(atom_types)}

        with open(output_file, "w") as f:
            f.write("LAMMPS Description\n\n")
            f.write(f"{len(atom_data)} atoms\n")
            f.write(f"{len(atom_types)} atom types\n\n")
            f.write(f"0.0 {box_size:.1f} xlo xhi\n")
            f.write(f"0.0 {box_size:.1f} ylo yhi\n")
            f.write(f"0.0 {box_size:.1f} zlo zhi\n\n")
            f.write("Masses\n\n")
            for atom, idx in atom_type_map.items():
                mass = {"C": 12.01, "H": 1.008}.get(atom, 1.0)  # Add more as needed
                f.write(f"{idx} {mass:.3f}\n")

            f.write("\nAtoms\n\n")
            for i, (atom_type, x, y, z) in enumerate(atom_data, start=1):
                charge_num = element_info[atom_type]["iron_num"]
                f.write(f"{i} {atom_type_map[atom_type]} {charge_num:.2f} {x:.6f} {y:.6f} {z:.6f}\n")

    def write_xyz(self, atom_data, output_file, box_size=10.0):
        with open(output_file, "w") as f:
            f.write(f"{len(atom_data)}\n")
            f.write("Converted by CCMD Converter\n")
            for atom_type, x, y, z in atom_data:
                f.write(f"{atom_type} {x:.6f} {y:.6f} {z:.6f}\n")

    def convert(self, input_file, output_file, output_format=None, box_size=10.0):
        input_suffix = os.path.splitext(input_file)[1].lower()
        if input_suffix not in self.read_methods:
            raise ValueError(f"Unsupported input file format: {input_suffix}")
        read_method = self.read_methods[input_suffix]

        if output_format is None:
            output_suffix = os.path.splitext(output_file)[1].lower()
            if output_suffix not in self.write_methods:
                raise ValueError(f"Unsupported output file format: {output_suffix}")
            output_format = output_suffix.lstrip(".")
        if output_format not in self.write_methods:
            raise ValueError(f"Unsupported output format: {output_format}")
        write_method = self.write_methods[output_format]

        atom_data = read_method(input_file)
        write_method(atom_data, output_file, box_size)

        print(f"Converted {input_file} to {output_file} in {output_format} format.")

    def read_lammps(self, file_path):
        atom_data = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            atoms_index = {}
            for line in lines[10:]:
                parts = line.split()
                if len(parts) == 4:
                    index_number = int(parts[0])
                    atom_type_index = parts[-1]
                    atoms_index[index_number] = atom_type_index
                elif len(parts) == 6:
                    atom_type = atoms_index[int(parts[1])]
                    x, y, z = map(float, parts[3:6])
                    atom_data.append((atom_type, x, y, z))
        return atom_data


# converter = Converter()

# in_path = r'C:\Users\Thinkstation2\Desktop\MYMD_testdata\1.6Na2WO4-2WO3\6Na2WO4-2WO3.data'
# out_path = r'C:\Users\Thinkstation2\Desktop\MYMD_testdata\1.6Na2WO4-2WO3\6Na2WO4-2WO3.xyz'
# converter.convert(in_path, out_path, output_format="xyz")

# in_path = r'C:\Users\Thinkstation2\Desktop\MYMD_testdata\1.Liquid Argon\Ar1e4.xyz'
# out_path = r'C:\Users\Thinkstation2\Desktop\MYMD_testdata\1.Liquid Argon\Ar1e4.lmp'
# converter.convert(in_path, out_path, output_format="lammps")

# in_path = r'C:\Users\Administrator\Desktop\phd_workspace\code_repo\Simulon\run_data\W_bcc_stable.xyz'
# out_path = r'C:\Users\Administrator\Desktop\phd_workspace\code_repo\Simulon\run_data\W_bcc_stable.lmp'
# converter.convert(in_path, out_path, output_format="lammps", box_size=9.6)
