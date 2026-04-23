import torch
from scipy.interpolate import CubicSpline
import numpy as np
from collections import defaultdict

class EAMParser:
    """
    A parser for EAM (Embedded Atom Model) potential files, supporting both
    DYNAMO 'setfl' (single element) and LAMMPS 'eam.fs' or 'eam.alloy' (multi-element) formats.
    """
    def __init__(self, filepath: str, device: torch.device):
        """
        Initializes the parser and processes the EAM potential file.

        Args:
            filepath (str): The path to the EAM potential file.
            device (torch.device): The PyTorch device (e.g., 'cuda' or 'cpu') to store tensors on.
        """
        self.filepath = filepath
        self.device = device
        self._parse()

    def _parse(self):
        """
        Parses the EAM file, detecting the format and extracting data.
        """
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        # Skip header comments (usually 3 lines, but can vary)
        # The first non-comment line should contain element info
        first_data_line_index = 0
        for i, line in enumerate(lines):
            if i > 2 and len(line.strip()) > 0:
                first_data_line_index = i
                break
        
        header_lines = lines[first_data_line_index:]
        
        # Detect format based on the first line of the header
        header_line_1_parts = header_lines[0].split()
        
        if len(header_line_1_parts) > 1 and all(c.isalpha() for c in header_line_1_parts[1:]):
            # Looks like a multi-element format (e.g., '2 W Re')
            self._parse_alloy(header_lines)
        else:
            # Assume single-element setfl format
            self._parse_setfl(lines) # Pass original lines for setfl

    def _parse_setfl(self, lines):
        """Parses the single-element DYNAMO setfl format."""
        # Find header lines, skipping comments
        header_start_idx = 0
        for i in range(len(lines)):
            if len(lines[i].strip()) > 0 and lines[i][0] != '#':
                header_start_idx = i
                break
        
        header_line_4 = lines[header_start_idx+3].split()
        self.elements = [header_line_4[3]] # e.g., FCC
        self.element_map = {self.elements[0]: 0}
        
        params_line_5 = lines[header_start_idx+4].split()
        self.n_rho = int(params_line_5[0])
        self.d_rho = float(params_line_5[1])
        self.n_r = int(params_line_5[2])
        self.d_r = float(params_line_5[3])
        self.cutoff = float(params_line_5[4])

        # Read all data points from subsequent lines
        data_lines = [float(val) for line in lines[header_start_idx+5:] for val in line.split()]
        
        self.embedding_splines = {}
        self.embedding_deriv_splines = {}
        self.density_splines = {}
        self.density_deriv_splines = {}
        self.pair_potential_splines = defaultdict(dict)
        self.pair_potential_deriv_splines = defaultdict(dict)

        # Embedding function F(rho)
        embedding_data = np.array(data_lines[:self.n_rho])
        rho_values = np.arange(self.n_rho) * self.d_rho
        self.embedding_splines[0] = CubicSpline(rho_values, embedding_data, bc_type='not-a-knot')
        self.embedding_deriv_splines[0] = self.embedding_splines[0].derivative(1)

        # Electron density function f(r)
        density_data = np.array(data_lines[self.n_rho : self.n_rho + self.n_r])
        r_values = np.arange(self.n_r) * self.d_r
        self.density_splines[0] = CubicSpline(r_values, density_data, bc_type='not-a-knot')
        self.density_deriv_splines[0] = self.density_splines[0].derivative(1)

        # Pair potential phi(r)
        pair_potential_data = np.array(data_lines[self.n_rho + self.n_r : self.n_rho + 2 * self.n_r])
        # In setfl, pair potential is often r*phi(r)
        phi_r = np.divide(pair_potential_data, r_values, out=np.zeros_like(pair_potential_data), where=r_values!=0)
        self.pair_potential_splines[0][0] = CubicSpline(r_values, phi_r, bc_type='not-a-knot')
        self.pair_potential_deriv_splines[0][0] = self.pair_potential_splines[0][0].derivative(1)

    def _parse_alloy(self, lines):
        """Parses the multi-element eam.alloy or eam.fs format."""
        line_idx = 0
        
        # Line 1: n_elements, element names
        header_line_1 = lines[line_idx].split()
        n_elements = int(header_line_1[0])
        self.elements = header_line_1[1:]
        self.element_map = {name: i for i, name in enumerate(self.elements)}
        line_idx += 1

        # Line 2: n_rho, d_rho, n_r, d_r, cutoff
        params_line_2 = lines[line_idx].split()
        self.n_rho = int(params_line_2[0])
        self.d_rho = float(params_line_2[1])
        self.n_r = int(params_line_2[2])
        self.d_r = float(params_line_2[3])
        self.cutoff = float(params_line_2[4])
        line_idx += 1

        # Check if we have element-specific header lines
        # Some .eam.fs files have one line per element, others have just one line for all
        element_header_lines = 0
        for i in range(min(n_elements, 5)):  # Check at most 5 lines
            test_line = lines[line_idx + i].strip()
            # Check if line contains non-scientific notation letters (not E+ or E-)
            if test_line and any(c.isalpha() and c not in 'Ee+-' for c in test_line):
                element_header_lines += 1
            else:
                break
        
        # Skip the detected element header lines
        line_idx += element_header_lines

        self.embedding_splines = {}
        self.embedding_deriv_splines = {}
        self.density_splines = {}
        self.density_deriv_splines = {}
        self.pair_potential_splines = defaultdict(dict)
        self.pair_potential_deriv_splines = defaultdict(dict)

        # Read numerical data from remaining lines
        all_data = []
        for line in lines[line_idx:]:
            line = line.strip()
            if line:  # Process all non-empty lines
                try:
                    values = [float(val) for val in line.split()]
                    all_data.extend(values)
                except ValueError:
                    # Skip lines that can't be converted to float
                    continue
        
        data_ptr = 0

        for i in range(n_elements):            
            # Embedding function F(rho)
            if data_ptr + self.n_rho > len(all_data):
                raise ValueError(f"嵌入函数数据不足: 需要 {self.n_rho} 点，但只有 {len(all_data) - data_ptr} 点可用")
            
            embedding_data = np.array(all_data[data_ptr : data_ptr + self.n_rho])
            data_ptr += self.n_rho
            rho_values = np.arange(self.n_rho) * self.d_rho
            
            self.embedding_splines[i] = CubicSpline(rho_values, embedding_data, bc_type='not-a-knot')
            self.embedding_deriv_splines[i] = self.embedding_splines[i].derivative(1)

            # Electron density function f(r)
            if data_ptr + self.n_r > len(all_data):
                raise ValueError(f"电子密度函数数据不足: 需要 {self.n_r} 点，但只有 {len(all_data) - data_ptr} 点可用")
                
            density_data = np.array(all_data[data_ptr : data_ptr + self.n_r])
            data_ptr += self.n_r
            r_values = np.arange(self.n_r) * self.d_r
            self.density_splines[i] = CubicSpline(r_values, density_data, bc_type='not-a-knot')
            self.density_deriv_splines[i] = self.density_splines[i].derivative(1)

        # Read pair potentials phi_ij(r)
        r_values = np.arange(self.n_r) * self.d_r
        for i in range(n_elements):
            for j in range(i, n_elements):
                pair_potential_data = np.array(all_data[data_ptr : data_ptr + self.n_r])
                data_ptr += self.n_r
                
                # In alloy/fs format, it's r*phi(r)
                phi_r = np.divide(pair_potential_data, r_values, out=np.zeros_like(pair_potential_data), where=r_values!=0)
                
                spline = CubicSpline(r_values, phi_r, bc_type='not-a-knot')
                deriv_spline = spline.derivative(1)
                
                self.pair_potential_splines[i][j] = spline
                self.pair_potential_deriv_splines[i][j] = deriv_spline
                if i != j:
                    self.pair_potential_splines[j][i] = spline
                    self.pair_potential_deriv_splines[j][i] = deriv_spline

if __name__ == '__main__':
    # Example usage with a dummy eam.fs file
    dummy_fs_file = 'dummy_W_Re.eam.fs'
    with open(dummy_fs_file, 'w') as f:
        f.write("Dummy W-Re EAM file\\n")
        f.write("Generated for testing purposes\\n")
        f.write("Format: LAMMPS eam.fs\\n")
        f.write("2 W Re\\n")
        f.write("100 0.1 100 0.05 5.0\\n") # n_rho, d_rho, n_r, d_r, cutoff
        f.write("74 183.84 3.16 bcc\\n") # W info
        f.write("75 186.21 3.10 hcp\\n") # Re info
        # Dummy data (zeros)
        # W: F(rho), f(r) -> 100 + 100 = 200 values
        # Re: F(rho), f(r) -> 100 + 100 = 200 values
        # Pairs: W-W, W-Re, Re-Re -> 100 + 100 + 100 = 300 values
        # Total = 700 values
        for _ in range(70):
            f.write(" ".join(["0.0"] * 10) + "\\n")

    parser = EAMParser(dummy_fs_file, torch.device('cpu'))
    print(f"Parsed EAM potential for elements: {parser.elements}")
    print(f"Element map: {parser.element_map}")
    print(f"Cutoff distance: {parser.cutoff} Angstroms")
    assert len(parser.embedding_splines) == 2
    assert len(parser.density_splines) == 2
    assert len(parser.pair_potential_splines) == 2
    assert len(parser.pair_potential_splines[0]) == 2 # W-W, W-Re
    assert len(parser.pair_potential_splines[1]) == 2 # Re-W, Re-Re
    print("\\nParsing successful!")

    import os
    os.remove(dummy_fs_file)
