import re

def extract_energies(input_file, output_file):
    steps = []
    energies = []
    
    with open(input_file, 'r') as f:
        while True:
            n_atoms_line = f.readline().strip()
            if not n_atoms_line:
                break
            
            try:
                n_atoms = int(n_atoms_line)
            except ValueError:
                print(f"无效的原子数量行: {n_atoms_line}")
                break
            
            comment_line = f.readline().strip()
            match = re.match(r'Step\s+(\d+),\s*Energy\s*=\s*(-?\d+\.?\d+)', comment_line)
            
            if not match:
                print(f"无法解析注释行: {comment_line}")
                for _ in range(n_atoms):
                    f.readline()
                continue
            
            steps.append(match.group(1))
            energies.append(match.group(2))
            
            for _ in range(n_atoms):
                f.readline()
    
    with open(output_file, 'w') as f_out:
        f_out.write("Step|Energy\n")
        for step, energy in zip(steps, energies):
            f_out.write(f"{step}|{energy}\n")