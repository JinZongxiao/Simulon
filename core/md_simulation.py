import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3950"

import torch
# from torch.amp import autocast

from typing import Dict, List
import time
from matplotlib import rcParams
from torch_geometric.data import Data
from graph_diffusion.graph_utils import calc_rho

from core.energy_minimizer import minimize_energy_steepest_descent
import math
from pathlib import Path  

from core.analyser import RDFAccumulator 


matplotlib_config = {
    'font.family': 'Times New Roman',
    'axes.unicode_minus': False,
}
rcParams.update(matplotlib_config)


class MDSimulator:

    def __init__(self, model, num_steps: int, print_interval: int = 10, save_to_graph_dataset = False, spread_mode: str = None,
                 output_dir: str | None = None, dump_interval: int = 0, dump_format: str = 'xyz',
                 write_forces: bool = False, write_energies: bool = False,
                 traj_interval: int | None = None, forces_interval: int | None = None, energies_interval: int | None = None,
                 rdf_accumulator = None):
        self.model = model
        self.num_steps = num_steps
        self.print_interval = print_interval

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model.to(self.device)

        self.save_to_graph_dataset = save_to_graph_dataset
        if self.save_to_graph_dataset:
            self.dataset = []

        self.trajectory = []  
        self.energy_list = []  
        self.force_list = []  
        self.spread_mode = spread_mode

        self.rdf_accumulator = rdf_accumulator

        self._max_store_steps = 2000
        self._store_steps = min(num_steps, self._max_store_steps)
        self._store_buffers = num_steps <= self._max_store_steps
        if self._store_buffers:
            dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
            self.gpu_trajectory = torch.empty((self._store_steps, model.molecular.atom_count, 3), device=self.device, dtype=dtype)
            self.gpu_forces = torch.empty((self._store_steps, model.molecular.atom_count, 3), device=self.device, dtype=dtype)
            self.gpu_energies = torch.empty(self._store_steps, device=self.device, dtype=dtype)
        else:
            self.gpu_trajectory = None
            self.gpu_forces = None
            self.gpu_energies = None

        self.output_dir = Path(output_dir) if output_dir else None
        self.dump_format = (dump_format or 'xyz').lower()
        self.write_forces = bool(write_forces)
        self.write_energies = bool(write_energies)
        self.dump_interval = int(dump_interval or 0)
        self.traj_interval = int(traj_interval) if traj_interval is not None else self.dump_interval
        self.forces_interval = int(forces_interval) if forces_interval is not None else self.dump_interval
        self.energies_interval = int(energies_interval) if energies_interval is not None else None  # None 表示每步写出（兼容旧行为）

        any_traj = (self.traj_interval > 0)
        any_forces = (self.write_forces and self.forces_interval > 0)
        any_energies = self.write_energies
        self.output_enabled = bool(self.output_dir and (any_traj or any_forces or any_energies))
        self._paths = {}
        if self.output_enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if any_traj and self.dump_format == 'xyz':
                traj_path = self.output_dir / 'traj.xyz'
                with open(traj_path, 'w', encoding='utf-8') as _f:
                    pass
                self._paths['traj'] = str(traj_path)
            if self.write_energies:
                e_csv = self.output_dir / 'energies.csv'
                with open(e_csv, 'w', encoding='utf-8') as f:
                    f.write('step,potential,kinetic,total,temperature\n')
                self._paths['energies'] = str(e_csv)
            if any_forces:
                f_dir = self.output_dir / 'forces'
                f_dir.mkdir(parents=True, exist_ok=True)
                self._paths['forces_dir'] = str(f_dir)

        # output profiling
        self.sim_profile = {'output_time': 0.0}

    def _apply_spread(self):
        if self.spread_mode not in ('scale','random'): return
        mol = self.model.molecular
        with torch.no_grad():
            coords = mol.coordinates.clone()
            L = mol.box_length_cpu
            if self.spread_mode == 'scale':
                min_xyz = coords.min(dim=0).values
                max_xyz = coords.max(dim=0).values
                extent = (max_xyz - min_xyz).max().clamp(min=1e-6)
                scale = (L - 1e-6) / extent
                new_coords = (coords - min_xyz) * scale
            else:
                new_coords = torch.rand_like(coords) * L
            mol.update_coordinates(new_coords)
        # print expected neighbors for reference
        number_density = mol.atom_count / (L ** 3)
        cutoff = float(mol.cutoff)
        expected_half = (4.0/3.0)*math.pi*(cutoff**3)*number_density
        print(f"[Spread] Mode={self.spread_mode}, edges={mol.graph_data.edge_index.shape[1]}, expected ave neigh (half, cutoff)~{expected_half:.4f}")

    def _dump_step_outputs(self, step: int, out: Dict[str, torch.Tensor]):
        if not self.output_enabled:
            return
        t0 = time.perf_counter()
        try:
            step_i = step + 1
            if self.traj_interval > 0 and self.dump_format == 'xyz' and (step_i % self.traj_interval == 0):
                traj_path = Path(self._paths.get('traj', ''))
                if traj_path:
                    coords = out['updated_coordinates'].detach().cpu()
                    n = coords.shape[0]
                    pot = float(out['energy'].detach().cpu().item())
                    with open(traj_path, 'a', encoding='utf-8') as f:
                        f.write(f"{n}\n")
                        f.write(f"Step {step_i}, Energy = {pot:.6f}\n")
                        types = self.model.molecular.atom_types if hasattr(self.model.molecular, 'atom_types') else None
                        if types and len(types) == n:
                            for i in range(n):
                                x, y, z = coords[i].tolist()
                                f.write(f"{types[i]} {x:.6f} {y:.6f} {z:.6f}\n")
                        else:
                            for i in range(n):
                                x, y, z = coords[i].tolist()
                                f.write(f"X {x:.6f} {y:.6f} {z:.6f}\n")
            if self.write_forces and self.forces_interval > 0 and (step_i % self.forces_interval == 0):
                fdir = Path(self._paths.get('forces_dir', ''))
                if fdir:
                    fpath = fdir / f'forces_{step_i:06d}.pt'
                    torch.save(out['forces'].detach().cpu(), fpath)
            if self.write_energies:
                write_energy = (self.energies_interval is None) or \
                               (self.energies_interval > 0 and (step_i % self.energies_interval == 0))
                if write_energy:
                    ecsv = Path(self._paths.get('energies', ''))
                    if ecsv:
                        pot = float(out['energy'].detach().cpu().item())
                        kin = float(out['kinetic_energy'].detach().cpu().item())
                        temp = float(out['temperature'].detach().cpu().item())
                        tot = pot + kin
                        with open(ecsv, 'a', encoding='utf-8') as f:
                            f.write(f"{step_i},{pot:.8f},{kin:.8f},{tot:.8f},{temp:.6f}\n")
        finally:
            t1 = time.perf_counter()
            self.sim_profile['output_time'] += (t1 - t0)

    def run(self, enable_minimize_energy: bool = True):
        if self.device == torch.device('cuda'):
            gpu_name = torch.cuda.get_device_name(self.device.index if self.device.index is not None else 0)
            print(f"Simulation will run on GPU: {gpu_name}")
        else:
            print("Simulation will run on CPU")

        self._apply_spread()

        if enable_minimize_energy:
            print("=== Minimizing energy after spreading ===")
            minimize_energy_steepest_descent(self.model)
            print("=== Energy Minimization Completed ===\n")

        loop_t0 = time.perf_counter()
        for step in range(self.num_steps):
            out = self.model()
            if self.rdf_accumulator is not None:
                self.rdf_accumulator.update(step, self.model.molecular)

            if self.gpu_forces is not None:
                idx = step % self._store_steps
                self.gpu_forces[idx] = out['forces'].to(self.gpu_forces.dtype)
                self.gpu_energies[idx] = out['energy'].to(self.gpu_energies.dtype)
                self.gpu_trajectory[idx] = out['updated_coordinates'].to(self.gpu_trajectory.dtype)
            if (step + 1) % self.print_interval == 0:
                t_out0 = time.perf_counter()
                rho = calc_rho(self.model.molecular.graph_data,self.model.molecular.box_length)
                step_data = {
                    'energy': out['energy'],
                    'kinetic_energy': out['kinetic_energy'],
                    'temperature': out['temperature']
                }
                step_data_cpu = {k: v.detach().cpu().item() for k, v in step_data.items()}
                total_energy = step_data_cpu['energy'] + step_data_cpu['kinetic_energy']
                print(
                    f"Step {step + 1}/{self.num_steps}:Tot_E={total_energy:.4f} ev, Pot_E = {step_data_cpu['energy']:.4f} ev, Kin_E = {step_data_cpu['kinetic_energy']:.4f} ev, T = {step_data_cpu['temperature']:.4f} K, Density = {rho: .4f} g/cm3"
                )
                t_out1 = time.perf_counter()
                self.sim_profile['output_time'] += (t_out1 - t_out0)

            self._dump_step_outputs(step, out)

            if self.save_to_graph_dataset:
                graph_data = self.model.molecular.graph_data
                data = Data(
                    x=graph_data.get('x', None),
                    pos = graph_data.get('pos', None),
                    edge_index=graph_data.get('edge_index', None),
                    edge_attr=graph_data.get('edge_attr', None),
                    forces=out['forces'].detach(),
                    energy=out['energy'].detach()
                )
                self.dataset.append(data)

        loop_t1 = time.perf_counter()
        loop_time = loop_t1 - loop_t0

        if self.rdf_accumulator is not None:
            self.rdf_accumulator.finalize()

        if self.gpu_trajectory is not None:
            self.trajectory = self.gpu_trajectory.detach().cpu().numpy()
            self.force_list = self.gpu_forces.detach().cpu().numpy()
            self.energy_list = self.gpu_energies.detach().cpu().numpy()
        else:
            self.trajectory = []
            self.force_list = []
            self.energy_list = []

        print(f"{self.num_steps} steps simulation finished.")
        print(f"Total simulation time: {loop_time:.2f} seconds.")

        self.sim_profile['loop_time'] = loop_time
        return {
            'trajectory': self.trajectory,
            'energies': self.energy_list,
            'forces': self.force_list,
            'output_paths': self._paths if self.output_enabled else {},
            'rdf_file': getattr(self.rdf_accumulator, 'outfile', None) if self.rdf_accumulator else None
        }

    def summarize_profile(self):
        mol = self.model.molecular
        loop_time = self.sim_profile.get('loop_time', 0.0)
        pair_t = self.model.profile['pair_time']
        integ_t = self.model.profile['integrate_time']
        neigh_t = mol.profile['neighbor_time']
        out_t = self.sim_profile['output_time']
        other_t = loop_time - pair_t - integ_t - neigh_t - out_t
        atoms = mol.atom_count
        steps = self.num_steps
        dt_ps = getattr(self.model.Integrator, 'dt', 0.0)
        sim_time_ns = steps * dt_ps / 1000.0
        steps_per_s = steps / loop_time if loop_time>0 else 0
        matom_steps_per_s = (steps*atoms)/loop_time/1e6 if loop_time>0 else 0
        ns_per_day = sim_time_ns / (loop_time/86400.0) if loop_time>0 else 0
        print("\nMPI task timing breakdown (internal):")
        print("Section |  time (s) | %total")
        total = max(loop_time,1e-12)
        def line(name,t):
            print(f"{name:<7}| {t:9.4f} | {100.0*t/total:6.2f}")
        line('Pair',pair_t)
        line('Neigh',neigh_t)
        line('Integr',integ_t)
        line('Output',out_t)
        line('Other',other_t)
        print(f"Loop   | {loop_time:9.4f} | 100.00")
        try:
            eff_edges = mol.effective_edge_count()
            ave_half = eff_edges/atoms
            ave_full = 2*eff_edges/atoms
            # Theoretical expectation (uniform random distribution)
            L = mol.box_length_cpu
            cutoff = float(mol.cutoff)
            number_density = atoms / (L**3)
            expected_full = (4.0/3.0)*math.pi*(cutoff**3)*number_density  # per-atom full neighbor expectation
            expected_half = expected_full / 2.0  # per-atom half-list (edges per atom) expectation
            expected_half_edges_total = expected_half * atoms
            print(f"Effective neighbors (<=cutoff): {eff_edges}  Half-ave/atom={ave_half:.4f}  Full-ave/atom={ave_full:.4f}")
            print(f"Theoretical (uniform): full-ave/atom~{expected_full:.4f}  half-ave/atom~{expected_half:.4f}  expected half edges total~{int(expected_half_edges_total):d}")
        except Exception:
            pass
        print(f"Performance: {ns_per_day:.3f} ns/day, {steps_per_s:.3f} steps/s, {matom_steps_per_s:.3f} Matom-step/s")

    # ── post-run save helpers ─────────────────────────────────────────────────
    def save_energy_curve(self, path: str):
        """
        Save a PNG plot of potential energy vs step number.
        Uses the energies.csv written during the run (if output_dir was set)
        or falls back to the in-memory energy_list buffer.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        steps, pot_energies = [], []

        # Prefer the CSV written during the run (more complete)
        csv_path = self._paths.get('energies', '')
        if csv_path and os.path.exists(csv_path):
            try:
                with open(csv_path, 'r') as f:
                    next(f)  # skip header
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            steps.append(int(parts[0]))
                            pot_energies.append(float(parts[1]))
            except Exception:
                steps, pot_energies = [], []

        # Fall back to in-memory buffer
        if not pot_energies and len(self.energy_list) > 0:
            e = np.asarray(self.energy_list).ravel()
            steps = list(range(1, len(e) + 1))
            pot_energies = e.tolist()

        if not pot_energies:
            print(f"[save_energy_curve] No energy data available; skipping plot.")
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, pot_energies, linewidth=1.0, color='steelblue')
        ax.set_xlabel('Step')
        ax.set_ylabel('Potential Energy (eV)')
        ax.set_title('MD Potential Energy')
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)