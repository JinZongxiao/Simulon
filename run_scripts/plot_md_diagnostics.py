# -*- coding: utf-8 -*-
"""
Plot comprehensive diagnostics for the Simulon MD framework.

- Runs a short MD (default 500 steps) on a given structure (default run_data/Ar10000.xyz)
- Plots (English, Times New Roman):
  1) Potential Energy vs Step
  2) Energy per Atom vs Step
  3) Max Force vs Step
  4) Force Percentiles (50/90/95/99%) vs Step
  5) Force Magnitude Histogram (final)
  6) Mean Square Displacement (MSD) vs Step (based on saved trajectory)
  7) Radial Distribution Function (g(r)) at initial and final snapshots
  8) Final snapshot XY scatter (periodic projection)
  9) Neighbor Degree Histogram (initial & final)
 10) Performance Summary (atoms, edges, device, total time, steps/s, peak GPU memory)

Usage (Windows PowerShell):
  python run_scripts/plot_md_diagnostics.py --steps 500
  python run_scripts/plot_md_diagnostics.py --structure run_data/Ar10000.xyz --box-length 200 --cutoff 7 --no-minimize

Notes:
- To ensure arrays are saved for plotting, keep steps <= 2000 (per current MDSimulator buffering).
- Plots are saved to run_data/output/plots_YYYYmmdd_HHMMSS/
"""
import os
import sys
import math
import time
import argparse
from pathlib import Path

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
from matplotlib import rcParams

import matplotlib.font_manager as fm
fm.fontManager.addfont('/public/share/normal_bgd/fonts/TIMES.TTF')

# Set global font to Times New Roman, English labels
rcParams.update({
    'font.family': 'Times New Roman',
    'axes.unicode_minus': False,
})

# Project imports
from io_utils.reader import AtomFileReader
from core.force.lennard_jones_force_cu import LennardJonesForce
from core.md_model import SumBackboneInterface, BaseModel
from core.integrator.integrator import VerletIntegrator
from core.md_simulation import MDSimulator


def _safe_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _find_structure_path(name_or_path: str) -> str:
    p = Path(name_or_path)
    if p.suffix.lower() == '.xyz' and p.exists():
        return str(p)
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / 'run_data' / 'Ar10000.xyz',
        repo_root / 'run_data' / 'Ar1000.xyz',
        repo_root / 'run_data' / 'Ar.xyz',
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"Cannot find structure file. Tried: {candidates}")


def compute_rdf(positions: np.ndarray, box_length: float, r_max: float, dr: float):
    """
    Compute radial distribution function g(r) for a single snapshot.
    positions: (N,3) in Angstrom
    box_length: cubic box length (Angstrom)
    r_max: maximum radius (<= box_length/2)
    dr: bin width
    """
    N = positions.shape[0]
    L = float(box_length)
    r_max = min(r_max, L / 2.0)

    # Number density (per A^3)
    number_density = N / (L**3)

    # Bins
    edges = np.arange(0.0, r_max + dr, dr)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros_like(centers)

    # Pair distances with minimum image convention
    # For large N, sample to reduce cost
    max_pairs = 600000  # cap pairs for speed
    if N * (N - 1) // 2 > max_pairs:
        idx = np.random.choice(N, size=min(N, 3000), replace=False)
        pos = positions[idx]
    else:
        pos = positions

    M = pos.shape[0]
    for i in range(M - 1):
        rij = pos[i+1:] - pos[i]
        # minimum image
        rij -= L * np.round(rij / L)
        dij = np.linalg.norm(rij, axis=1)
        hist, _ = np.histogram(dij, bins=edges)
        counts += hist

    # Normalize counts to g(r)
    pairs = M * (M - 1) / 2
    shell_vol = 4.0 * math.pi * centers**2 * dr
    # Scale to full system density
    ideal_counts = shell_vol * number_density * pairs / max(1.0, (N * (N - 1) / 2))
    g_r = np.divide(counts, ideal_counts + 1e-12)

    return centers, g_r


def compute_msd(traj: np.ndarray):
    """
    Mean Square Displacement vs step using first frame as reference.
    traj: (T, N, 3)
    """
    if not isinstance(traj, np.ndarray) or traj.size == 0:
        return None, None
    ref = traj[0]
    disp2 = np.sum((traj - ref[None, ...])**2, axis=2)  # (T,N)
    msd = disp2.mean(axis=1)
    steps = np.arange(traj.shape[0])
    return steps, msd


def degree_hist(edge_index: np.ndarray, n_atoms: int):
    """Compute per-atom degree (both ends) from single-directed edge_index (2,E)."""
    if edge_index is None:
        return None
    idx = np.concatenate([edge_index[0], edge_index[1]], axis=0)
    deg = np.bincount(idx, minlength=n_atoms)
    return deg


def plot_energy(energies: np.ndarray, out_dir: Path):
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(len(energies)), energies, lw=1.5)
    plt.xlabel('Step')
    plt.ylabel('Potential Energy (eV)')
    plt.title('Energy vs Step')
    plt.tight_layout()
    plt.savefig(out_dir / 'energy_vs_step.png', dpi=150)
    plt.close()


def plot_energy_per_atom(energies: np.ndarray, n_atoms: int, out_dir: Path):
    epa = energies / max(1, n_atoms)
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(len(epa)), epa, lw=1.5)
    plt.xlabel('Step')
    plt.ylabel('Energy per Atom (eV/atom)')
    plt.title('Energy per Atom vs Step')
    plt.tight_layout()
    plt.savefig(out_dir / 'energy_per_atom_vs_step.png', dpi=150)
    plt.close()


def plot_max_force(forces: np.ndarray, out_dir: Path):
    # forces: (T, N, 3)
    norms = np.linalg.norm(forces, axis=2)
    maxf = norms.max(axis=1)
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(len(maxf)), maxf, lw=1.5)
    plt.xlabel('Step')
    plt.ylabel('Max |F| (eV/Å)')
    plt.title('Max Force vs Step')
    plt.tight_layout()
    plt.savefig(out_dir / 'max_force_vs_step.png', dpi=150)
    plt.close()


def plot_force_percentiles(forces: np.ndarray, out_dir: Path):
    norms = np.linalg.norm(forces, axis=2)  # (T,N)
    prc = [50, 90, 95, 99]
    curves = {p: np.percentile(norms, p, axis=1) for p in prc}
    plt.figure(figsize=(7,4))
    for p in prc:
        plt.plot(curves[p], label=f'P{p}')
    plt.xlabel('Step')
    plt.ylabel('|F| Percentiles (eV/Å)')
    plt.title('Force Percentiles vs Step')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'force_percentiles_vs_step.png', dpi=150)
    plt.close()


def plot_force_hist(forces_last: np.ndarray, out_dir: Path):
    norms = np.linalg.norm(forces_last, axis=1)
    plt.figure(figsize=(6,4))
    plt.hist(norms, bins=60, alpha=0.85)
    plt.xlabel('|F| (eV/Å)')
    plt.ylabel('Count')
    plt.title('Force Magnitude Histogram (Final)')
    plt.tight_layout()
    plt.savefig(out_dir / 'force_hist_final.png', dpi=150)
    plt.close()


def plot_xy_scatter(pos_last: np.ndarray, box_length: float, out_dir: Path):
    plt.figure(figsize=(6,6))
    L = float(box_length)
    xy = pos_last[:, :2] % L
    plt.scatter(xy[:,0], xy[:,1], s=1, alpha=0.5)
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x (Å)')
    plt.ylabel('y (Å)')
    plt.title('Final Snapshot (XY)')
    plt.tight_layout()
    plt.savefig(out_dir / 'final_xy_scatter.png', dpi=150)
    plt.close()


def plot_msd(steps: np.ndarray, msd: np.ndarray, out_dir: Path):
    if steps is None or msd is None:
        return
    plt.figure(figsize=(6,4))
    plt.plot(steps, msd, lw=1.5)
    plt.xlabel('Step')
    plt.ylabel('MSD (Å²)')
    plt.title('Mean Square Displacement vs Step')
    plt.tight_layout()
    plt.savefig(out_dir / 'msd_vs_step.png', dpi=150)
    plt.close()


def plot_neighbor_histogram(edge_index: np.ndarray, n_atoms: int, out_dir: Path, tag: str):
    deg = degree_hist(edge_index, n_atoms)
    if deg is None:
        return
    plt.figure(figsize=(6,4))
    plt.hist(deg, bins=range(int(deg.max()) + 2), alpha=0.85)
    plt.xlabel('Degree (neighbors)')
    plt.ylabel('Count')
    plt.title(f'Neighbor Degree Histogram ({tag})')
    plt.tight_layout()
    plt.savefig(out_dir / f'neighbor_degree_{tag}.png', dpi=150)
    plt.close()


def plot_performance(metrics: dict, out_dir: Path):
    plt.figure(figsize=(7,5))
    plt.axis('off')
    lines = [
        f"Device: {metrics.get('device_name', metrics.get('device'))}",
        f"Atoms: {metrics.get('atoms')}  |  Edges: {metrics.get('edges')}",
        f"Steps: {metrics.get('steps')}  |  Total Time: {metrics.get('total_time'):.3f} s",
        f"Throughput: {metrics.get('steps')/max(1e-9, metrics.get('total_time')):.2f} steps/s",
    ]
    if metrics.get('gpu_peak_mem') is not None:
        lines.append(f"Peak GPU Memory: {metrics['gpu_peak_mem'] / (1024**3):.2f} GiB")
    text = "\n".join(lines)
    plt.text(0.05, 0.95, text, va='top', ha='left', fontsize=12)
    plt.title('Performance Summary', pad=20)
    plt.tight_layout()
    plt.savefig(out_dir / 'performance_summary.png', dpi=150)
    plt.close()


def analyze_and_print_console(mol, args, energies, forces, traj, perf, out_dir: Path):
    N = int(mol.atom_count)
    steps = int(args.steps)
    dt_ps = float(args.dt)
    sim_time_ps = steps * dt_ps
    wall_s = perf['total_time']

    # LAMMPS-like metrics
    steps_per_s = steps / max(1e-12, wall_s)
    atom_steps_per_s = (N * steps) / max(1e-12, wall_s)
    ns_per_day = (sim_time_ps * 1e-3) / max(1e-12, wall_s / 86400.0)  # ps->ns, s->day
    time_per_step_ms = 1000.0 * wall_s / max(1, steps)

    # Neighbor stats
    try:
        ei = mol.graph_data.edge_index
        if ei is not None:
            ei_np = ei.detach().cpu().numpy()
            E = int(ei_np.shape[1])
            deg = degree_hist(ei_np, N)
            avg_neighbors = float(deg.mean()) if deg is not None else None
        else:
            E = None
            avg_neighbors = None
    except Exception:
        E = None
        avg_neighbors = None

    print("\n================ MD PERFORMANCE SUMMARY ================")
    print(f"Device         : {perf.get('device_name', perf.get('device'))}")
    print(f"Atoms          : {N}")
    if E is not None:
        print(f"Edges (directed): {E}  |  Avg neighbors/atom ~ {avg_neighbors:.2f}")
    print(f"Steps          : {steps}  |  Timestep: {dt_ps} ps  |  Sim time: {sim_time_ps} ps")
    print(f"Wall time (s)  : {wall_s:.3f}  |  Time/step: {time_per_step_ms:.3f} ms")
    print(f"Throughput     : {steps_per_s:.2f} steps/s  |  {atom_steps_per_s/1e6:.3f} M atom-steps/s  |  {ns_per_day:.2f} ns/day")
    if perf.get('gpu_peak_mem') is not None:
        print(f"Peak GPU Mem   : {perf['gpu_peak_mem']/(1024**3):.2f} GiB")
    print("=======================================================\n")

    # Data quality assessment
    print("---------------- Data Quality Assessment ---------------")
    # Energy drift
    if isinstance(energies, np.ndarray) and energies.size > 1:
        dE = float(energies[-1] - energies[0])
        drift_per_1k = dE / (len(energies) - 1) * 1000.0
        trend = "decreasing" if dE < 0 else ("increasing" if dE > 0 else "flat")
        print(f"Energy drift   : total ΔE = {dE:.4f} eV ({trend}); per 1000 steps ~ {drift_per_1k:.4f} eV")
    else:
        print("Energy drift   : N/A (no energy buffer)")

    # Force stats (final)
    if isinstance(forces, np.ndarray) and forces.size > 0:
        f_last = forces[-1]
        mags = np.linalg.norm(f_last, axis=1)
        p50, p90, p99 = np.percentile(mags, [50, 90, 99])
        fmax = mags.max()
        print(f"Forces (final) : P50={p50:.4f}, P90={p90:.4f}, P99={p99:.4f}, Max={fmax:.4f} (eV/Å)")
    else:
        print("Forces (final) : N/A (no force buffer)")

    # RDF peak (final)
    try:
        if isinstance(traj, np.ndarray) and traj.size > 0:
            pos_last = traj[-1]
        else:
            pos_last = mol.coordinates.detach().cpu().numpy()
        L = float(mol.box_length.detach().cpu().item() if torch.is_tensor(mol.box_length) else mol.box_length)
        r_max = min(L/2.0, max(8.0, args.cutoff * 1.5))
        r, g = compute_rdf(pos_last, L, r_max=r_max, dr=0.05)
        # ignore very small r bins when finding first peak
        start = max(1, int(0.2 / 0.05))  # ~0.2 Å
        peak_idx = start + int(np.argmax(g[start:]))
        print(f"RDF (final)    : first peak at r ≈ {r[peak_idx]:.2f} Å, g(r) ≈ {g[peak_idx]:.2f}; g(r)->{np.mean(g[-10:]):.2f} at large r")
    except Exception as e:
        print(f"RDF (final)    : N/A ({e})")

    # MSD and diffusion estimate
    try:
        if isinstance(traj, np.ndarray) and traj.size > 0:
            steps_arr, msd = compute_msd(traj)
            if steps_arr is not None:
                t_ps = steps_arr * dt_ps
                # linear fit on last half for diffusion estimate
                k0 = len(t_ps) // 2
                if len(t_ps) - k0 >= 2:
                    coeff = np.polyfit(t_ps[k0:], msd[k0:], 1)
                    slope = coeff[0]  # Å^2 / ps
                    D_A2_ps = slope / 6.0
                    D_cm2_s = D_A2_ps * 1e-4
                    print(f"MSD (final)    : MSD_end={msd[-1]:.3f} Å²; diffusion D≈{D_A2_ps:.4e} Å²/ps ({D_cm2_s:.4e} cm²/s)")
                else:
                    print(f"MSD (final)    : MSD_end={msd[-1]:.3f} Å² (not enough points for diffusion fit)")
            else:
                print("MSD            : N/A (no trajectory buffer)")
        else:
            print("MSD            : N/A (no trajectory buffer)")
    except Exception as e:
        print(f"MSD            : N/A ({e})")

    print("-------------------------------------------------------\n")
    print(f"Images saved to: {out_dir.resolve()}")


def _ensure_np(x):
    try:
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, 'detach') and callable(getattr(x, 'detach')):
            return x.detach().cpu().numpy()
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return np.array([])
            # try stack if shapes align
            try:
                return np.stack([_ensure_np(e) for e in x], axis=0)
            except Exception:
                return np.array(x)
        return np.array(x)
    except Exception:
        return np.array([])


def main():
    parser = argparse.ArgumentParser(description='Plot MD diagnostics')
    parser.add_argument('--structure', type=str, default='run_data/Ar10000.xyz', help='Structure .xyz path')
    parser.add_argument('--box-length', type=float, default=200.0, help='Cubic box length (Å)')
    parser.add_argument('--cutoff', type=float, default=7.0, help='Neighbor cutoff (Å)')
    parser.add_argument('--steps', type=int, default=10000, help='MD steps (<=2000 to store arrays)')
    parser.add_argument('--dt', type=float, default=0.001, help='Time step (ps)')
    # 默认开启最小化，提供显式关闭开关
    parser.add_argument('--minimize', dest='minimize', action='store_true', default=True, help='Enable energy minimization before MD (default: ON)')
    parser.add_argument('--no-minimize', dest='minimize', action='store_false', help='Disable energy minimization before MD')
    args = parser.parse_args()

    device = _safe_device()
    structure_path = _find_structure_path(args.structure)

    # Build system
    parameters_pair = {"[0 0]": {"epsilon": 0.0104, "sigma": 3.405}}
    mol = AtomFileReader(
        filename=structure_path,
        box_length=args.box_length,
        cutoff=args.cutoff,
        device=device,
        parameter=parameters_pair,
        skin_thickness=3.0,
    )

    # Build model
    lj = LennardJonesForce(mol)
    bone = SumBackboneInterface([lj], mol)
    vi = VerletIntegrator(
        molecular=mol,
        dt=args.dt,
        force_field=lj,
        ensemble='NVT',
        temperature=[94.4, 94.4],
        gamma=1/0.001,
    )
    model = BaseModel(bone, vi, mol)

    # Timing & GPU peak memory
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()

    # 明确打印当前最小化开关状态
    print(f"[Diagnostics] Energy minimization: {'ON' if args.minimize else 'OFF'}")

    # Run MD
    sim = MDSimulator(model, num_steps=int(args.steps), print_interval=min(args.steps, 50), save_to_graph_dataset=False)
    sim.run(enable_minimize_energy=bool(args.minimize))

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()

    # Prepare output dir
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = Path('run_data') / 'output' / f'plots_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Access buffers
    traj = getattr(sim, 'trajectory', [])
    forces = getattr(sim, 'force_list', [])
    energies = getattr(sim, 'energy_list', [])

    # 转为 numpy，便于后续一致处理
    traj = _ensure_np(traj)
    forces = _ensure_np(forces)
    energies = _ensure_np(energies)

    # Plot energy vs step if available
    if isinstance(energies, np.ndarray) and energies.size > 0:
        plot_energy(energies, out_dir)
        plot_energy_per_atom(energies, mol.atom_count, out_dir)
    else:
        print('[Warn] Energies buffer not available (steps > buffer size?). Skipping energy plots.')

    # Plot max force and percentiles if available
    if isinstance(forces, np.ndarray) and forces.size > 0:
        plot_max_force(forces, out_dir)
        plot_force_percentiles(forces, out_dir)
        plot_force_hist(forces[-1], out_dir)
    else:
        print('[Warn] Forces buffer not available. Skipping force plots.')

    # MSD and XY scatter & RDF
    try:
        if isinstance(traj, np.ndarray) and traj.size > 0:
            pos_last = traj[-1]
        else:
            pos_last = mol.coordinates.detach().cpu().numpy()
        steps_arr, msd = compute_msd(traj) if isinstance(traj, np.ndarray) and traj.size > 0 else (None, None)
        plot_msd(steps_arr, msd, out_dir)

        L = float(mol.box_length.detach().cpu().item() if torch.is_tensor(mol.box_length) else mol.box_length)
        r_max = min(L/2.0, max(8.0, args.cutoff * 1.5))
        # RDF initial & final
        try:
            if isinstance(traj, np.ndarray) and traj.shape[0] > 0:
                pos_init = traj[0]
            else:
                pos_init = mol.coordinates.detach().cpu().numpy()
            r_i, g_i = compute_rdf(pos_init, L, r_max=r_max, dr=0.05)
            r_f, g_f = compute_rdf(pos_last, L, r_max=r_max, dr=0.05)
            plt.figure(figsize=(6,4))
            plt.plot(r_i, g_i, label='Initial', lw=1.5)
            plt.plot(r_f, g_f, label='Final', lw=1.5)
            plt.xlabel('r (Å)')
            plt.ylabel('g(r)')
            plt.title('Radial Distribution Function')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / 'rdf_initial_final.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f'[Warn] Failed to compute RDF: {e}')

        plot_xy_scatter(pos_last, L, out_dir)
    except Exception as e:
        print(f'[Warn] Failed to process trajectory-based plots: {e}')

    # Neighbor degree histogram (final)
    try:
        # Access current graph
        edge_index = None
        try:
            ei = mol.graph_data.edge_index
            if ei is not None:
                edge_index = ei.detach().cpu().numpy()
        except Exception:
            edge_index = None
        if edge_index is not None:
            plot_neighbor_histogram(edge_index, mol.atom_count, out_dir, tag='final')
        # If trajectory saved, we cannot easily rebuild edge_index per frame here; showing final is enough
    except Exception as e:
        print(f'[Warn] Neighbor histogram failed: {e}')

    # Performance summary for plotting
    gpu_peak = None
    if device.type == 'cuda':
        try:
            gpu_peak = torch.cuda.max_memory_allocated()
        except Exception:
            gpu_peak = None
        try:
            dev_name = torch.cuda.get_device_name(0)
        except Exception:
            dev_name = 'CUDA'
    else:
        dev_name = 'CPU'

    edges_now = None
    try:
        edges_now = int(mol.graph_data.edge_index.shape[1])
    except Exception:
        edges_now = None

    perf = {
        'device': device.type,
        'device_name': dev_name,
        'atoms': int(mol.atom_count),
        'edges': edges_now,
        'steps': int(args.steps),
        'total_time': float(t1 - t0),
        'gpu_peak_mem': gpu_peak,
    }
    plot_performance(perf, out_dir)

    # Console performance + data assessment
    analyze_and_print_console(mol, args, energies, forces, traj, perf, out_dir)


if __name__ == '__main__':
    main()
