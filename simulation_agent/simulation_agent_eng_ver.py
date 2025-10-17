# -*- coding: utf-8 -*-
import os
from pathlib import Path
import datetime

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

# Extra imports
import csv
import json
from typing import Dict, Any, Optional
try:
    import numpy as np
except Exception:
    np = None


LJ_SCRIPT_EXAMPLE = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Example: Lennard-Jones system script (interactive + auto hints)
import torch, os
from io_utils.reader import AtomFileReader
from core.force.lennard_jones_force_cu import LennardJonesForce
from core.md_model import SumBackboneInterface, BaseModel
from core.integrator.integrator import VerletIntegrator
from core.md_simulation import MDSimulator

DEFAULT_XYZ = 'run_data/Ar10000.xyz'
DEFAULT_EPS = 0.0104  # Argon
DEFAULT_SIG = 3.405

def detect_atom_types(path: str):
    types = set()
    try:
        with open(path, 'r') as f:
            first = f.readline().strip()
            try: n = int(first)
            except: return types
            _ = f.readline()
            for _ in range(n):
                line = f.readline()
                if not line: break
                sp = line.split()
                if sp: types.add(sp[0])
    except Exception:
        pass
    return types

xyz_path = input(f'Structure file path (default {DEFAULT_XYZ}): ').strip() or DEFAULT_XYZ
box_input = input('Box length (Å) (leave blank to use 1550.0): ').strip()
box_length = float(box_input) if box_input else 1550.0
cutoff_input = input('Cutoff (Å) (default 7.0): ').strip()
cutoff = float(cutoff_input) if cutoff_input else 7.0
# Force field params
atom_types = detect_atom_types(xyz_path)
print(f'Detected atom types: {sorted(atom_types) if atom_types else "(unknown)"}')
if atom_types and atom_types == {'Ar'}:
    default_eps, default_sig = DEFAULT_EPS, DEFAULT_SIG
else:
    default_eps = DEFAULT_EPS
    default_sig = DEFAULT_SIG

eps_in = input(f'LJ epsilon (eV) default {default_eps}: ').strip()
sig_in = input(f'LJ sigma (Å) default {default_sig}: ').strip()
epsilon = float(eps_in) if eps_in else default_eps
sigma = float(sig_in) if sig_in else default_sig
parameters_pair = {"[0 0]": {"epsilon": epsilon, "sigma": sigma}}

steps_in = input('Number of MD steps (default 1000): ').strip()
num_steps = int(steps_in) if steps_in else 1000
temp_in = input('Temperature (K or K1,K2; default 94.4): ').strip()
if temp_in:
    if ',' in temp_in:
        t1,t2 = [float(x) for x in temp_in.split(',')[:2]]
    else:
        t1 = float(temp_in); t2 = t1
else:
    t1 = t2 = 94.4
min_in = input('Energy minimization before run? (y/N): ').strip().lower()
need_min = (min_in == 'y')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
molecular = AtomFileReader(filename=xyz_path, box_length=box_length, cutoff=cutoff, device=device, parameter=parameters_pair, skin_thickness=3.0)
force_field = LennardJonesForce(molecular)
bone = SumBackboneInterface([force_field], molecular)
vi = VerletIntegrator(molecular=molecular, dt=0.001, force_field=force_field, ensemble='NVT', temperature=[t1,t2], gamma=1000.0)
model = BaseModel(bone, vi, molecular)
sim = MDSimulator(model, num_steps=num_steps, print_interval=max(1, num_steps//100), save_to_graph_dataset=False)
sim.run(enable_minimize_energy=need_min)
sim.summarize_profile()
print('LJ simulation finished.')
"""

EAM_SCRIPT_SKELETON = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
# EAM simulation script (interactive + defaults)
import torch
from io_utils.reader import AtomFileReader
from io_utils.eam_parser import EAMParser
from core.force.eam_force_cu import EAMForceCUDAExt as EAMForce
from core.md_model import SumBackboneInterface, BaseModel
from core.integrator.integrator import VerletIntegrator
from core.md_simulation import MDSimulator

DEFAULT_POT = 'run_data/WRe_YC2.eam.fs'
DEFAULT_XYZ = 'run_data/W_bcc_stable.xyz'

eam_path = input(f'EAM potential file (default {DEFAULT_POT}): ').strip() or DEFAULT_POT
xyz_path = input(f'Structure file path (default {DEFAULT_XYZ}): ').strip() or DEFAULT_XYZ
box_in = input('Box length (Å) (default 9.6): ').strip()
box_length = float(box_in) if box_in else 9.6
cut_in = input('Cutoff (Å) (default 6.0): ').strip()
cutoff = float(cut_in) if cut_in else 6.0
steps_in = input('Number of MD steps (default 2000): ').strip()
num_steps = int(steps_in) if steps_in else 2000
temp_in = input('Temperature (K or K1,K2; default 300): ').strip()
if temp_in:
    if ',' in temp_in:
        t1,t2 = [float(x) for x in temp_in.split(',')[:2]]
    else:
        t1 = t2 = float(temp_in)
else:
    t1 = t2 = 300.0
min_in = input('Energy minimization before run? (y/N): ').strip().lower()
need_min = (min_in == 'y')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = EAMParser(filepath=eam_path, device=device)
molecular = AtomFileReader(filename=xyz_path, box_length=box_length, cutoff=cutoff, device=device, is_fs=True, skin_thickness=3.0)
force_field = EAMForce(eam_parser=parser, molecular=molecular)
bone = SumBackboneInterface([force_field], molecular)
vi = VerletIntegrator(molecular=molecular, dt=0.001, force_field=force_field, ensemble='NVT', temperature=[t1,t2], gamma=1000.0)
model = BaseModel(bone, vi, molecular)
sim = MDSimulator(model, num_steps=num_steps, print_interval=max(1, num_steps//100), save_to_graph_dataset=False)
sim.run(enable_minimize_energy=need_min)
sim.summarize_profile()
print('EAM simulation finished.')
"""

# MLPS example (CHGNet)
MLPS_SCRIPT_EXAMPLE = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Machine Learning Potential (CHGNet) script (interactive + finetune/load)
import torch, os, time
from io_utils.reader import AtomFileReader
from core.md_model import SumBackboneInterface, BaseModel
from core.integrator.integrator import VerletIntegrator
from core.md_simulation import MDSimulator
from machine_learning_potentials.machine_learning_force import MachineLearningForce

DEFAULT_XYZ = 'run_data/Ar1000.xyz'

def ask_float(prompt, default):
    s = input(f"{prompt} (default {default}): ").strip()
    try: return float(s) if s else float(default)
    except: return float(default)

def ask_int(prompt, default):
    s = input(f"{prompt} (default {default}): ").strip()
    try: return int(s) if s else int(default)
    except: return int(default)

xyz_path = input(f'Structure file path (default {DEFAULT_XYZ}): ').strip() or DEFAULT_XYZ
box_length = ask_float('Box length (Å)', 1550.0)
cutoff = ask_float('Cutoff (Å)', 7.0)
steps = ask_int('Number of MD steps', 1000)

Tin = input('Temperature (K or K1,K2; default 300): ').strip()
if Tin:
    if ',' in Tin:
        T1,T2 = [float(x) for x in Tin.split(',')[:2]]
    else:
        T1 = T2 = float(Tin)
else:
    T1 = T2 = 300.0

mode = input('Mode: [1] Finetune; [2] Load pretrained (default 2): ').strip() or '2'
aimd_pos = ''
aimd_force = ''
model_path = ''
if mode == '1':
    aimd_pos = input('AIMD positions xyz path: ').strip()
    aimd_force = input('AIMD forces xyz path: ').strip()
else:
    model_path = input('Pretrained model path: ').strip()

# Optional training params
epochs = ask_int('Training epochs', 10)
lr = ask_float('Learning rate', 0.002)
batch = ask_int('Batch size', 8)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
molecular = AtomFileReader(filename=xyz_path, box_length=box_length, cutoff=cutoff, device=device, is_mlp=True, skin_thickness=3.0)

mlps_params = {'epochs': epochs, 'learning_rate': lr, 'batch_size': batch, 'targets': 'ef', 'optimizer': 'Adam', 'scheduler': 'CosLR', 'criterion': 'MSE', 'print_freq': 6}
force = MachineLearningForce(molecular=molecular, aimd_pos_file=aimd_pos or 'TODO_POS', aimd_force_file=aimd_force or 'TODO_FORCE', mlp_model_name='chgnet', mlps_finetune_params=mlps_params if mode=='1' else None, mlps_model_path=model_path if mode!='1' else None)

bone = SumBackboneInterface([force], molecular)
vi = VerletIntegrator(molecular=molecular, dt=0.001, force_field=force, ensemble='NVT', temperature=[T1,T2], gamma=1000.0)
model = BaseModel(bone, vi, molecular)
sim = MDSimulator(model, num_steps=steps, print_interval=max(1, steps//100), save_to_graph_dataset=False)
sim.run(enable_minimize_energy=False)
sim.summarize_profile()
print('MLPS simulation finished.')
"""

LLM_SYSTEM_INSTRUCTION = """You are an MD script generation assistant. Generate a complete runnable Python script from natural language. Requirements:
A. The script must start with: #!/usr/bin/env python then a UTF-8 encoding comment.
B. Include an interactive input section: ask for structure path, box length, cutoff, steps, temperature, whether to minimize first. For LJ also ask epsilon/sigma; for EAM ask potential file; for MLPS ask AIMD data or model path.
C. If inputs are empty, use reasonable defaults or infer based on known elements (Ar -> epsilon 0.0104, sigma 3.405).
D. If not enough information and cannot infer, add TODO comments.
E. Keep variable names clear, print a summary and call sim.summarize_profile() at the end.
F. Only use project APIs: AtomFileReader, LennardJonesForce or EAMForceCUDAExt or MachineLearningForce, SumBackboneInterface, VerletIntegrator, BaseModel, MDSimulator.
G. Output only the script body, no explanations.
The following are three examples following the new spec:
""" + "\n===== LJ Example =====\n" + LJ_SCRIPT_EXAMPLE + "\n===== EAM Example =====\n" + EAM_SCRIPT_SKELETON + "\n===== MLPS Example =====\n" + MLPS_SCRIPT_EXAMPLE + "\n"

# ============ Post-run analysis ============
class OutputAnalyzer:
    def __init__(self, out_dir: str):
        self.dir = Path(out_dir)
        self.paths: Dict[str, Path] = {
            'energies': self.dir / 'energies.csv',
            'traj': self.dir / 'traj.xyz',
            'forces': self.dir / 'forces',
        }
        self.summary: Dict[str, Any] = {}
        self.analysis_dir = self.dir / 'analysis'
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def _load_energies(self) -> Optional[Dict[str, Any]]:
        p = self.paths['energies']
        if not p.exists():
            return None
        steps = []
        pot = []
        kin = []
        tot = []
        temp = []
        with open(p, 'r', encoding='utf-8') as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if len(row) < 5:
                    continue
                try:
                    steps.append(int(row[0]))
                    pot.append(float(row[1]))
                    kin.append(float(row[2]))
                    tot.append(float(row[3]))
                    temp.append(float(row[4]))
                except Exception:
                    continue
        if not steps:
            return None
        if np is None:
            summary = {
                'steps': len(steps),
                'E_total_first': tot[0],
                'E_total_last': tot[-1],
                'T_mean': sum(temp)/len(temp),
            }
        else:
            steps_np = np.asarray(steps)
            pot_np = np.asarray(pot)
            kin_np = np.asarray(kin)
            tot_np = np.asarray(tot)
            temp_np = np.asarray(temp)
            slope_tot = float(np.polyfit(steps_np, tot_np, 1)[0]) if len(steps_np) > 1 else 0.0
            summary = {
                'steps': int(steps_np.size),
                'E_pot_mean': float(pot_np.mean()),
                'E_pot_std': float(pot_np.std()),
                'E_kin_mean': float(kin_np.mean()),
                'E_tot_mean': float(tot_np.mean()),
                'E_tot_std': float(tot_np.std()),
                'E_tot_drift_per_step': slope_tot,
                'T_mean': float(temp_np.mean()),
                'T_std': float(temp_np.std()),
                'E_total_first': float(tot_np[0]),
                'E_total_last': float(tot_np[-1]),
            }
        self.summary['energies'] = summary
        (self.analysis_dir / 'energies_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return summary

    def _count_traj_frames(self) -> Optional[int]:
        p = self.paths['traj']
        if not p.exists():
            return None
        frames = 0
        try:
            with open(p, 'r', encoding='utf-8') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    try:
                        n = int(line)
                    except Exception:
                        break
                    _ = f.readline()
                    for _ in range(n):
                        if not f.readline():
                            break
                    frames += 1
        except Exception:
            return None
        self.summary['traj_frames'] = frames
        return frames

    def _forces_files_count(self) -> Optional[int]:
        d = self.paths['forces']
        if not d.exists() or not d.is_dir():
            return None
        cnt = len([p for p in d.glob('forces_*.pt')])
        self.summary['forces_files'] = cnt
        return cnt

    def analyze(self) -> Dict[str, Any]:
        res = {}
        res['energies'] = self._load_energies()
        res['traj_frames'] = self._count_traj_frames()
        res['forces_files'] = self._forces_files_count()
        lines = [
            '# Analysis Report',
            f'- Output dir: {self.dir}',
        ]
        if res['energies']:
            e = res['energies']
            lines += [
                '## Energies',
                f"Steps: {e.get('steps')}\n",
                f"E_tot_mean: {e.get('E_tot_mean', 'NA')}",
                f"E_tot_std: {e.get('E_tot_std', 'NA')}",
                f"E_tot_drift_per_step: {e.get('E_tot_drift_per_step', 'NA')}",
                f"T_mean: {e.get('T_mean', 'NA')}  T_std: {e.get('T_std', 'NA')}\n",
            ]
        if res['traj_frames'] is not None:
            lines += [f"## Trajectory\nFrames: {res['traj_frames']}\n"]
        if res['forces_files'] is not None:
            lines += [f"## Forces\nFiles: {res['forces_files']}\n"]
        (self.analysis_dir / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
        return res

    def _iter_xyz_frames(self, stride_frames: int = 1, max_frames: Optional[int] = None):
        p = self.paths['traj']
        if not p.exists():
            raise FileNotFoundError(f'Trajectory file not found: {p}')
        yielded = 0
        with open(p, 'r', encoding='utf-8') as f:
            frame_idx = 0
            while True:
                n_line = f.readline()
                if not n_line:
                    break
                n_line = n_line.strip()
                try:
                    n = int(n_line)
                except Exception:
                    break
                _ = f.readline()
                coords = []
                types = []
                for _i in range(n):
                    line = f.readline()
                    if not line:
                        break
                    sp = line.split()
                    if len(sp) >= 4:
                        types.append(sp[0])
                        try:
                            x, y, z = float(sp[1]), float(sp[2]), float(sp[3])
                        except Exception:
                            x = y = z = 0.0
                        coords.append((x, y, z))
                if frame_idx % max(1, stride_frames) == 0:
                    if np is None:
                        arr = coords
                    else:
                        arr = np.asarray(coords, dtype=float)
                    yield arr, types
                    yielded += 1
                    if (max_frames is not None) and (yielded >= max_frames):
                        break
                frame_idx += 1

    def generate_gif(self, gif_path: Optional[str] = None, stride_frames: int = 5, max_frames: int = 200, fps: int = 20,
                     dpi: int = 100, figsize=(6, 6), point_size: float = 5.0, color: str = 'dodgerblue') -> str:
        from matplotlib import pyplot as plt
        from matplotlib import animation as mpl_anim
        if np is None:
            raise RuntimeError('numpy is required to generate GIF. Please install: pip install numpy')
        frames = []
        types_list = []
        for arr, types in self._iter_xyz_frames(stride_frames, max_frames):
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr, dtype=float)
            frames.append(arr)
            types_list.append(types)
        if not frames:
            raise RuntimeError('Empty trajectory, cannot generate GIF.')
        xs = np.concatenate([f[:, 0] for f in frames])
        ys = np.concatenate([f[:, 1] for f in frames])
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        mx = max(1e-6, x_max - x_min, y_max - y_min)
        pad = 0.05 * mx
        xlim = (x_min - pad, x_max + pad)
        ylim = (y_min - pad, y_max + pad)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        sc = ax.scatter(frames[0][:, 0], frames[0][:, 1], s=point_size, c=color, edgecolors='none')
        ax.set_title('MD Trajectory (XY)')
        def init():
            sc.set_offsets(frames[0][:, :2])
            return (sc,)
        def update(i):
            sc.set_offsets(frames[i][:, :2])
            ax.set_title(f'MD Trajectory (frame {i+1}/{len(frames)})')
            return (sc,)
        anim = mpl_anim.FuncAnimation(fig, update, init_func=init, frames=len(frames), interval=1000/max(1, fps), blit=False)
        out_path = Path(gif_path) if gif_path else (self.analysis_dir / 'trajectory_xy.gif')
        try:
            anim.save(str(out_path), writer='pillow', fps=fps)
        except Exception as e:
            plt.close(fig)
            raise RuntimeError(f'Failed to save GIF. Perhaps pillow is missing: {e}\nTry: pip install pillow')
        plt.close(fig)
        return str(out_path)

# Interactive: analyze outputs in a directory
def analyze_outputs_cli():
    out_dir = input('Please input the output directory (containing energies.csv / traj.xyz): ').strip()
    if not out_dir:
        print('No output directory provided.')
        return
    an = OutputAnalyzer(out_dir)
    res = an.analyze()
    print('Analysis done. Summary:')
    print(json.dumps(an.summary, indent=2, ensure_ascii=False))
    print(f"Report: {an.analysis_dir / 'report.md'}")

# Interactive: Q&A based on outputs (optional LLM)
def qna_cli():
    out_dir = input('Please input the output directory for Q&A: ').strip()
    if not out_dir:
        print('No output directory provided.')
        return
    an = OutputAnalyzer(out_dir)
    an.analyze()
    context = json.dumps(an.summary, ensure_ascii=False)
    print('You can ask questions based on the following JSON context. Exit by entering empty line or q/quit/exit/:q/back.')
    print(context)
    use_llm = False
    if ChatOpenAI is not None:
        use_llm = (input('Use LLM to answer? (y/N): ').strip().lower() == 'y')
    llm = None
    if use_llm:
        try:
            api_key = os.environ.get('SIMULON_LLM_API_KEY') or os.environ.get('OPENAI_API_KEY') or input('Please enter LLM API Key: ').strip()
            if not api_key:
                print('No API Key provided, falling back to local answering.')
                use_llm = False
            else:
                base_url = input('Base URL (default https://api.deepseek.com/v1): ').strip() or 'https://api.deepseek.com/v1'
                model_name = input('Model name (default deepseek-reasoner): ').strip() or 'deepseek-reasoner'
                llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url) if ChatOpenAI else None
        except EOFError:
            use_llm = False
    while True:
        try:
            q = input('\nQ: ').strip()
        except EOFError:
            print('Back to menu.')
            break
        if not q or q.lower() in ('q', 'quit', 'exit', ':q', 'back'):
            print('Back to menu.')
            break
        if use_llm and llm is not None:
            from langchain_core.messages import SystemMessage, HumanMessage
            sys_msg = SystemMessage(content=f"You are an assistant analyzing MD outputs. Context JSON:\n{context}\nAnswer based on data and general knowledge. If data is missing, say so.")
            user_msg = HumanMessage(content=q)
            try:
                ans = llm.invoke([sys_msg, user_msg]).content
            except Exception as e:
                ans = f"LLM failed: {e}"
            print(f"A: {ans}")
        else:
            ql = q.lower()
            if 'step' in ql:
                print(f"A: Steps (energies): {an.summary.get('energies',{}).get('steps','unknown')}")
            elif 'temperature' in ql or 'temp' in ql:
                print(f"A: Mean temperature: {an.summary.get('energies',{}).get('T_mean','NA')}")
            elif 'drift' in ql or 'energy drift' in ql:
                print(f"A: Total energy drift per step: {an.summary.get('energies',{}).get('E_tot_drift_per_step','NA')}")
            else:
                print('A: Local answer unavailable. Enable LLM or ask a more specific question.')

class ScriptLLMAgent:
    def __init__(self, model_name: str = 'deepseek-reasoner', temperature: float = 0.2, api_key: str | None = None, base_url: str | None = None):
        if ChatOpenAI is None:
            raise RuntimeError('langchain_openai ChatOpenAI not installed or import failed.')
        kwargs = dict(model=model_name, temperature=temperature)
        if api_key: kwargs['api_key'] = api_key
        if base_url: kwargs['base_url'] = base_url
        self.llm = ChatOpenAI(**kwargs)

    def build_prompt(self, user_text: str) -> list:
        examples = f"\n===== LJ Example =====\n{LJ_SCRIPT_EXAMPLE}\n===== EAM Skeleton =====\n{EAM_SCRIPT_SKELETON}\n===== MLPS Example =====\n{MLPS_SCRIPT_EXAMPLE}\n"
        messages = [
            ("system", LLM_SYSTEM_INSTRUCTION + examples),
            ("user", user_text.strip())
        ]
        return messages

    def generate(self, user_text: str) -> str:
        messages = self.build_prompt(user_text)
        from langchain_core.messages import SystemMessage, HumanMessage
        lc_msgs = []
        for role, content in messages:
            if role == 'system':
                lc_msgs.append(SystemMessage(content=content))
            else:
                lc_msgs.append(HumanMessage(content=content))
        resp = self.llm.invoke(lc_msgs)
        code = resp.content
        if '```' in code:
            parts = code.split('```')
            picked = None
            for seg in parts:
                if 'import torch' in seg:
                    picked = seg
                    break
            if picked:
                code = picked
        return code.strip() + '\n'

def save_generated_script(code: str, base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = base_dir / f'llm_gen_md_{ts}.py'
    path.write_text(code, encoding='utf-8')
    return path

# Interactive: MLPS run/finetune and simulate
def mlps_cli():
    import time
    try:
        import torch
    except Exception as e:
        print('PyTorch is required to run MLPS:', e)
        return
    from io_utils.reader import AtomFileReader
    from core.md_model import SumBackboneInterface, BaseModel
    from core.integrator.integrator import VerletIntegrator
    from core.md_simulation import MDSimulator
    from machine_learning_potentials.machine_learning_force import MachineLearningForce

    def _ask_float(prompt, default):
        s = input(f"{prompt} (default {default}): ").strip()
        try: return float(s) if s else float(default)
        except: return float(default)
    def _ask_int(prompt, default):
        s = input(f"{prompt} (default {default}): ").strip()
        try: return int(s) if s else int(default)
        except: return int(default)

    print('— MLPS (CHGNet) interactive run —')
    xyz_path = input('Structure xyz path (default run_data/Ar1000.xyz): ').strip() or 'run_data/Ar1000.xyz'
    box_length = _ask_float('Box length (Å)', 1550.0)
    cutoff = _ask_float('Cutoff (Å)', 7.0)
    steps = _ask_int('Number of MD steps', 1000)
    dt = _ask_float('Time step dt', 0.001)
    Tin = input('Temperature (K or K1,K2; default 300): ').strip()
    if Tin:
        if ',' in Tin:
            T1,T2 = [float(x) for x in Tin.split(',')[:2]]
        else:
            T1 = T2 = float(Tin)
    else:
        T1 = T2 = 300.0

    mode = input('Mode: [1] Finetune; [2] Load pretrained (default 2): ').strip() or '2'
    aimd_pos = aimd_force = model_path = ''
    if mode == '1':
        aimd_pos = input('AIMD positions xyz: ').strip()
        aimd_force = input('AIMD forces xyz: ').strip()
        if not (aimd_pos and aimd_force):
            print('AIMD data missing, cannot finetune. Back to menu.')
            return
    else:
        model_path = input('Model path: ').strip()
        if not model_path:
            print('Model path missing. Switching to finetune mode.')
            mode = '1'
            aimd_pos = input('AIMD positions xyz: ').strip()
            aimd_force = input('AIMD forces xyz: ').strip()
            if not (aimd_pos and aimd_force):
                print('AIMD data missing. Back to menu.')
                return

    epochs = _ask_int('Training epochs', 10)
    lr = _ask_float('Learning rate', 0.002)
    batch = _ask_int('Batch size', 8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mol = AtomFileReader(filename=xyz_path, box_length=box_length, cutoff=cutoff, device=device, is_mlp=True, skin_thickness=3.0)

    mlps_params = {'epochs': epochs, 'learning_rate': lr, 'batch_size': batch, 'targets': 'ef', 'optimizer': 'Adam', 'scheduler': 'CosLR', 'criterion': 'MSE', 'print_freq': 6}
    force = MachineLearningForce(
        molecular=mol,
        aimd_pos_file=aimd_pos or 'TODO_POS',
        aimd_force_file=aimd_force or 'TODO_FORCE',
        mlp_model_name='chgnet',
        mlps_finetune_params=mlps_params if mode=='1' else None,
        mlps_model_path=model_path if mode!='1' else None
    )

    bone = SumBackboneInterface([force], mol)
    vi = VerletIntegrator(molecular=mol, dt=dt, force_field=bone, ensemble='NVT', temperature=[T1,T2], gamma=1000.0)
    model = BaseModel(bone, vi, mol)
    sim = MDSimulator(model, num_steps=steps, print_interval=max(1, steps//100), save_to_graph_dataset=False)
    sim.run(enable_minimize_energy=False)
    sim.summarize_profile()

    # Save basic outputs
    now_time = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
    out_dir = Path('output')
    out_dir.mkdir(exist_ok=True)
    energy_path = out_dir / f"MD_energy_curve_{now_time}.png"
    sim.save_energy_curve(str(energy_path))
    traj_path = out_dir / f"MD_traj_{now_time}.xyz"
    sim.save_xyz_trajectory(str(traj_path), atom_types=mol.atom_types)
    force_path = out_dir / f"forces_{now_time}.xyz"
    sim.save_forces_grad(str(force_path), with_no_ele=True, atom_types=mol.atom_types)
    print('Saved:')
    print('  Energies:', energy_path)
    print('  Trajectory:', traj_path)
    print('  Forces:', force_path)


def main():
    print('=== Simulon Agent ===')
    if ChatOpenAI is None:
        pass
    while True:
        print('\n1) Generate MD script (LLM)')
        print('2) Analyze outputs (from output directory)')
        print('3) Q&A based on outputs (optional LLM)')
        print('4) Generate GIF (from traj.xyz)')
        print('5) Run MLPS (CHGNet)')
        print('6) Exit')
        try:
            choice = input('Please choose (1/2/3/4/5/6): ').strip()
        except EOFError:
            choice = '6'
        if choice == '2':
            analyze_outputs_cli()
            continue
        if choice == '3':
            qna_cli()
            continue
        if choice == '4':
            fn = globals().get('make_gif_cli')
            if callable(fn):
                fn()
            else:
                print('GIF feature unavailable.')
            continue
        if choice == '5':
            mlps_cli()
            continue
        if choice == '6' or choice.lower() in ('q', 'quit', 'exit'):
            print('Exited.')
            break
        if choice != '1':
            print('Invalid choice, try again.')
            continue
        # Generate MD script (LLM)
        if ChatOpenAI is None:
            print('ChatOpenAI not detected. Install with: pip install langchain-openai')
            continue
        try:
            api_key_env = os.environ.get('SIMULON_LLM_API_KEY') or os.environ.get('OPENAI_API_KEY')
            if api_key_env:
                use_env = input('Detected API Key in environment. Use it? (Y/n): ').strip().lower()
                if use_env in ('', 'y', 'yes'): api_key = api_key_env
                else: api_key = input('Please input LLM API Key: ').strip()
            else:
                api_key = input('Please input LLM API Key: ').strip()
            if not api_key:
                print('API Key not provided, back to menu.')
                continue
            base_url = input('Please input Base URL (Default https://api.deepseek.com/v1): ').strip() or 'https://api.deepseek.com/v1'
            model_name = input('Please input Model Name (Default deepseek-reasoner): ').strip() or 'deepseek-reasoner'
        except EOFError:
            print('Input interrupted, back to menu.')
            continue
        try:
            user_text = input('Describe your needs (e.g., "LJ Ar10000 5000 steps at 94.4 K and minimize first"):\n> ').strip()
        except EOFError:
            user_text = ''
        if not user_text:
            print('No input, back to menu')
            continue
        agent = ScriptLLMAgent(model_name=model_name, api_key=api_key, base_url=base_url)
        print('[LLM] Generating script...')
        code = agent.generate(user_text)
        out_dir = Path(__file__).resolve().parents[1] / 'run_scripts'
        script_path = save_generated_script(code, out_dir)
        print(f'[LLM] Script saved: {script_path}')
        try:
            run_now = input('Run this script now? (y/N): ').strip().lower()
        except EOFError:
            run_now = 'n'
        if run_now == 'y':
            print('[Runner] Executing...')
            try:
                scope = {}
                exec(compile(code, str(script_path), 'exec'), scope, scope)
                print('[Runner] Done')
            except Exception as e:
                print(f'[Runner] Failed: {e}')
        else:
            print('You can run manually: python', script_path)

# Interactive: generate GIF from traj.xyz
def make_gif_cli():
    out_dir = input('Please input the output directory (contains traj.xyz): ').strip()
    if not out_dir:
        print('No output directory provided.')
        return
    try:
        stride = int(input('Frame stride (default 5): ').strip() or '5')
        maxf = int(input('Max frames (default 200): ').strip() or '200')
        fps = int(input('GIF fps (default 20): ').strip() or '20')
    except Exception:
        stride, maxf, fps = 5, 200, 20
    an = OutputAnalyzer(out_dir)
    try:
        path = an.generate_gif(stride_frames=stride, max_frames=maxf, fps=fps)
        print('GIF saved:', path)
    except Exception as e:
        print('Failed to generate:', e)

if __name__ == "__main__":
    main()
