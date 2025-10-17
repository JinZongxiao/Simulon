# -*- coding: utf-8 -*-
import os
from pathlib import Path
import datetime

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

# 新增与修复导入
import csv
import json
from typing import Dict, Any, Optional
try:
    import numpy as np
except Exception:
    np = None


LJ_SCRIPT_EXAMPLE = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 示例: Lennard-Jones 系统脚本 (交互 + 自动推断)
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

xyz_path = input(f'结构文件路径 (默认 {DEFAULT_XYZ}): ').strip() or DEFAULT_XYZ
box_input = input('盒长(Å) (留空自动/使用 1550.0): ').strip()
box_length = float(box_input) if box_input else 1550.0
cutoff_input = input('截断距离(Å) (默认 7.0): ').strip()
cutoff = float(cutoff_input) if cutoff_input else 7.0
# 力场参数交互
atom_types = detect_atom_types(xyz_path)
print(f'Detected atom types: {sorted(atom_types) if atom_types else "(未识别)"}')
if atom_types and atom_types == {'Ar'}:
    default_eps, default_sig = DEFAULT_EPS, DEFAULT_SIG
else:
    default_eps = DEFAULT_EPS
    default_sig = DEFAULT_SIG

eps_in = input(f'LJ epsilon (eV) 默认 {default_eps}: ').strip()
sig_in = input(f'LJ sigma (Å) 默认 {default_sig}: ').strip()
epsilon = float(eps_in) if eps_in else default_eps
sigma = float(sig_in) if sig_in else default_sig
parameters_pair = {"[0 0]": {"epsilon": epsilon, "sigma": sigma}}

steps_in = input('模拟步数 (默认 1000): ').strip()
num_steps = int(steps_in) if steps_in else 1000
temp_in = input('温度(K 或 K1,K2; 默认 94.4): ').strip()
if temp_in:
    if ',' in temp_in:
        t1,t2 = [float(x) for x in temp_in.split(',')[:2]]
    else:
        t1 = float(temp_in); t2 = t1
else:
    t1 = t2 = 94.4
min_in = input('是否先最小化? (y/N): ').strip().lower()
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
print('完成 LJ 模拟。')
"""

EAM_SCRIPT_SKELETON = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
# EAM 模拟脚本 (交互 + 默认)
import torch
from io_utils.reader import AtomFileReader
from io_utils.eam_parser import EAMParser
from core.force.eam_force_cu import EAMForceCUDAExt as EAMForce
from core.md_model import SumBackboneInterface, BaseModel
from core.integrator.integrator import VerletIntegrator
from core.md_simulation import MDSimulator

DEFAULT_POT = 'run_data/WRe_YC2.eam.fs'
DEFAULT_XYZ = 'run_data/W_bcc_stable.xyz'

eam_path = input(f'EAM 势文件路径 (默认 {DEFAULT_POT}): ').strip() or DEFAULT_POT
xyz_path = input(f'结构文件路径 (默认 {DEFAULT_XYZ}): ').strip() or DEFAULT_XYZ
box_in = input('盒长(Å) (默认 9.6): ').strip()
box_length = float(box_in) if box_in else 9.6
cut_in = input('截断距离(Å) (默认 6.0): ').strip()
cutoff = float(cut_in) if cut_in else 6.0
steps_in = input('模拟步数 (默认 2000): ').strip()
num_steps = int(steps_in) if steps_in else 2000
temp_in = input('温度(K 或 K1,K2; 默认 300): ').strip()
if temp_in:
    if ',' in temp_in:
        t1,t2 = [float(x) for x in temp_in.split(',')[:2]]
    else:
        t1 = t2 = float(temp_in)
else:
    t1 = t2 = 300.0
min_in = input('是否先最小化? (y/N): ').strip().lower()
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
print('完成 EAM 模拟。')
"""

# 新增：MLPS 示例（基于 MachineLearningForce / CHGNet）
MLPS_SCRIPT_EXAMPLE = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 机器学习势 (CHGNet) 脚本 (交互 + 可微调/加载)
import torch, os, time
from io_utils.reader import AtomFileReader
from core.md_model import SumBackboneInterface, BaseModel
from core.integrator.integrator import VerletIntegrator
from core.md_simulation import MDSimulator
from machine_learning_potentials.machine_learning_force import MachineLearningForce

DEFAULT_XYZ = 'run_data/Ar1000.xyz'

def ask_float(prompt, default):
    s = input(f"{prompt} (默认 {default}): ").strip()
    try: return float(s) if s else float(default)
    except: return float(default)

def ask_int(prompt, default):
    s = input(f"{prompt} (默认 {default}): ").strip()
    try: return int(s) if s else int(default)
    except: return int(default)

xyz_path = input(f'结构文件路径 (默认 {DEFAULT_XYZ}): ').strip() or DEFAULT_XYZ
box_length = ask_float('盒长(Å)', 1550.0)
cutoff = ask_float('截断距离(Å)', 7.0)
steps = ask_int('模拟步数', 1000)

Tin = input('温度(K 或 K1,K2; 默认 300): ').strip()
if Tin:
    if ',' in Tin:
        T1,T2 = [float(x) for x in Tin.split(',')[:2]]
    else:
        T1 = T2 = float(Tin)
else:
    T1 = T2 = 300.0

mode = input('选择模式: [1] 微调; [2] 直接加载模型 (默认 2): ').strip() or '2'
aimd_pos = ''
aimd_force = ''
model_path = ''
if mode == '1':
    aimd_pos = input('AIMD 位置 xyz 路径: ').strip()
    aimd_force = input('AIMD 力 xyz 路径: ').strip()
else:
    model_path = input('已训练模型路径: ').strip()

# 可选训练参数
epochs = ask_int('训练 epochs', 10)
lr = ask_float('学习率', 0.002)
batch = ask_int('batch_size', 8)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
molecular = AtomFileReader(filename=xyz_path, box_length=box_length, cutoff=cutoff, device=device, is_mlp=True, skin_thickness=3.0)

mlps_params = {'epochs': epochs, 'learning_rate': lr, 'batch_size': batch, 'targets': 'ef', 'optimizer': 'Adam', 'scheduler': 'CosLR', 'criterion': 'MSE', 'print_freq': 6}
force = MachineLearningForce(molecular=molecular, aimd_pos_file=aimd_pos or 'TODO_POS', aimd_force_file=aimd_force or 'TODO_FORCE', mlp_model_name='chgnet', mlps_finetune_params=mlps_params if mode=='1' else None, mlps_model_path=model_path if mode!='1' else None)

bone = SumBackboneInterface([force], molecular)
vi = VerletIntegrator(molecular=molecular, dt=0.001, force_field=force_field, ensemble='NVT', temperature=[T1,T2], gamma=1000.0)
model = BaseModel(bone, vi, molecular)
sim = MDSimulator(model, num_steps=steps, print_interval=max(1, steps//100), save_to_graph_dataset=False)
sim.run(enable_minimize_energy=False)
sim.summarize_profile()
print('完成 MLPS 模拟。')
"""

LLM_SYSTEM_INSTRUCTION = """你是一个分子动力学脚本生成助手。根据用户自然语言指令生成一个完整可运行的 Python 脚本。新增要求:
A. 脚本首行必须: #!/usr/bin/env python 其后加 UTF-8 编码注释。
B. 脚本包含交互输入段: 询问结构文件路径、盒长、截断、步数、温度、是否最小化。LJ 还需 epsilon/sigma；EAM 需势文件路径；MLPS 需 AIMD 数据或模型路径。
C. 用户输入留空则使用合理默认或根据已知元素自动推断 (Ar -> epsilon 0.0104, sigma 3.405)。
D. 若信息不足且无法推断，用 TODO 注释提示。
E. 保持变量命名清晰，最后打印总结并调用 sim.summarize_profile()。
F. 只能使用本项目已有 API：AtomFileReader, LennardJonesForce 或 EAMForceCUDAExt 或 MachineLearningForce, SumBackboneInterface, VerletIntegrator, BaseModel, MDSimulator。
G. 不输出解释文字，只输出脚本本体。
下面提供三个遵循新规范的示例：
""" + "\n===== LJ 示例 =====\n" + LJ_SCRIPT_EXAMPLE + "\n===== EAM 示例 =====\n" + EAM_SCRIPT_SKELETON + "\n===== MLPS 示例 =====\n" + MLPS_SCRIPT_EXAMPLE + "\n"

# ============ 输出后分析 ============
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
            # 仅统计简单信息
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
            # 线性漂移（每步）
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
        # 保存一份 JSON
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
                    _ = f.readline()  # 注释
                    # 跳过 n 行
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
        # 文本报告
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

    # 读取 XYZ 轨迹（按帧生成器）
    def _iter_xyz_frames(self, stride_frames: int = 1, max_frames: Optional[int] = None):
        p = self.paths['traj']
        if not p.exists():
            raise FileNotFoundError(f'未找到轨迹文件: {p}')
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
                _ = f.readline()  # 注释行
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
                        arr = coords  # list of tuples
                    else:
                        arr = np.asarray(coords, dtype=float)
                    yield arr, types
                    yielded += 1
                    if (max_frames is not None) and (yielded >= max_frames):
                        break
                frame_idx += 1

    # 生成 GIF 动图（xy 投影）
    def generate_gif(self, gif_path: Optional[str] = None, stride_frames: int = 5, max_frames: int = 200, fps: int = 20,
                     dpi: int = 100, figsize=(6, 6), point_size: float = 5.0, color: str = 'dodgerblue') -> str:
        from matplotlib import pyplot as plt
        from matplotlib import animation as mpl_anim
        if np is None:
            raise RuntimeError('需要 numpy 才能生成 GIF，请先安装：pip install numpy')
        frames = []
        types_list = []
        for arr, types in self._iter_xyz_frames(stride_frames, max_frames):
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr, dtype=float)
            frames.append(arr)
            types_list.append(types)
        if not frames:
            raise RuntimeError('轨迹为空，无法生成 GIF。')
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
            raise RuntimeError(f'保存 GIF 失败，可能缺少 pillow：{e}\n请尝试: pip install pillow')
        plt.close(fig)
        return str(out_path)

# 交互式：基于输出目录做后分析
def analyze_outputs_cli():
    out_dir = input('请输入模拟输出目录 (包含 energies.csv / traj.xyz 等): ').strip()
    if not out_dir:
        print('未提供输出目录。')
        return
    an = OutputAnalyzer(out_dir)
    res = an.analyze()
    print('分析完成。概要:')
    print(json.dumps(an.summary, indent=2, ensure_ascii=False))
    print(f"报告: {an.analysis_dir / 'report.md'}")

# 交互式：就输出目录的结果进行问答（可选用 LLM）
def qna_cli():
    out_dir = input('请输入模拟输出目录以便问答: ').strip()
    if not out_dir:
        print('未提供输出目录。')
        return
    an = OutputAnalyzer(out_dir)
    an.analyze()
    context = json.dumps(an.summary, ensure_ascii=False)
    print('可就以下上下文提问。退出方式：输入空行或 q/quit/exit/:q/back 返回菜单。')
    print(context)
    use_llm = False
    if ChatOpenAI is not None:
        use_llm = (input('使用 LLM 回答? (y/N): ').strip().lower() == 'y')
    llm = None
    if use_llm:
        try:
            api_key = os.environ.get('SIMULON_LLM_API_KEY') or os.environ.get('OPENAI_API_KEY') or input('请输入 LLM API Key: ').strip()
            if not api_key:
                print('未提供 API Key，改用本地简易回答。')
                use_llm = False
            else:
                base_url = input('请输入 Base URL (默认 https://api.deepseek.com/v1): ').strip() or 'https://api.deepseek.com/v1'
                model_name = input('请输入模型名称 (默认 deepseek-reasoner): ').strip() or 'deepseek-reasoner'
                llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url) if ChatOpenAI else None
        except EOFError:
            use_llm = False
    while True:
        try:
            q = input('\nQ: ').strip()
        except EOFError:
            print('已返回菜单。')
            break
        if not q or q.lower() in ('q', 'quit', 'exit', ':q', 'back'):
            print('已返回菜单。')
            break
        if use_llm and llm is not None:
            from langchain_core.messages import SystemMessage, HumanMessage
            sys_msg = SystemMessage(content=f"你是模拟结果分析助手。以下是上下文 JSON：\n{context}\n请基于这些数据与常识回答问题。若无数据支撑，请说明。")
            user_msg = HumanMessage(content=q)
            try:
                ans = llm.invoke([sys_msg, user_msg]).content
            except Exception as e:
                ans = f"LLM 调用失败: {e}"
            print(f"A: {ans}")
        else:
            # 简易本地回答：从 summary 中检索
            ql = q.lower()
            if 'step' in ql:
                print(f"A: 步数(energies): {an.summary.get('energies',{}).get('steps','未知')}")
            elif '温度' in q or 'temperature' in ql:
                print(f"A: 平均温度: {an.summary.get('energies',{}).get('T_mean','NA')}")
            elif '能量漂移' in q or 'drift' in ql:
                print(f"A: 总能量每步漂移: {an.summary.get('energies',{}).get('E_tot_drift_per_step','NA')}")
            else:
                print('A: 本地无法回答，请启用 LLM 或提供更具体问题。')

class ScriptLLMAgent:
    def __init__(self, model_name: str = 'deepseek-reasoner', temperature: float = 0.2, api_key: str | None = None, base_url: str | None = None):
        if ChatOpenAI is None:
            raise RuntimeError('未安装或无法导入 langchain_openai ChatOpenAI，无法使用 LLM 生成。')
        kwargs = dict(model=model_name, temperature=temperature)
        if api_key: kwargs['api_key'] = api_key
        if base_url: kwargs['base_url'] = base_url
        self.llm = ChatOpenAI(**kwargs)

    def build_prompt(self, user_text: str) -> list:
        examples = f"\n===== LJ 示例 =====\n{LJ_SCRIPT_EXAMPLE}\n===== EAM 骨架 =====\n{EAM_SCRIPT_SKELETON}\n===== MLPS 示例 =====\n{MLPS_SCRIPT_EXAMPLE}\n"
        messages = [
            ("system", LLM_SYSTEM_INSTRUCTION + examples),
            ("user", user_text.strip())
        ]
        return messages

    def generate(self, user_text: str) -> str:
        messages = self.build_prompt(user_text)
        # ChatOpenAI 期望 message 对象列表; 转换
        from langchain_core.messages import SystemMessage, HumanMessage
        lc_msgs = []
        for role, content in messages:
            if role == 'system':
                lc_msgs.append(SystemMessage(content=content))
            else:
                lc_msgs.append(HumanMessage(content=content))
        resp = self.llm.invoke(lc_msgs)
        code = resp.content
        # 简单裁剪: 若模型用了 markdown 代码块
        if '```' in code:
            parts = code.split('```')
            # 取第一个包含 import torch 的块
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

# 新增：交互式 MLPS 运行/微调并模拟
def mlps_cli():
    import time
    try:
        import torch
    except Exception as e:
        print('需要 PyTorch 才能运行 MLPS:', e)
        return
    from io_utils.reader import AtomFileReader
    from core.md_model import SumBackboneInterface, BaseModel
    from core.integrator.integrator import VerletIntegrator
    from core.md_simulation import MDSimulator
    from machine_learning_potentials.machine_learning_force import MachineLearningForce

    def _ask_float(prompt, default):
        s = input(f"{prompt} (默认 {default}): ").strip()
        try: return float(s) if s else float(default)
        except: return float(default)
    def _ask_int(prompt, default):
        s = input(f"{prompt} (默认 {default}): ").strip()
        try: return int(s) if s else int(default)
        except: return int(default)

    print('— MLPS (CHGNet) 交互式运行 —')
    xyz_path = input('结构 xyz 路径 (默认 run_data/Ar1000.xyz): ').strip() or 'run_data/Ar1000.xyz'
    box_length = _ask_float('盒长(Å)', 1550.0)
    cutoff = _ask_float('截断(Å)', 7.0)
    steps = _ask_int('模拟步数', 1000)
    dt = _ask_float('时间步长 dt', 0.001)
    Tin = input('温度(K 或 K1,K2; 默认 300): ').strip()
    if Tin:
        if ',' in Tin:
            T1,T2 = [float(x) for x in Tin.split(',')[:2]]
        else:
            T1 = T2 = float(Tin)
    else:
        T1 = T2 = 300.0

    mode = input('选择模式: [1] 微调; [2] 加载已训练模型 (默认 2): ').strip() or '2'
    aimd_pos = aimd_force = model_path = ''
    if mode == '1':
        aimd_pos = input('AIMD 位置 xyz: ').strip()
        aimd_force = input('AIMD 力 xyz: ').strip()
        if not (aimd_pos and aimd_force):
            print('未提供 AIMD 数据，无法微调。返回菜单。')
            return
    else:
        model_path = input('模型路径: ').strip()
        if not model_path:
            print('未提供模型路径，将尝试微调。')
            mode = '1'
            aimd_pos = input('AIMD 位置 xyz: ').strip()
            aimd_force = input('AIMD 力 xyz: ').strip()
            if not (aimd_pos and aimd_force):
                print('未提供 AIMD 数据，无法继续。返回菜单。')
                return

    epochs = _ask_int('训练 epochs', 10)
    lr = _ask_float('学习率', 0.002)
    batch = _ask_int('batch_size', 8)

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

    # 保存基础输出
    now_time = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
    out_dir = Path('output')
    out_dir.mkdir(exist_ok=True)
    energy_path = out_dir / f"MD_energy_curve_{now_time}.png"
    sim.save_energy_curve(str(energy_path))
    traj_path = out_dir / f"MD_traj_{now_time}.xyz"
    sim.save_xyz_trajectory(str(traj_path), atom_types=mol.atom_types)
    force_path = out_dir / f"forces_{now_time}.xyz"
    sim.save_forces_grad(str(force_path), with_no_ele=True, atom_types=mol.atom_types)
    print('已保存:')
    print('  Energies:', energy_path)
    print('  Traj:', traj_path)
    print('  Forces:', force_path)


def main():
    print('=== Simulon Agent ===')
    if ChatOpenAI is None:
        pass  # 仅在选择 1 时再提示未安装
    while True:
        print('\n1) 生成 MD 脚本 (LLM)')
        print('2) 分析模拟输出 (基于输出目录)')
        print('3) 基于输出进行问答 (可选 LLM)')
        print('4) 生成 GIF 动图 (基于 traj.xyz)')
        print('5) 运行 MLPS (CHGNet)')
        print('6) 退出')
        try:
            choice = input('请选择 (1/2/3/4/5/6): ').strip()
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
                print('GIF 功能不可用。')
            continue
        if choice == '5':
            mlps_cli()
            continue
        if choice == '6' or choice.lower() in ('q', 'quit', 'exit'):
            print('已退出。')
            break
        if choice != '1':
            print('无效选择，请重试。')
            continue
        # 生成 MD 脚本（原逻辑）
        if ChatOpenAI is None:
            print('未检测到 ChatOpenAI, 可执行: pip install langchain-openai')
            continue
        try:
            api_key_env = os.environ.get('SIMULON_LLM_API_KEY') or os.environ.get('OPENAI_API_KEY')
            if api_key_env:
                use_env = input(f'检测到环境变量 API Key，直接使用? (Y/n): ').strip().lower()
                if use_env in ('', 'y', 'yes'): api_key = api_key_env
                else: api_key = input('Please input LLM API Key: ').strip()
            else:
                api_key = input('Please input LLM API Key: ').strip()
            if not api_key:
                print('API Key is not provided, back to menu.')
                continue
            base_url = input('Please input Base URL (Default https://api.deepseek.com/v1): ').strip() or 'https://api.deepseek.com/v1'
            model_name = input('Please input Model Name (Default deepseek-reasoner): ').strip() or 'deepseek-reasoner'
        except EOFError:
            print('输入中断，返回菜单。')
            continue
        try:
            user_text = input('请输入你的需求（例如："用 LJ 力场模拟 Ar10000 5000步 温度 94.4 K 先最小化"）:\n> ').strip()
        except EOFError:
            user_text = ''
        if not user_text:
            print('未输入，返回菜单')
            continue
        agent = ScriptLLMAgent(model_name=model_name, api_key=api_key, base_url=base_url)
        print('[LLM] 正在生成脚本...')
        code = agent.generate(user_text)
        out_dir = Path(__file__).resolve().parents[1] / 'run_scripts'
        script_path = save_generated_script(code, out_dir)
        print(f'[LLM] 脚本已保存: {script_path}')
        try:
            run_now = input('立即运行该脚本? (y/N): ').strip().lower()
        except EOFError:
            run_now = 'n'
        if run_now == 'y':
            print('[Runner] 执行中...')
            try:
                scope = {}
                exec(compile(code, str(script_path), 'exec'), scope, scope)
                print('[Runner] 执行完成')
            except Exception as e:
                print(f'[Runner] 运行失败: {e}')
        else:
            print('可手动运行: python', script_path)

# 交互式：生成 GIF 动图（基于 traj.xyz）
def make_gif_cli():
    out_dir = input('请输入模拟输出目录 (包含 traj.xyz): ').strip()
    if not out_dir:
        print('未提供输出目录。')
        return
    try:
        stride = int(input('帧间隔 (默认 5): ').strip() or '5')
        maxf = int(input('最大帧数 (默认 200): ').strip() or '200')
        fps = int(input('GIF 帧率 fps (默认 20): ').strip() or '20')
    except Exception:
        stride, maxf, fps = 5, 200, 20
    an = OutputAnalyzer(out_dir)
    try:
        path = an.generate_gif(stride_frames=stride, max_frames=maxf, fps=fps)
        print('GIF 已保存:', path)
    except Exception as e:
        print('生成失败:', e)

if __name__ == "__main__":
    main()
