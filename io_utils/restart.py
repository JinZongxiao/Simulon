"""
io_utils/restart.py — MD 模拟断点续跑

保存/恢复完整模拟状态：
  - 原子坐标、速度
  - 盒子 H 矩阵（支持正交与三斜）
  - 当前步号
  - CPU/GPU 随机数生成器状态（保证随机序列连续）
  - 力场预分配缓冲区（下次启动无需重建）

文件格式：PyTorch .pt（pickle+tensor），可跨平台读取。

用法
----
    from io_utils.restart import save_checkpoint, load_checkpoint

    # 每 N 步保存一次
    save_checkpoint(model, step=1000, path='restart.pt')

    # 续跑时恢复
    step_start = load_checkpoint(model, path='restart.pt')
    for step in range(step_start, total_steps):
        model()
"""
from __future__ import annotations
import torch
from pathlib import Path


# ─── 保存 ─────────────────────────────────────────────────────────────────────

def save_checkpoint(model, step: int, path: str | Path) -> None:
    """
    将模拟状态序列化到磁盘。

    Parameters
    ----------
    model : BaseModel
        当前运行的 MD 模型（含 molecular、Integrator、sum_bone）。
    step : int
        当前步号（从 0 开始），恢复后从 step+1 继续。
    path : str | Path
        输出文件路径（建议后缀 .pt）。
    """
    mol = model.molecular

    ckpt = {
        'step': int(step),
        'coordinates': mol.coordinates.cpu(),
        'velocities':  mol.atom_velocities.cpu(),
    }

    # 盒子：Box 对象或标量
    if hasattr(mol, 'box'):
        ckpt['box_H'] = mol.box.state_dict()['H'].cpu()   # [3, 3]
    elif hasattr(mol, 'box_length'):
        ckpt['box_length'] = float(mol.box_length)

    # 随机数状态（CPU + GPU）
    ckpt['rng_cpu'] = torch.get_rng_state()
    if torch.cuda.is_available():
        ckpt['rng_cuda'] = torch.cuda.get_rng_state(mol.device)

    # 积分器内部半步速度（断点后第一步无需重新计算 vel_half）
    integrator = model.Integrator
    if hasattr(integrator, 'vel_half'):
        ckpt['vel_half'] = integrator.vel_half.cpu()

    # 力缓存（避免断点后第一步多算一次力）
    if hasattr(model, 'force_cache') and model.force_cache is not None:
        ckpt['force_cache'] = model.force_cache.cpu()
    if hasattr(model, 'energy_cache') and model.energy_cache is not None:
        ckpt['energy_cache'] = model.energy_cache.cpu()
    if hasattr(model, 'virial_cache') and model.virial_cache is not None:
        ckpt['virial_cache'] = model.virial_cache.cpu()

    torch.save(ckpt, path)
    print(f"[Restart] checkpoint saved → {path}  (step={step})")


# ─── 恢复 ─────────────────────────────────────────────────────────────────────

def load_checkpoint(model, path: str | Path) -> int:
    """
    从磁盘恢复模拟状态。

    Parameters
    ----------
    model : BaseModel
        当前运行的 MD 模型。
    path : str | Path
        checkpoint 文件路径。

    Returns
    -------
    int
        下一步的步号（即 ckpt['step'] + 1）。
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    mol  = model.molecular
    dev  = mol.device

    # ── 坐标 & 速度 ──────────────────────────────────────────────────────────
    mol.coordinates     = ckpt['coordinates'].to(dev)
    mol.atom_velocities = ckpt['velocities'].to(dev)
    # graph_data.pos 与 coordinates 保持同步
    if hasattr(mol, 'graph_data') and hasattr(mol.graph_data, 'pos'):
        mol.graph_data.pos = mol.coordinates

    # ── 盒子 ─────────────────────────────────────────────────────────────────
    if 'box_H' in ckpt and hasattr(mol, 'box'):
        from core.box import Box
        mol.box = Box.from_state_dict({'H': ckpt['box_H']}, device=dev)
    elif 'box_length' in ckpt and hasattr(mol, 'box_length'):
        mol._box_length = ckpt['box_length']   # 向后兼容标量存储

    # ── 邻居列表：强制重建 ──────────────────────────────────────────────────
    # 将 last_positions 置 None，而非 zeros。
    # 原因：update_coordinates 在 last_positions is not None 时会做位移检查，
    # 若检查结果 <= skin/2（例如所有原子恰好都在坐标轴原点附近）则会把
    # needs_update 覆盖写回 False，导致邻居表跳过重建，产生错误的 edge_index。
    # 置为 None 走 else 分支，needs_update = True 无条件触发，完全安全。
    if hasattr(mol, 'last_positions'):
        mol.last_positions = None  # → update_coordinates else-branch → needs_update=True 无条件

    # ── 积分器内部状态 ───────────────────────────────────────────────────────
    integrator = model.Integrator
    if 'vel_half' in ckpt and hasattr(integrator, 'vel_half'):
        integrator.vel_half = ckpt['vel_half'].to(dev)

    # ── 力缓存 ───────────────────────────────────────────────────────────────
    if 'force_cache' in ckpt:
        model.force_cache  = ckpt['force_cache'].to(dev)
    if 'energy_cache' in ckpt:
        model.energy_cache = ckpt['energy_cache'].to(dev)
    if 'virial_cache' in ckpt:
        model.virial_cache = ckpt['virial_cache'].to(dev)
    # 跳过第一次重算力（数据已从 checkpoint 恢复）
    model._first_call = False

    # ── 随机数状态 ──────────────────────────────────────────────────────────
    if 'rng_cpu' in ckpt:
        torch.set_rng_state(ckpt['rng_cpu'])
    if 'rng_cuda' in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt['rng_cuda'].to(dev))

    next_step = int(ckpt['step']) + 1
    print(f"[Restart] checkpoint loaded ← {path}  (resuming from step={next_step})")
    return next_step
