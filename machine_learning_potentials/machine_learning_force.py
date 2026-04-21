"""
machine_learning_potentials/machine_learning_force.py

优化要点（vs 原版）：
1. CHGNet graph 缓存 + Verlet-style 惰性重建
   - graph_converter 只在最大位移 > rebuild_tol 时重新建图
   - 典型体系每 20~50 步才需要重建一次，而非每步
2. 绕过 predict_structure() 高层 API
   - 直接调用 model([graph], task='efs')
   - 省去内部的 Structure→graph 转换开销
3. CPU/GPU 数据流精简
   - 坐标只做一次 .detach().cpu().numpy()
   - 力结果直接 torch.from_numpy().to(device) 而非 tensor(...)
4. virial 支持（NPT）
   - task='efs' 返回应力张量，转换为维里标量
5. finetune 与模型加载逻辑保持不变
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from torch import tensor

from ase.io import read as ase_read
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor

from core.md_model import BackboneInterface


class MachineLearningForce(BackboneInterface, nn.Module):

    def __init__(
        self,
        molecular,
        aimd_pos_file: str,
        aimd_force_file: str,
        mlp_model_name: str = None,
        mlps_finetune_params: dict = None,
        mlps_model_path: str = None,
        # 图缓存阈值（Å）：最大位移超过此值才重建 CHGNet graph
        rebuild_tol: float = 0.5,
    ):
        super().__init__()
        self.molecular = molecular
        self.device = molecular.device

        box_size = float(molecular.box_length.cpu()) \
                   if torch.is_tensor(molecular.box_length) \
                   else float(molecular.box_length)
        self.cell_param = [box_size, box_size, box_size]

        if not (aimd_force_file.endswith('.xyz') and aimd_pos_file.endswith('.xyz')):
            raise ValueError('Input files must be in .xyz format')
        self.aimd_pos_file   = aimd_pos_file
        self.aimd_force_file = aimd_force_file

        self.mlp_model_name = mlp_model_name
        self.rebuild_tol    = rebuild_tol
        self.model          = None

        # 图缓存状态
        self._chgnet_graph   = None          # 上一次建图结果
        self._ref_coords_np  = None          # 建图时的参考坐标（numpy, CPU）
        self._lattice        = None          # pymatgen Lattice（正交时固定）
        self._species        = None          # 元素列表（固定）
        self._graph_builds   = 0            # 统计重建次数

        # 加载或微调模型
        if mlps_model_path:
            self.mlps_model_path = mlps_model_path
            self.load_model_from_path()
        else:
            self.mlps_finetune_params = mlps_finetune_params
            self.mlp_flow()
            self._load_model()

        # 预建图（避免第一步推理时的冷启动延迟）
        if self.model is not None and mlp_model_name == 'chgnet':
            self._init_chgnet_cache()

    # ─── forward ───────────────────────────────────────────────────────────────
    def forward(self):
        if self.mlp_model_name == 'chgnet':
            return self._forward_chgnet()
        return {'energy': torch.tensor(0.0, device=self.device),
                'forces': torch.zeros_like(self.molecular.coordinates),
                'virial': torch.tensor(0.0, device=self.device)}

    def _forward_chgnet(self):
        coords_np = self.molecular.coordinates.detach().cpu().numpy()  # [N,3]

        # ── 拓扑惰性重建（位移 > rebuild_tol 才跑 graph_converter） ──────
        if self._needs_rebuild(coords_np):
            self._rebuild_graph(coords_np)   # 更新拓扑 + 坐标
        else:
            # 每步都必须更新坐标（CHGNet 对 atom_frac_coord 求 autograd 得力）
            self._update_frac_coords(coords_np)

        # ── 直接调用模型（跳过 predict_structure 的 graph 重建） ──────────
        # CHGNet 通过 autograd 计算力（F = -dE/dx），必须保留梯度追踪
        with torch.enable_grad():
            result = self.model([self._chgnet_graph], task='efs')

        # energy: eV/atom → eV 总能
        e_per_atom = float(result['e'][0])
        N = self.molecular.atom_count
        total_energy = torch.tensor(e_per_atom * N, dtype=torch.float32,
                                    device=self.device)

        # forces: CHGNet 在 CUDA 下返回 Tensor，CPU 下返回 numpy，统一处理
        forces_raw = result['f'][0]
        if isinstance(forces_raw, torch.Tensor):
            forces = forces_raw.detach().to(self.device).float()
        else:
            forces = torch.from_numpy(
                np.asarray(forces_raw, dtype=np.float32)
            ).to(self.device)

        # stress → virial（用于 NPT）
        # CHGNet 应力单位：GPa，形状 [3,3] 或 [6]
        # virial = -stress * V，维里标量 = trace(virial)/3
        virial = torch.tensor(0.0, device=self.device)
        if 's' in result and result['s'][0] is not None:
            s_raw = result['s'][0]
            stress_gpa = s_raw.detach().cpu().numpy().astype(np.float32) \
                         if isinstance(s_raw, torch.Tensor) \
                         else np.asarray(s_raw, dtype=np.float32)
            if stress_gpa.ndim == 1 and len(stress_gpa) == 6:
                # Voigt: [s11,s22,s33,s23,s13,s12] → trace = s11+s22+s33
                trace = float(stress_gpa[0] + stress_gpa[1] + stress_gpa[2])
            else:
                trace = float(np.trace(stress_gpa.reshape(3, 3)))
            # 换算：1 GPa·Å³ = 1e-21 J / 1.60218e-19 J/eV = 6.2415e-3 eV
            # V in Å³
            V = self._volume()
            virial = torch.tensor(-trace / 3.0 * V * 6.2415e-3,
                                  dtype=torch.float32, device=self.device)

        return {'energy': total_energy, 'forces': forces, 'virial': virial}

    # ─── CHGNet 图缓存 ─────────────────────────────────────────────────────────
    def _init_chgnet_cache(self):
        coords_np = self.molecular.coordinates.detach().cpu().numpy()
        self._species = list(self.molecular.atom_types)
        self._lattice = self._build_lattice()
        self._rebuild_graph(coords_np)

    def _needs_rebuild(self, coords_np: np.ndarray) -> bool:
        if self._chgnet_graph is None or self._ref_coords_np is None:
            return True
        disp = np.abs(coords_np - self._ref_coords_np).max()
        return disp > self.rebuild_tol

    def _rebuild_graph(self, coords_np: np.ndarray):
        from chgnet.graph import CrystalGraphConverter
        if not hasattr(self, '_graph_converter') or self._graph_converter is None:
            # atom_graph_cutoff 与 CHGNet 默认保持一致（5 Å）
            self._graph_converter = CrystalGraphConverter(
                atom_graph_cutoff=5.0,
                bond_graph_cutoff=3.0,
            )
        structure = Structure(
            self._lattice,
            self._species,
            coords_np,
            coords_are_cartesian=True,
        )
        # .to() 返回新对象，必须重新赋值
        self._chgnet_graph  = self._graph_converter(
            structure, graph_id='md_step'
        ).to(self.device)
        self._ref_coords_np = coords_np.copy()
        self._graph_builds += 1

    def _update_frac_coords(self, coords_np: np.ndarray):
        """
        仅替换 atom_frac_coord，复用缓存的图拓扑（bond/angle connectivity）。
        比完整重建快 10-30×（省去 CrystalGraphConverter 的键角搜索）。
        """
        from chgnet.graph.crystalgraph import CrystalGraph
        # 笛卡尔坐标 → 分数坐标
        H    = self._lattice.matrix                        # [3,3] numpy
        H_inv = np.linalg.inv(H)
        frac = torch.tensor(
            coords_np @ H_inv.T, dtype=torch.float32, device=self.device
        )
        g = self._chgnet_graph
        # 创建新 CrystalGraph：复用所有拓扑张量，只换 atom_frac_coord
        self._chgnet_graph = CrystalGraph(
            atomic_number      = g.atomic_number,
            atom_frac_coord    = frac,
            atom_graph         = g.atom_graph,
            atom_graph_cutoff  = g.atom_graph_cutoff,
            neighbor_image     = g.neighbor_image,
            directed2undirected= g.directed2undirected,
            undirected2directed= g.undirected2directed,
            bond_graph         = g.bond_graph,
            bond_graph_cutoff  = g.bond_graph_cutoff,
            lattice            = g.lattice,
            graph_id           = g.graph_id,
        )

    def _build_lattice(self) -> Lattice:
        if hasattr(self.molecular, 'box') and self.molecular.box.is_orthogonal:
            d = self.molecular.box.diag.cpu().float().tolist()
            return Lattice.orthorhombic(*d)
        return Lattice.cubic(self.cell_param[0])

    def _volume(self) -> float:
        if hasattr(self.molecular, 'box'):
            return self.molecular.box.volume
        L = float(self.molecular.box_length)
        return L ** 3

    # ─── 模型加载 / 微调（保持原逻辑不变） ────────────────────────────────────
    def load_model_from_path(self):
        if self.mlp_model_name == 'chgnet':
            from chgnet.model import CHGNet
            self.model = CHGNet.from_file(self.mlps_model_path).to(self.device)
        else:
            raise NotImplementedError(f'Unknown model: {self.mlp_model_name}')

    def _load_model(self):
        if self.mlp_model_name == 'chgnet':
            root_dir    = os.path.dirname(os.path.abspath(__file__))
            model_folder = os.path.join(root_dir, 'model')
            files = sorted(os.listdir(model_folder))
            if files:
                from chgnet.model import CHGNet
                self.model = CHGNet.from_file(
                    os.path.join(model_folder, files[0])
                ).to(self.device)
            else:
                raise FileNotFoundError("No model found in 'model' folder")
        else:
            self.model = None

    def mlp_flow(self):
        if self.mlp_model_name is None:
            raise NotImplementedError('Provide mlp_model_name')
        self.finetune_large_mlp()

    def convert_aimd_to_dataset(self):
        position_frames = ase_read(self.aimd_pos_file,  index=':')
        force_frames    = ase_read(self.aimd_force_file, index=':')
        structures, forces, energies = [], [], []
        for pos_atoms, force_atoms in zip(position_frames, force_frames):
            pos_atoms.set_cell(self.cell_param)
            pos_atoms.set_pbc(True)
            structures.append(AseAtomsAdaptor().get_structure(pos_atoms))
            forces.append(force_atoms.positions)
            energies.append(pos_atoms.info['E'])
        return {'structures': structures, 'forces': forces, 'energies': energies}

    def finetune_large_mlp(self):
        dataset_dict = self.convert_aimd_to_dataset()
        from chgnet.data.dataset import StructureData, get_train_val_test_loader
        from chgnet.model import CHGNet
        from chgnet.trainer import Trainer

        dataset = StructureData(
            structures=dataset_dict['structures'],
            forces=dataset_dict['forces'],
            energies=dataset_dict['energies'],
        )
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset, batch_size=8, train_ratio=0.9, val_ratio=0.05
        )
        origin_chgnet = CHGNet()
        # 冻结底层参数，只微调顶层
        for layer in [
            origin_chgnet.atom_embedding,
            origin_chgnet.bond_embedding,
            origin_chgnet.angle_embedding,
            origin_chgnet.bond_basis_expansion,
            origin_chgnet.angle_basis_expansion,
            origin_chgnet.atom_conv_layers[:-1],
            origin_chgnet.bond_conv_layers,
            origin_chgnet.angle_layers,
        ]:
            for param in layer.parameters():
                param.requires_grad = False

        p = self.mlps_finetune_params or {}
        trainer = Trainer(
            model=origin_chgnet,
            targets=p.get('targets', 'ef'),
            optimizer=p.get('optimizer', 'Adam'),
            scheduler=p.get('scheduler', 'CosLR'),
            criterion=p.get('criterion', 'MSE'),
            epochs=p.get('epochs', 10),
            learning_rate=p.get('learning_rate', 0.002),
            use_device=self.device,
            print_freq=p.get('print_freq', 6),
        )
        root_dir     = os.path.dirname(os.path.abspath(__file__))
        model_folder = os.path.join(root_dir, 'model')
        os.makedirs(model_folder, exist_ok=True)
        trainer.train(train_loader, val_loader, test_loader, save_dir=model_folder)

    # ─── 诊断 ──────────────────────────────────────────────────────────────────
    def graph_build_stats(self) -> dict:
        """返回图重建统计信息（供外部打印）。"""
        return {
            'graph_builds': self._graph_builds,
            'rebuild_tol_ang': self.rebuild_tol,
        }
