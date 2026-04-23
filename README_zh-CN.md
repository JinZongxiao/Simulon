# Simulon

[中文] | [English](README.md)

一个基于 PyTorch 的轻量级分子动力学（MD）引擎，提供可选的自定义 CUDA 加速内核。Simulon 注重清晰的代码结构、可扩展性与科研/工程实用工作流。

---

## 最新更新

| 模块 | 变更内容 |
|------|---------|
| **系综** | NVE（微正则）、NVT（Langevin 热浴）、**NPT（Berendsen 压浴）** |
| **三斜盒子** | 通过 `core/box.py` 支持完整 3×3 H 矩阵 PBC，正交与非正交统一接口 |
| **Restart** | `save_checkpoint` / `load_checkpoint` — 保存坐标、速度、盒子、RNG 状态 |
| **力场** | LJ、EAM、BMH 均新增 `virial` 返回值，支持 NPT 压力耦合 |
| **邻居搜索** | 修复重复边 Bug；Box-aware 最小镜像；CUDA 内核前缀和 O(N²)→O(1) |
| **BMH** | 全面重写：边列化解析力，消除 O(N²) 内存分配 |
| **EAM** | 删除死代码；向量化查表（无 Python 循环）；加入 virial |
| **W 拉伸** | 新增 `run_scripts/w_tensile.py`：张量应力输出、取向 BCC-W 结构生成、应力应变绘图、横向各向异性 NPT 支持 |
| **W 纳米压痕** | 新增 `run_scripts/w_indent.py`：球形压头加载、底层固定 W slab、载荷-深度输出和 smoke test |
| **性能** | RTX 3050 上 100 原子 Ar NVT 约 **384 步/s** |

---

## 核心能力

- **PyTorch 优先**：所有状态均存放在张量中，一行代码切换 CPU/GPU。
- **三种系综**：NVE、NVT（Langevin 热浴）、NPT（Berendsen 压浴 + Langevin）。
- **张量应力输出**：力场现在返回标量 `virial` 和 `virial_tensor`，拉伸流程使用完整应力张量。
- **三斜 PBC**：统一 `Box` 类，通过 3×3 格矢矩阵处理立方、正交、任意三斜盒子。
- **Verlet 邻居表**：基于位移阈值（skin/2）的惰性重建；可选 CUDA 扩展加速。
- **模块化力场**：Lennard-Jones、EAM、Born–Mayer–Huggins，以及用户自定义对势模板。
- **W 力学流程**：内置钨拉伸和纳米压痕脚本，支持 `[100]/[110]/[111]` 取向结构生成、CSV/PNG 输出和 smoke test。
- **Restart**：完整断点续跑支持，每 N 步保存一次，重启无需重新平衡。
- **RDF 分析器**：在线累积，同类/异类原子对均有正确归一化。
- **I/O 与工具**：XYZ 读写、CSV 能量日志、轨迹输出、EAM 表格解析、pymatgen/ASE 集成。
- **机器学习势**：示例性接入 CHGNet 类模型。

---

## 仓库结构

```
core/
  box.py                  # 统一正交+三斜 PBC（H 矩阵）
  barostat.py             # 各向同性 Berendsen + 对角各向异性 NPT 压浴
  mechanics/loading.py    # 单轴拉伸加载器
  md_model.py             # SumBackboneInterface、BaseModel（主 MD 循环）
  md_simulation.py        # MDSimulator：运行循环、日志、轨迹输出
  analyser.py             # RDF 累积器
  energy_minimizer.py     # 最速下降能量最小化
  force/
    lennard_jones_force.py
    eam_force.py
    born_mayer_huggins_force.py
    template/pair_force_template.py
  integrator/integrator.py  # 速度 Verlet（NVE / NVT / NPT）
  neighbor_search/gpu_kdtree.py

io_utils/
  reader.py               # AtomFileReader：XYZ → 张量 + 邻居表
  w_bcc.py                # 取向 BCC-W 结构生成
  restart.py              # save_checkpoint / load_checkpoint
  writer.py / output_logger.py / eam_parser.py / ...

postprocess/
  stress_strain.py        # 应力应变摘要 + PNG 绘图
  indentation.py          # 载荷-深度摘要 + PNG 绘图

cuda source/
  neighbor_search_kernel.cu
  lj_energy_force*.cu
  eam_cuda_ext*.cu

run_scripts/
  demo_ar_nvt.py          # 快速演示：100 原子 Ar NVT
  lj_run.py               # JSON 驱动的 LJ 模拟
  user_defined_run.py
  mlps_run.py
  w_tensile.py            # 钨拉伸工作流
  w_indent.py             # 钨纳米压痕工作流
  check_w_orientation.py  # 取向 BCC-W 结构静态检查
  plot_md_diagnostics.py

run_data/                 # 示例结构（Ar、Cu、W 等）
simulation_agent/         # 中英文交互 MD 助手
```

---

## 环境依赖

- Python 3.10+（已在 3.11 测试）
- PyTorch ≥ 2.0（可选 CUDA）。参考 https://pytorch.org
- `numpy scipy matplotlib ase pymatgen tqdm torch_geometric`
- 可选（ML 示例）：`chgnet`

---

## 安装

```bash
# 1. Python 依赖
pip install torch torchvision torchaudio          # 按实际 CUDA 版本选择
pip install numpy scipy matplotlib ase pymatgen tqdm
pip install torch_geometric

# 2. CUDA 扩展（可选，大体系推荐）
#    需要：MSVC Build Tools（Windows）或 GCC，以及匹配的 CUDA 工具包
python setup.py build_ext --inplace
```

> **Windows 说明**：仓库已包含适用于 Python 3.11 + CUDA 12.x 的预编译 `simulon_cuda.cp311-win_amd64.pyd`，若环境匹配可直接使用，否则建议从源码编译。

---

## 快速开始

### 1. 即刻演示 — Ar NVT

```bash
python run_scripts/demo_ar_nvt.py
```

100 个 Ar 原子，FCC 结构，LJ 力场，Langevin NVT 90 K，500 步。轨迹和能量 CSV 输出到 `run_output/demo_ar_nvt/`。

### 2. JSON 驱动的 LJ 模拟

```bash
python run_scripts/lj_run.py --config run_scripts/lj_run.json
```

编辑 `lj_run.json` 调整结构、盒长、LJ 参数、系综、温度与输出路径。

### 3. NPT 模拟（Python API）

```python
from io_utils.reader import AtomFileReader
from core.force.lennard_jones_force import LennardJonesForce
from core.md_model import SumBackboneInterface, BaseModel
from core.integrator.integrator import VerletIntegrator
from core.barostat import BerendsenBarostat
from core.md_simulation import MDSimulator

mol   = AtomFileReader('structure.xyz', box_length=30.0, cutoff=10.0,
                       parameter={"[0 0]": {"epsilon": 0.0104, "sigma": 3.4}})
ff    = LennardJonesForce(mol)
integ = VerletIntegrator(mol, dt=0.001, ensemble='NPT',
                         temperature=(300, 300), gamma=0.01)
baro  = BerendsenBarostat(mol, target_pressure=1.0, tau_p=0.5)
model = BaseModel(SumBackboneInterface([ff], mol), integ, mol, barostat=baro)

MDSimulator(model, num_steps=5000, print_interval=100).run()
```

### 4. 三斜盒子

```python
from core.box import Box
import torch

H = torch.tensor([[a, 0, 0],
                  [b*cos(gamma), b*sin(gamma), 0],
                  [...]])          # 任意合法格矢矩阵
mol = AtomFileReader('structure.xyz', box_length=a, box_vectors=H, ...)
```

### 5. Restart / 断点续跑

```python
from io_utils.restart import save_checkpoint, load_checkpoint

# 每 1000 步保存一次
save_checkpoint(model, step=1000, path='ckpt.pt')

# 续跑
next_step = load_checkpoint(model, path='ckpt.pt')
for step in range(next_step, total_steps):
    model()
```

### 6. W 拉伸工作流

最小 smoke test：

```bash
python run_scripts/w_tensile.py --smoke
python cuda_test/test_w_tensile_smoke.py
```

推荐的 W `[100]` 拉伸基线参数（横向各向异性 NPT）：

```bash
python run_scripts/w_tensile.py \
  --orientation 100 \
  --replicas 4,4,3 \
  --lateral-mode stress-free \
  --steps 5000 \
  --strain-rate 0.00005 \
  --barostat-tau 0.1 \
  --barostat-gamma 1.0 \
  --gamma 2.0
```

一次跑完三个常用 W 拉伸取向，且结果不会互相覆盖：

```bash
python run_scripts/w_tensile.py --orientation 100 --replicas 4,4,3 --lateral-mode stress-free --steps 5000 --strain-rate 0.00005 --barostat-tau 0.1 --barostat-gamma 1.0 --gamma 2.0 --output-dir run_output/w_tensile
python run_scripts/w_tensile.py --orientation 110 --replicas 4,4,3 --lateral-mode stress-free --steps 5000 --strain-rate 0.00005 --barostat-tau 0.1 --barostat-gamma 1.0 --gamma 2.0 --output-dir run_output/w_tensile
python run_scripts/w_tensile.py --orientation 111 --replicas 3,3,2 --lateral-mode stress-free --steps 5000 --strain-rate 0.00005 --barostat-tau 0.1 --barostat-gamma 1.0 --gamma 2.0 --output-dir run_output/w_tensile
```

取向结构静态检查：

```bash
python run_scripts/check_w_orientation.py --orientation all
```

输出会按取向分目录保存：

- `run_output/w_tensile/orientation_100/`
- `run_output/w_tensile/orientation_110/`
- `run_output/w_tensile/orientation_111/`

每个拉伸输出目录包括：

- `stress_strain.csv`
- `summary.json`
- `stress_strain.png`
- 自动生成的取向结构，如 `W_100_generated.xyz`

CSV 中包含 `stress_xx_bar`、`stress_yy_bar`、`stress_zz_bar`、盒长、能量、温度和维里张量对角元。

### 7. W 纳米压痕工作流

最小 smoke test：

```bash
python run_scripts/w_indent.py --smoke
python cuda_test/test_w_indent_smoke.py
```

W `[100]` 球形压头示例：

```bash
python run_scripts/w_indent.py \
  --orientation 100 \
  --replicas 6,6,4 \
  --steps 5000 \
  --indenter-radius-A 8.0 \
  --indenter-stiffness 5.0 \
  --indent-rate-A-ps 0.05 \
  --gamma 2.0
```

输出按取向分目录保存，例如 `run_output/w_indent/orientation_100/`，包含 `load_depth.csv`、`summary.json`、`load_depth.png` 和生成的 slab 结构。

---

## 系综对照表

| 系综 | `VerletIntegrator` 参数 | 附加组件 |
|------|------------------------|---------|
| NVE | `ensemble='NVE'` | — |
| NVT | `ensemble='NVT', temperature=(T_init, T_target), gamma=γ` | Langevin |
| NPT | `ensemble='NPT', temperature=(T_init, T_target), gamma=γ` | + `BerendsenBarostat` 或 `AnisotropicNPTBarostat` 传入 `BaseModel` |

---

## 常见问题

| 问题 | 解决方法 |
|------|---------|
| CUDA 编译失败 | 确认 `nvcc --version` 与 PyTorch CUDA 版本一致；Windows 需 MSVC ≥ 2019 |
| `ImportError: simulon_cuda` | 重新编译：`python setup.py build_ext --inplace` |
| `KeyError '[0 0]'` | 参数字典键须为 `str(np.array([type_i, type_j]))`，单元素体系用 `"[0 0]"` |
| 初始温度不对 | `temperature=(T_init, T_target)`，第一个值用于初始化 Maxwell-Boltzmann 速度 |

---

## 贡献

欢迎提交 Issue / PR。反馈问题时请尽量提供最小可复现示例或小型输入结构。
