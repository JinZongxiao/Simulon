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
| **W 裂纹** | 新增 `run_scripts/w_crack.py`：中心预裂纹生成、刚性 grip 开口位移加载、应力-CMOD 输出和 smoke test |
| **W DBTT** | 新增 `run_scripts/w_dbtt_scan.py` 和 `postprocess/dbtt.py`：基于裂纹开口的温度扫描与 DBTT 趋势分析 |
| **性能** | RTX 3050 上 100 原子 Ar NVT 约 **384 步/s** |

---

## 核心能力

- **PyTorch 优先**：所有状态均存放在张量中，一行代码切换 CPU/GPU。
- **三种系综**：NVE、NVT（Langevin 热浴）、NPT（Berendsen 压浴 + Langevin）。
- **张量应力输出**：力场现在返回标量 `virial` 和 `virial_tensor`，拉伸流程使用完整应力张量。
- **三斜 PBC**：统一 `Box` 类，通过 3×3 格矢矩阵处理立方、正交、任意三斜盒子。
- **Verlet 邻居表**：基于位移阈值（skin/2）的惰性重建；可选 CUDA 扩展加速。
- **模块化力场**：Lennard-Jones、EAM、Born–Mayer–Huggins，以及用户自定义对势模板。
- **W 力学流程**：内置钨拉伸、纳米压痕和裂纹开口脚本，支持 `[100]/[110]/[111]` 取向结构生成、CSV/PNG 输出和 smoke test。
- **Restart**：完整断点续跑支持，每 N 步保存一次，重启无需重新平衡。
- **RDF 分析器**：在线累积，同类/异类原子对均有正确归一化。
- **I/O 与工具**：XYZ 读写、CSV 能量日志、轨迹输出、EAM 表格解析、pymatgen/ASE 集成。
- **机器学习势**：示例性接入 CHGNet 类模型。

---

## 模拟视频

W 力学流程视频位于 `docs/videos/`：

| 工作流 | 视频 |
|--------|------|
| W 拉伸 | [w_tensile.mp4](docs/videos/w_tensile.mp4) |
| W 纳米压痕 | [w_nanoindentation.mp4](docs/videos/w_nanoindentation.mp4) |
| W 裂纹开口 | [w_crack_opening.mp4](docs/videos/w_crack_opening.mp4) |

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
  crack.py                # 应力-CMOD 摘要 + PNG 绘图
  dbtt.py                 # 温度扫描聚合 + PNG 绘图

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
  w_crack.py              # 钨裂纹开口工作流
  w_dbtt_scan.py          # 钨 DBTT 温度扫描
  w_batch_report.py       # W 力学批量运行与汇总报告
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

CSV 中包含有符号应力列（`stress_xx_bar`、`stress_yy_bar`、`stress_zz_bar`）、拉伸为正的展示列（`tension_xx_bar`、`tension_yy_bar`、`tension_zz_bar`）、盒长、能量、温度和维里张量对角元。

新版拉伸工作流现在会：

- 先通过 `--equil-steps` 做零压预平衡
- 在 `stress_xx_bar` 中输出相对于平衡初态的应力
- 额外输出拉伸为正的展示列：`tension_xx_bar`、`tension_yy_bar`、`tension_zz_bar`
- 同时保留绝对应力列：`stress_xx_abs_bar`、`stress_yy_abs_bar`、`stress_zz_abs_bar`
- 通过 `--barostat-compressibility-bar-inv` 和 `--barostat-pressure-tolerance-bar` 稳定各向异性侧向控压
- 如果横向盒长超过 `--max-lateral-box-ratio`，会直接报错终止，避免静默生成失真曲线
- 通过 `--traj-interval` 输出 `trajectory.xyz`

对大体系 `--orientation custom`，仍建议检查 `summary.json` 里的 `initial_stress_xx_abs_bar`、`initial_stress_yy_abs_bar`、`initial_stress_zz_abs_bar`。如果预平衡后它们仍然很大，先增加 `--equil-steps` 或重新调 barostat 参数，再去解释拉伸曲线。

服务器大体系自定义结构示例：

```bash
python run_scripts/w_tensile.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --steps 100000 \
  --equil-steps 1000 \
  --strain-rate 0.0004 \
  --lateral-mode stress-free \
  --barostat-tau 0.1 \
  --barostat-gamma 1.0 \
  --barostat-compressibility-bar-inv 3.2e-7 \
  --barostat-pressure-tolerance-bar 25.0 \
  --max-lateral-box-ratio 2.0 \
  --gamma 2.0 \
  --traj-interval 1000 \
  --output-dir run_output/prod_w_tensile_W31250
```

`W31250.xyz` 是一个立方 BCC W 体系，`31250 / 2 = 15625 = 25^3` 个晶胞，晶格常数 `3.2 A`，所以正确的 `--box-length` 是 `80.0`。

### W bulk relax 工作流

如果大体系 tensile 的 `summary.json` 里 `initial_stress_*_abs_bar` 仍然很大，先用这条工作流把 bulk W 结构放松到接近零压，再拿 relaxed 结构去做拉伸。

```bash
python run_scripts/w_bulk_relax.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --steps 5000 \
  --temperature 300 \
  --gamma 2.0 \
  --target-pressure-bar 0.0 \
  --barostat-tau 0.5 \
  --barostat-compressibility-bar-inv 3.2e-7 \
  --barostat-mu-max 0.005 \
  --traj-interval 500 \
  --output-dir run_output/w_bulk_relax_W31250
```

输出包括：

- `relaxation.csv`
- `summary.json`
- relaxed XYZ 结构，例如 `W_custom_relaxed.xyz`
- 可选的 `trajectory.xyz`

`summary.json` 里重点看：

- `recommended_box_length_A`
- 如果脚本能识别出立方 BCC 晶胞数，还会给出 `recommended_lattice_param_A`
- `final_pressure_bar`
- `final_box_length_x/y/z`

推荐使用顺序：

1. 先把 bulk W 放松到接近零压
2. 取 `recommended_box_length_A` 和 relaxed XYZ
3. 再把它们作为下一条 tensile 的输入

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
  --equil-steps 1000 \
  --hold-steps 1000 \
  --unload-steps 5000 \
  --indenter-radius-A 8.0 \
  --indenter-stiffness 5.0 \
  --initial-depth-A 0.0 \
  --target-depth-A 2.0 \
  --final-unload-depth-A 0.2 \
  --gamma 2.0
```

输出按取向分目录保存，例如 `run_output/w_indent/orientation_100/`，包含 `nanoindent_log.csv`、兼容旧脚本的 `load_depth.csv`、`summary.json`、`report.md`、`load_depth.png`、`load_depth_with_popin.png`、`trajectory.xyz`、`snapshots/`、`snapshots_png/` 和生成的 slab 结构。

新版压痕工作流支持：

- 单次运行中的 loading、可选 hold、unloading
- `nanoindent_log.csv` 里用 `phase=loading/hold/unloading` 区分阶段
- 默认输出 `trajectory.xyz`，并可用 `--traj-interval` 追加间隔帧
- 基于载荷突降或加载刚度突降的 pop-in 检测
- 面向 production 解读的 `report.md`
- `summary.json` 里给出球形几何接触面积近似硬度：
  - `max_depth_A`
  - `max_load_nN`
  - `residual_depth_A`
  - `unloading_stiffness_nN_per_A`
  - `work_loading`
  - `work_unloading`
  - `plastic_work_fraction`
  - `contact_area_A2`
  - `hardness_GPa`
  - `hardness_method=geometric_spherical_contact_area`
  - `pop_in_detected`

这里的 `hardness_GPa` 使用 `A = pi(2Rh - h^2)` 和 `H = Pmax/A`。它目前应理解为 Simulon 内部比较用的几何近似，不是严格校准后的实验 Oliver-Pharr 分析。塑性指标如果尚未实现，会明确输出 `plasticity_indicator_available=false`，不会伪造。

大体系自定义结构示例：

```bash
python run_scripts/w_indent.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --steps 10000 \
  --equil-steps 1000 \
  --hold-steps 1000 \
  --unload-steps 5000 \
  --indenter-radius-A 8.0 \
  --indenter-stiffness 5.0 \
  --initial-depth-A 0.0 \
  --target-depth-A 4.0 \
  --final-unload-depth-A 0.5 \
  --gamma 2.0 \
  --traj-interval 500 \
  --output-dir run_output/prod_w_indent_W31250
```

对 `--orientation custom`，当前实现要求输入 XYZ 对应的是正交立方盒。

一次跑完三个取向：

```bash
python run_scripts/w_indent.py --orientation 100 --replicas 6,6,4 --steps 5000 --equil-steps 1000 --indenter-radius-A 8.0 --indenter-stiffness 5.0 --initial-depth-A 0.0 --target-depth-A 2.0 --gamma 2.0
python run_scripts/w_indent.py --orientation 110 --replicas 6,6,4 --steps 5000 --equil-steps 1000 --indenter-radius-A 8.0 --indenter-stiffness 5.0 --initial-depth-A 0.0 --target-depth-A 2.0 --gamma 2.0
python run_scripts/w_indent.py --orientation 111 --replicas 5,5,3 --steps 5000 --equil-steps 1000 --indenter-radius-A 8.0 --indenter-stiffness 5.0 --initial-depth-A 0.0 --target-depth-A 2.0 --gamma 2.0
```

### 8. W 裂纹工作流

最小 smoke test：

```bash
python run_scripts/w_crack.py --smoke
python cuda_test/test_w_crack_smoke.py
```

W `[100]` 裂纹开口示例：

```bash
python run_scripts/w_crack.py \
  --orientation 100 \
  --replicas 8,8,4 \
  --steps 5000 \
  --equil-steps 500 \
  --crack-half-length-A 8.0 \
  --crack-gap-A 1.2 \
  --target-strain 0.02 \
  --gamma 2.0
```

输出按取向分目录保存，例如 `run_output/w_crack/orientation_100/`，包含 `crack_response.csv`、`summary.json`、`crack_response.png` 和生成的裂纹结构。

裂纹工作流现在也支持 `--traj-interval` 输出 `trajectory.xyz`。
`crack_response.csv` 中的 `stress_bar` 采用开裂拉伸为正的口径；内部 virial 原始符号保留在 `native_stress_yy_bar`。
裂纹报告还会跟踪 `stress_drop_ratio`、`crack_length_A` 和 `crack_extension_A`；做 DBTT 扫描前，先用这些指标确认至少一个温度点真的发生裂纹扩展。
`summary.json` 会把单次裂纹结果分类为 `brittle`、`ductile`、`opening_only`、`no_crack_growth` 或 `invalid`。当前 plasticity 指标会明确标记为不可用，因此不会强行给出 ductile DBTT 结论。

大体系自定义结构示例：

```bash
python run_scripts/w_crack.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --steps 10000 \
  --equil-steps 1000 \
  --crack-half-length-A 8.0 \
  --crack-gap-A 1.2 \
  --target-strain 0.03 \
  --gamma 2.0 \
  --traj-interval 500 \
  --output-dir run_output/prod_w_crack_W31250
```

### 9. W DBTT 温度扫描

最小 smoke test：

```bash
python cuda_test/test_w_dbtt_smoke.py
```

基于裂纹开口的温度扫描示例：

```bash
python run_scripts/w_dbtt_scan.py \
  --orientation 100 \
  --temperatures 100,200,300,400,500,600
```

输出写入 `run_output/w_dbtt/`，包含每个温度点的裂纹结果以及：

- `dbtt_summary.csv`
- `dbtt_summary.json`
- `dbtt_summary.png`

新版 DBTT 汇总不再只看峰值应力，还会重点汇总：

- `final_stress_bar`
- `stress_retention_ratio`
- `max_cmod_A`

对当前这套基于裂纹开口的 W DBTT 工作流，建议优先用上面三项判断脆-韧转变。`peak_stress_magnitude_bar` 仍然会保留，但不建议单独拿它下转变温度结论。
DBTT 汇总现在基于每个温度点的裂纹分类。如果所有温度点都是类似的 `opening_only`，`dbtt_candidate_temperature_k` 会保持为 `null`，`dbtt_status` 为 `not_identified`。

裂纹传播参数扫描：

```bash
python run_scripts/w_crack_sweep.py \
  --orientation custom \
  --structure run_output/prod_w_bulk_relax_W31250/orientation_custom/W_custom_relaxed.xyz \
  --box-length 79.28473306554223
```

大体系自定义结构示例：

```bash
python run_scripts/w_dbtt_scan.py \
  --orientation custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --temperatures 100,200,300,400,500,600 \
  --steps 5000 \
  --equil-steps 500 \
  --gamma 2.0 \
  --output-dir run_output/prod_w_dbtt_W31250
```

### 10. 批量运行与参数说明

用统一输出根目录批量跑四条工作流中的任意组合：

```bash
python run_scripts/w_batch_report.py \
  --workflows tensile,indent,crack,dbtt \
  --orientations 100,110,111 \
  --output-dir run_output/w_batch_report
```

所有参数含义和报告字段说明见：

- `W_WORKFLOWS_GUIDE.md`

大体系自定义结构批量运行示例：

```bash
python run_scripts/w_batch_report.py \
  --workflows tensile,indent,crack,dbtt \
  --orientations custom \
  --structure run_data/W/W31250.xyz \
  --box-length 80.0 \
  --output-dir run_output/w_batch_W31250
```

但对大体系正式生产，仍建议四条工作流分别提交，不要在一张 GPU 上一次性 batch 四个生产任务。

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
