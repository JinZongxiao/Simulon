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
| **W 晶界** | 新增严格 CSL `[001]` bicrystal 构型、刚体平移搜索和晶界能报告 |
| **W 结构基线** | 新增生产级 pure-W 结构基线矩阵：bulk、surface、defect、crack/notch seed 和 GB search |
| **ODS-W DFT 数据集** | 新增 backend-neutral 的 ODS-W DFT task 导出，支持 QE/VASP 模板 |
| **QE 标签入口** | 新增单个 Quantum ESPRESSO task runner，将能量、力、应力解析为统一 `dft_label.json` |
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
- **ODS-W 数据集入口**：生成 backend-neutral DFT task 目录，为后续 ODS-W 机器学习势训练准备结构和标签接口。
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
  w_structure_builder.py  # 纯 W bulk/surface/defect/CSL-GB/crack/notch 结构生成器
  restart.py              # save_checkpoint / load_checkpoint
  writer.py / output_logger.py / eam_parser.py / ...

postprocess/
  stress_strain.py        # 应力应变摘要 + PNG 绘图
  indentation.py          # 载荷-深度摘要 + PNG 绘图
  crack.py                # 应力-CMOD 摘要 + PNG 绘图
  dbtt.py                 # 温度扫描聚合 + PNG 绘图
  grain_boundary.py       # 晶界 excess energy 报告
  dft_qe.py               # Quantum ESPRESSO 输出解析 + 统一 DFT 标签 JSON

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
  build_w_structure.py    # 纯 W 结构生成器 CLI
  w_structure_baseline.py # 生产级 pure-W 结构基线矩阵
  w_gb_search.py          # CSL 晶界刚体平移搜索
  build_odsw_dft_dataset.py # ODS-W DFT task 数据集导出
  run_dft_qe_task.py      # 单个 QE task 运行 + dft_label.json 生成
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

### 6. 纯 W 结构生成器

先生成可复用的 W 结构，再进入力学模拟：

```bash
python run_scripts/build_w_structure.py --kind bulk --orientation 100 --replicas 10,10,10
python run_scripts/build_w_structure.py --kind surface --orientation 110 --replicas 10,10,6 --vacuum-A 30
python run_scripts/build_w_structure.py --kind vacancy --orientation 100 --replicas 10,10,10 --vacancy-count 5
python run_scripts/build_w_structure.py --kind interstitial --orientation 100 --replicas 8,8,8 --interstitial-count 2
python run_scripts/build_w_structure.py --kind substitution --orientation 100 --replicas 8,8,8 --substitution-element Re --substitution-count 8
python run_scripts/build_w_structure.py --kind void --orientation 100 --replicas 12,12,12 --void-radius-A 8
python run_scripts/build_w_structure.py --kind bicrystal --gb-plane 3,1,0 --replicas 8,6,6
python run_scripts/build_w_structure.py --kind crack --orientation 100 --replicas 20,10,10 --crack-half-length-A 30 --crack-opening-A 2
python run_scripts/build_w_structure.py --kind notch --orientation 100 --replicas 20,10,10 --notch-radius-A 10 --notch-depth-A 10
```

在同一个 case 目录里附加固定盒子的几何弛豫：

```bash
python run_scripts/build_w_structure.py \
  --kind vacancy \
  --orientation 100 \
  --replicas 8,8,8 \
  --vacancy-count 2 \
  --relax \
  --relax-steps 500 \
  --relax-force-threshold 0.05
```

输出位于 `run_output/w_structure_builder/<case>/`：

- `structure.xyz`
- `summary.json`
- `composition.csv`
- `preview.png`
- 可选 `relaxed_structure.xyz`
- 可选 `relaxation.csv`
- 可选 `relax_summary.json`

`--kind bicrystal` 目前是严格 CSL 周期的 BCC `[001]` 对称倾转晶界种子。默认 `--gb-plane 3,1,0` 会生成 `Sigma5(310)[001]`，失配角为 `36.8699 deg`。其它晶界面必须是互素正整数 `(h,k,0)`。

这个 builder 可以通过 `--relax` 做固定盒子的最速下降几何弛豫，用来消除建模后的局部大力。但它不替代正式 NVT/NPT 生产弛豫。晶界生产模拟仍然需要刚体平移搜索和结构弛豫。位错生成器留到下一阶段单独实现。

Smoke test：

```bash
python cuda_test/test_w_structure_builder_smoke.py
```

### 6a. ODS-W DFT 数据集导出

`run_scripts/build_odsw_dft_dataset.py` 用于为第一版 ODS-W 机器学习势数据集准备 DFT-ready 结构任务。它只负责生成结构、统一 metadata、后端输入模板和可读报告，不直接运行 DFT，也不把 Simulon 绑定到某一个 DFT 软件。

Simulon 的设计是：

- 顶层任务目录统一为 `dft_tasks/<task_id>/`
- 通用结构和元数据放在 `common/`
- QE、VASP 或未来 CP2K 只是每个 task 下面的 backend writer
- 后续 MLP 训练读取统一标签 `dft_label.json`，而不是直接依赖某个 DFT 后端格式

当前推荐的第一版可验证化学体系是 `W-Zr-Y-O`。先把 `YZrWO` 这类最小可控体系跑通，再扩展到 `Ti/Hf` 或 `Er`，这样更容易判断结构生成、DFT 标签和后续 MLP 误差来源。

示例：导出一个 ODS-W pilot dataset。

```bash
python run_scripts/build_odsw_dft_dataset.py \
  --campaign pilot_diverse \
  --replicas 8,8,8 \
  --ods-a-element Zr \
  --ods-b-element Y \
  --oxide-formulas ABO3,A2B2O7 \
  --particle-radii-A 5.0,7.0 \
  --oxide-lattice-params-A 4.4,4.8 \
  --interface-clearances-A 0.8,1.2 \
  --dft-backends qe,vasp \
  --output-dir run_output/odsw_dft_dataset_WZrYO
```

参数选择：

- `--campaign interface_grid`：只扫 ODS 界面几何，适合快速检查结构生成器。
- `--campaign pilot_diverse`：推荐用于第一版 MLP 数据集，因为它会覆盖 pure-W bulk、弹性应变、rattle 热扰动近似、表面、点缺陷、稀释 Zr/Y 溶质和 ODS-W 界面。
- `--ods-a-element`：ABO 氧化物中的 A 位元素，例如 `Zr/Ti/Hf`。
- `--ods-b-element`：ABO 氧化物中的 B 位元素，例如 `Y/Er`。
- `--oxide-formulas`：氧化物化学式族，例如 `ABO3,A2B2O7`。
- `--particle-radii-A`：嵌入 W 基体的氧化物颗粒半径，单位 Å。
- `--interface-clearances-A`：氧化物颗粒与 W 基体界面的最小清理距离，避免初始原子严重重叠。
- `--dft-backends`：要写出的 DFT 后端输入模板，例如 `qe,vasp`。

主要输出：

- `manifest.json`：整个 dataset 的机器可读摘要。
- `metadata.csv`：每个 DFT task 一行，包含化学式、颗粒半径、界面距离、原子数、组成、task 目录和后端输入路径。
- `dataset_report.md`：给人读的数据集范围、化学空间和 DFT 标签要求说明。
- `structures/<task_id>/`：Simulon builder 原始输出、预览图、composition 和 interface sanity check。
- `dft_tasks/<task_id>/common/`：后端无关的 `structure.xyz` 和 `builder_summary.json`。
- `dft_tasks/<task_id>/qe/`：Quantum ESPRESSO `pw.in` 模板。
- `dft_tasks/<task_id>/vasp/`：`POSCAR`、`INCAR.template`、`KPOINTS.template` 和 `POTCAR.required.txt`。

后续训练 MLP 需要的 DFT 标签不是只有能量，还必须包括：

- 总能量 `energy_eV`
- 每个原子的力 `forces_eV_A`
- 应力张量 `stress_GPa`
- 最终晶胞 `cell_A`
- 元素种类 `species`
- 原子坐标 `positions_A`

注意：这里导出的 QE/VASP 输入只是起始模板，不等价于生产级 DFT 设置。正式计算必须记录赝势、截断能、k 点密度、展宽方式和收敛阈值，并做基本收敛性检查。

Smoke test：

```bash
python cuda_test/test_odsw_dft_dataset_smoke.py
```

### 6b. QE DFT 标签 runner

生成 `dft_tasks/<task_id>/qe/pw.in` 后，可以用 `run_scripts/run_dft_qe_task.py` 跑单个 QE task，并把 QE 输出转换为统一的 `dft_label.json`。

服务器上已安装 QE 环境时的示例：

```bash
source /public/home/normal_bgd/J1N/software/load_dft_qe_env.sh
python run_scripts/run_dft_qe_task.py \
  run_output/odsw_dft_dataset_WZrYO/dft_tasks/<task_id> \
  --np 8 \
  --omp 1 \
  --timeout 7200
```

主要输出：

- `dft_tasks/<task_id>/qe/qe.out`：QE 原始输出。
- `dft_tasks/<task_id>/qe/qe_status.json`：运行状态、返回码、耗时和标签是否可用。
- `dft_tasks/<task_id>/dft_label.json`：后端无关的 MLP 标签文件。

`dft_label.json` 的关键字段：

- `backend`：当前为 `qe`。
- `energy_eV`：总 DFT 能量，单位 eV。
- `forces_eV_A`：每个原子的力，单位 eV/Å。
- `stress_GPa`：应力张量，单位 GPa，由 QE 的 kbar 输出转换得到。
- `cell_A`、`species`、`positions_A`：DFT 输入结构。
- `converged`、`job_done`：QE 是否完成且 SCF 收敛。
- `label_ready`：只有能量、力、应力都存在，且没有 NaN/Inf 时才为 true。

Smoke test：

```bash
source /public/home/normal_bgd/J1N/software/load_dft_qe_env.sh
python cuda_test/test_dft_qe_smoke.py
```

这个 smoke test 总会检查 parser。如果环境里能找到 `pw.x`、`mpirun` 和 W 的 UPF 赝势，它还会实际跑一个 2 原子 BCC W 的 QE SCF，并输出真实 `dft_label.json`。这个结果只用于验证链路，不用于生产物理结论。

### 6c. 生产级 pure-W 结构基线

在做 ODS-W 嵌入或缺陷力学对比前，先跑完整 pure-W 结构基线矩阵：

```bash
python run_scripts/w_structure_baseline.py \
  --preset production \
  --orientation 100 \
  --relax-method fire \
  --relax-steps 3000 \
  --relax-force-threshold 0.05 \
  --output-dir run_output/prod_w_structure_baseline
```

production preset 会生成并弛豫：

- `bulk_100`
- `surface_100_z`
- `vacancy_1`
- `interstitial_1`
- `void_r8`
- `crack_seed`
- `notch_seed`
- `bicrystal_seed_sigma5_310_001`
- `gb_search_sigma5_310_001`

主要输出：

- `structure_baseline.csv`
- `summary.json`
- `report.md`
- 每个 case 的 `structure.xyz`、`relaxed_structure.xyz`、`summary.json`、`relax_summary.json`、`preview.png`

`summary.json` 会区分 `acceptance_pass` 和 `production_ready`。`acceptance_pass` 表示结构生成成功、弛豫无 NaN/Inf、能量没有升高；`production_ready` 还要求力收敛或 GB energy 判据有效。这个区分对短步 smoke 尤其重要。

Smoke test：

```bash
python cuda_test/test_w_structure_baseline_smoke.py
```

### 6d. W CSL 晶界搜索

晶界模拟不要直接把原始 bicrystal seed 当成最终模型。先做刚体平移搜索，对每个候选结构弛豫，再按 excess GB energy 排序：

```bash
python run_scripts/w_gb_search.py \
  --gb-plane 3,1,0 \
  --replicas 8,6,6 \
  --translations-x 5 \
  --translations-z 3 \
  --relax-method fire \
  --relax-steps 500 \
  --relax-force-threshold 0.05 \
  --output-dir run_output/w_gb_search
```

关键参数：

- `--gb-plane`：互素正整数 `(h,k,0)` CSL 晶界面。默认 `3,1,0` 是 `Sigma5(310)[001]`。
- `--translations-x`, `--translations-z`：晶界面内刚体平移网格。
- `--gb-overlap-cutoff-A`：两个周期晶界附近的近距离原子删除阈值。
- `--relax-method`：候选结构弛豫算法。复杂 GB 弛豫推荐 `fire`；`sd` 用于兼容早期运行。
- `--bulk-energy-per-atom-ev`：默认 `auto`。脚本会用同一个 EAM 文件自动计算匹配的 BCC bulk reference，避免硬编码 cohesive energy。
- `--bulk-reference-replicas`：自动 bulk reference 的可选超胞尺寸。

输出位于 `run_output/w_gb_search/<case>/`：

- `candidates.csv`
- `best_structure.xyz`
- `best_relaxed_structure.xyz`
- `gb_energy_report.json`
- `summary.json`

Smoke test：

```bash
python cuda_test/test_w_gb_search_smoke.py
```

### 7. W 拉伸工作流

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
- `report.md`
- `stress_strain.png`
- `lateral_stress.png`
- 自动生成的取向结构，如 `W_100_generated.xyz`

CSV 中包含原生有符号应力列（`stress_xx_bar`、`stress_yy_bar`、`stress_zz_bar`）、拉伸为正的展示列（`tension_xx_bar`、`tension_yy_bar`、`tension_zz_bar`）、盒长、能量、温度和维里张量对角元。画拉伸应力应变曲线和汇报时只使用轴向 `tension_xx_bar` / `tension_bar`。不要把 `xx`、`yy`、`zz` 平均成一条拉伸曲线；`tension_yy_bar` 和 `tension_zz_bar` 是横向应力诊断，用来检查 stress-free barostat。`stress_*` 保留内部压缩为正的 virial 符号，主要用于诊断。

新版拉伸工作流现在会：

- 先通过 `--equil-steps` 做零压预平衡
- 在 `stress_xx_bar` 中输出相对于平衡初态的原生有符号应力
- 额外输出拉伸为正的展示列：`tension_xx_bar`、`tension_yy_bar`、`tension_zz_bar`
- 将 `stress_strain.png` 输出为只包含轴向 `tension_xx_bar` 的主曲线
- 将 `lateral_stress.png` 输出为 `tension_yy_bar` 和 `tension_zz_bar` 的横向控压诊断图
- 自动生成 `report.md`，写清应力符号、主要结果和推荐画图列
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

### 8. W 纳米压痕工作流

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

### 9. W 裂纹工作流

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

### 10. W DBTT 温度扫描

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

### 11. 批量运行与参数说明

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
