# Simulon

[中文] | [English](README.md)

一个基于 PyTorch 的轻量级分子动力学（MD）引擎，并提供可选的自定义 CUDA 加速内核。Simulon 注重清晰的代码结构、可扩展性与科研/工程实用工作流。

核心能力
- PyTorch 优先：所有状态均存放在张量中，CPU/GPU 切换自然顺滑。
- 可选 CUDA 加速：通过编译扩展（`simulon_cuda`）提供 LJ、EAM 与邻域搜索的高性能实现。
- 力场模块化：内置 Lennard-Jones（CPU/CUDA）、EAM（CUDA）、Born–Mayer–Huggins，以及用户自定义对势模板。
- 邻域搜索：包含 PBC 处理与 GPU 加速实现。
- 积分器：简单稳健的速度 Verlet（NVT，带阻尼 gamma）。
- I/O 与工具：XYZ 读写、日志记录、诊断绘图、数据集准备辅助等。
- 机器学习势：示例性集成，可微调类似 CHGNet 的模型并用学习到的力进行 MD。

仓库结构
- `core/`：MD 基础模块：力场、积分器、模拟器、分析器。
- `io_utils/`：文件读写、日志、解析器（ASE/pymatgen）、EAM 解析、结构构造工具。
- `cuda source/`：LJ/EAM/邻域搜索的 C++/CUDA 内核与 Python 绑定。
- `run_scripts/`：可直接运行的示例：LJ、用户自定义对势、ML 势以及诊断绘图。
- `run_data/`：示例体系（Ar、Cu、W 等）。
- `simulation_agent/`：中英文交互助手，帮助生成与分析 MD 任务。
- `cuda_test/`：CUDA 后端的最小测试与示例。

环境依赖
- Python 3.10+（已在 3.11 测试）
- PyTorch（可选 CUDA）。参考 https://pytorch.org 获取安装方式。
- 第三方包：numpy、scipy、matplotlib、ase、pymatgen、tqdm
- 可选（用于 ML 示例）：chgnet

快速安装
- 基础 Python 依赖：
  - `pip install torch`（按照你的 CUDA/CPU 环境选择安装）
  - `pip install numpy scipy matplotlib ase pymatgen tqdm`
  - 可选：`pip install chgnet`
- CUDA 扩展（可选但推荐以获得更好性能）：
  - 先决条件：与你的 PyTorch 匹配的 C++ 工具链与 CUDA 工具包。
  - 本地构建：`python setup.py build_ext --inplace`
  - Windows 用户：确保已安装 MSVC Build Tools，且 CUDA 在 PATH 中。
  - 说明：仓库已包含适用于 Python 3.11/Windows 的预编译文件 `simulon_cuda.cp311-win_amd64.pyd`，若环境匹配可直接使用，否则建议从源码编译。

桌面应用（Windows/macOS）
- 仓库新增了基于 Tk 的跨平台桌面 GUI，入口为 `simulon_desktop.py`。它可以直接加载/编辑内置 JSON 模板、选择输出目录，并在不使用命令行的情况下运行 Lennard-Jones 与用户自定义对势模拟。
- 源码方式启动：
  - `python simulon_desktop.py`
- 在目标系统上使用 PyInstaller 构建可分发桌面应用：
  - `pip install pyinstaller`
  - `python packaging/build_desktop.py`
- 产物位置：
  - Windows：`dist/SimulonDesktop/SimulonDesktop.exe`（或等效 PyInstaller 输出）
  - macOS：`dist/SimulonDesktop.app`
- 打包说明：
  - 构建时会把 `run_scripts/` 的 JSON 模板和 `run_data/` 示例体系一起打包，生成后的应用可以直接打开并使用。
  - 需要在哪个平台分发，就在哪个平台执行构建：Windows 生成 Windows 可执行文件，macOS 生成 macOS `.app`。

快速开始
1）Lennard-Jones MD
- 如需修改输入，在 `run_scripts/lj_run.json` 中调整（结构路径、盒长、LJ 参数、截断、温度、步数、输出目录等）。
- 运行：
  - `python run_scripts/lj_run.py --config run_scripts/lj_run.json`
- 输出（默认写入 `run_data/output/`）：
  - 能量曲线 PNG、轨迹 `MD_traj_<timestamp>.xyz`、力 `forces_<timestamp>.xyz` 与日志等。

2）用户自定义对势
- 在 `run_scripts/user_defined_run.json` 中定义你的势函数，例如 `0.5 * k * (r - 1)**2`，并设置每对原子的参数。
- 运行：
  - `python run_scripts/user_defined_run.py --config run_scripts/user_defined_run.json`

3）机器学习势（示例）
- 编辑 `run_scripts/mlps_run.json`，指定 AIMD 的位置信息/力文件及训练超参。
- 需要额外依赖（如 `chgnet`）。
- 运行：
  - `python run_scripts/mlps_run.py --config run_scripts/mlps_run.json`

4）诊断绘图
- 一键生成能量/力/MSD/RDF/度分布等综合诊断图：
  - `python run_scripts/plot_md_diagnostics.py --steps 500`
- 图片将输出到 `run_data/output/plots_YYYYmmdd_HHMMSS/`。

配置要点
- JSON 配置中常见字段：
  - `data_path_xyz`：输入结构（.xyz）
  - `box_length`：立方盒长度（Å）
  - `pair_parameter`：对参数；LJ 需 `epsilon`/`sigma`；自定义势使用你在公式中的参数名（如 `k`）。
  - `potential_formula`：仅用于自定义势（例如 `0.5 * k * (r - 1)**2`）。
  - `cut_off`：邻域截断（Å）
  - `dt`：时间步长（fs 或你保持一致的自定义单位）
  - `temperature`：NVT 目标温度，可为标量或按类型的向量
  - `gamma`：NVT 摩擦系数
  - `num_steps`、`print_interval`、`output_save_path`

使用提示
- GPU/CPU：若可用将自动选用 CUDA，否则退回 CPU。
- PBC：周期性边界已在内部处理；请确保 `box_length` 与体系一致。
- 大规模体系建议启用 CUDA 扩展以获得更好性能。

常见问题
- CUDA 构建失败：请确保 CUDA 版本与 PyTorch 对应，且 MSVC/Clang 工具链安装正确。
- 缺少包：确认已安装所需依赖，运行脚本时保证仓库根目录在 `PYTHONPATH` 中或在仓库根目录下执行。

贡献
- 欢迎提交 Issue/PR。反馈问题时请尽量提供最小可复现示例或小型输入结构。

