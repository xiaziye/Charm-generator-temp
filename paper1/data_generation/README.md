# 论文一：独立连续数据生成模块

先打开[审阅入口](审阅入口_20261012.md)，可以直接看到配置、完整数学说明和实际生成结果。

入口配置在 `examples/*.toml`。修改配置后生成一批观测数据，查看实际 DAG、逐节点结构方程、校准前后边权、两段 O 集和结构反事实真值。完整数学定义见 [MATHEMATICS.md](MATHEMATICS.md)。

这个模块负责生成和审阅数据。后续可以把观测数据接到已知图的调整效率实验，或 `paper1` 方法入口。

## 先看哪些文件

| 文件 | 用途 |
|---|---|
| `examples/review_linear.toml` | 连续线性设计：31 列，带两段额外串联路径、一个并行中介和一个暴露后节点 |
| `examples/review_nonlinear.toml` | 保持相同结构参数，普通边可用 tanh/sin，查询因果路径使用 tanh，结局含 X×M 交互 |
| `examples/boundary_middle_confounding.toml` | 单独增加中间混杂 L→M、L→Y，检查图上可纳入条件的边界 |
| `schema.py` | 配置项、默认值和严格参数校验；拼错字段直接报错 |
| `generator.py` | 图生成、方程赋值、校准、观测/干预模拟、解析/Monte Carlo 真值 |
| `reporting.py` | 逐节点方程、真实 DAG、观测分布与相关图 |
| `run.py` | 多张图 × 多个抽样重复的运行、快照与产物保存 |
| `tests/test_generation.py` | 真值、方程重建、校准、配置和可复现性测试 |

## 如何运行

以下命令从 Git 仓库目录 `charm/` 执行：

```bash
.venv/bin/python -m paper1.data_generation.run \
  paper1/data_generation/examples/review_linear.toml

.venv/bin/python -m paper1.data_generation.run \
  paper1/data_generation/examples/review_nonlinear.toml

.venv/bin/python -m paper1.data_generation.run \
  paper1/data_generation/examples/boundary_middle_confounding.toml
```

默认输出分别在 `paper1_out/data_generation_review_v1/linear`、`nonlinear`、`boundary`。已有非空目录拒绝覆盖。要重新运行，给一个新的目录：

```bash
.venv/bin/python -m paper1.data_generation.run \
  paper1/data_generation/examples/review_linear.toml \
  --out paper1_out/my_new_design
```

`output.directory` 的相对路径以运行命令时的当前工作目录为准。`--out` 覆盖该字段；源 TOML 和解析后的实际设置会分别保存。

Python 3.11 使用内置 TOML 解析。Python 3.9/3.10 需要 `tomli`。已有虚拟环境可以直接运行；最小依赖见 `requirements.txt`。

## 配置项及其精确含义

| 配置组 | 主要设置 | 含义 |
|---|---|---|
| `[graph]` | `n_baseline` | 暴露前所有列数，包含 I、PM、PY、无关 U 和普通 C |
| | `baseline_mean_parents`、`baseline_max_parents` | 普通 C 内部随机 DAG 的 Poisson 父节点数参数和上限；实际入度另外记录 |
| | `n_shared`、`n_x_extra`、`n_m_extra`、`n_y_extra` | 共同直接父节点数量、每个端点额外父节点数量；额外父节点可以重叠 |
| | `n_inst`、`n_pm`、`n_py`、`n_irrelevant` | 独立根工具、纯 M/Y 预测变量、独立无关变量数量 |
| | `xm_chain_nodes`、`my_chain_nodes` | 在直接 X→M、M→Y 之外增加的串联路径内部节点数 |
| | `n_parallel`、`allow_parallel_mediators` | 并行 X→M2→Y 路径数量及允许开关 |
| | `n_post_descendants`、`allow_post_descendants` | X→D 的暴露后非中介数量及允许开关 |
| | `n_middle_confounders`、`allow_middle_confounding` | X→L→M 且 L→Y 的中间混杂数量及允许开关 |
| | `n_auxiliary_baseline_parents` | 每个额外路径/后代节点另有多少普通 C 父节点 |
| `[mechanism]` | `mode` | `linear` 或 `additive_nonlinear`；非线性模式可另外设置 X×M 交互 |
| | `xm_weight`、`my_weight`、`direct_weight`、`chain_weight`、`parallel_*_weight` | 受保护因果边的系数，校准不缩放这些边 |
| | `instrument_weight`、`predictor_weight` | 专属 I/PM/PY 边的初始系数；它们会参加基线贡献成组校准 |
| | `weight_min`、`weight_max`、`negative_edge_fraction` | 普通随机边初始幅度和负号抽样概率 |
| | `weak_fraction`、`weak_multiplier` | 普通边初始弱化的概率和倍数；校准后是否仍弱须看最终权重 |
| | `protected_function` | 查询主干、额外串联与并行因果边使用 id/tanh/sin 哪一种函数 |
| | `nonlinear_fraction`、`edge_functions` | 其他边选择非线性函数的概率及候选函数；实际每条边的函数会导出 |
| | `noise_distribution`、`noise_sd`、`noise_df` | 正态/Laplace/Student-t 噪声，统一以 noise_sd 为标准差 |
| | `intercept_sd`、`outcome_interaction` | 每个设计固定截距的抽样尺度，以及结局中的 γXM 系数 |
| `[calibration]` | `enabled`、`n_samples`、`target_signal_sd` | 用独立校准样本，把每个节点的基线父点贡献之和缩放到目标标准差 |
| | `scale_floor` | 信号标准差过低时停止并报错，避免不受控地放大系数 |
| `[experiment]` | `n_graphs`、`n_replicates`、`n_samples` | 独立图/机制设计数、每个固定设计的观测重复数、每份表的行数 |
| | 五类 `*_seed` | 图、机制、校准、观测数据、反事实真值的独立随机流 |
| | `x0`、`x1`、`truth_samples`、`chunk_size` | 暴露对比、非线性反事实样本数与分块大小 |
| `[output]` | `csv`、`plot`、`preview_rows` | 是否额外导出完整 CSV/图、预览多少行；完整 NPZ 总会保存 |
| | `max_plot_nodes`、`max_covariance_nodes` | 绘图和精确协方差的规模上限；跳过时记录原因 |

**数量与集合归属分开。** `n_shared` 是共同直接父节点数，实际 `O_M ∩ O_Y` 由最终图计算。`design.json` 中另有中介段独有、结局段独有、共享 O 集与不属于两段 O 集的基线列。

**校准的对象是基线贡献这一组的标准差。** 它不是总体 R² 或信噪比，也不保证每条边独立达到某个效应强度。原始边权、最终边权、缩放倍数和初始弱边标记均保存。需要研究原始系数变化时可关闭校准。

## 结果如何检查

每次运行根目录包含：

- `REVIEW.md`：中文审阅报告、真值表、集合归属与图。
- `config.toml`：输入配置快照；`resolved_config.json`：补全默认值和覆盖参数后的配置。
- `manifest.json`：代码哈希、Git 版本、环境、随机流、每份数据的哈希和运行状态。

每张图的 `graph_000/` 等目录包含：

- `design.json`：完整边/函数/截距/权重/校准记录、图上合法性、真值及 MC 标准误。
- `equations.md`：真实使用的全部逐节点结构方程，可逐项与代码对照。
- `data_r000.npz`、可选 `data_r000.csv`：观测数据；不同 `r` 是同一固定方程下的新样本。
- `preview_r000.csv`、`profile_r000.json`：头部数据和各列均值、标准差、范围。
- `graph.png/svg`、`data_overview.png/svg`：真图和 X/M/Y 分布、散点及部分相关矩阵。
- `population_moments.npz`：线性且规模不超限时的精确均值和协方差；总体 OLS 投影另外写入 `design.json`，可查合法臂是否有总体偏差。

中间混杂例的结构效应仍能计算，但图上调整对无效，会标为 `boundary_middle_confounding`。非线性或非高斯设计的边际可分性标为 `not_certified`，不会因为方程“看起来可加”就声称满足效率定理条件。

## 核验

```bash
.venv/bin/python -m unittest discover -s paper1/data_generation/tests -v
```

测试包括解析路径真值与同噪声反事实、非线性真值与独立积分、协方差、种子隔离、方程元数据重建、校准、O 集合法性和导出产物复现。

当前提供的是完整的连续生成过程及可审阅产物。正式实验的场景和终点应从已有实验设计确定；增加图或样本规模本身不能代替结论一的数学证明。
