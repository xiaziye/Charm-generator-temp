# 论文一：已知图的线性效率实验

本模块将现有数据生成器接入原 D3 的重复抽样实验。一次实验固定图、边权、函数、截距和校准结果；每次重新生成观测噪声，让所有方案使用同一份数据，最后比较 NIE、NDE 和 TE 的估计表现。

先看 [linear_pilot.toml](examples/linear_pilot.toml) 和 [试运行记录](试运行记录_20261016.md)。数学背景沿用 [生成器说明](../data_generation/MATHEMATICS.md)。正式实验的结构与终点依据现有实验一计划，本次增加的是可配置的执行接口；默认输出是探索性结果。

## 快速运行

在仓库 `charm/` 根目录执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m paper1.efficiency_experiment \
  paper1/efficiency_experiment/examples/linear_pilot.toml
```

默认输出到 `paper1_out/efficiency_linear_pilot_20261016/`。若该目录已运行过，用新目录：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m paper1.efficiency_experiment \
  paper1/efficiency_experiment/examples/linear_pilot.toml \
  --out paper1_out/efficiency_linear_pilot_next
```

不覆盖已有非空输出。设置 BLAS 线程数是为了避免多个重复任务同时再各开一组矩阵运算线程；它不会改变统计模型。

## 文件与接口

| 文件 | 负责什么 | 可复用入口 |
|---|---|---|
| `schema.py` | 严格配置校验，未知字段报错 | `load_config`、`from_dict`、`ExperimentConfig` |
| `io.py` | 文件校验、固定模型、逐重复提供样本 | `load_inputs`、`GraphInput.sample`、`read_npz` |
| `arms.py` | 原方案与自定义调整集，真图合法性复核 | `build_arms`、`AdjustmentArm` |
| `estimators.py` | 两段拟合、效应、区间与数值诊断 | `fit_estimate`、`register_estimator`、`EstimatorConfig` |
| `metrics.py` | 固定模型内的重复统计、配对方差比 | `summarize`、`paired_variance_ratio` |
| `endpoints.py` | 原五终点的探索性数值诊断 | `evaluate_endpoints` |
| `legacy_d3.py` | 原 48 格 D3 的独立结构适配 | `cells`、`make_design`、`simulate` |
| `run.py` | 组织运行、并行、快照、逐条记录、报告 | `run`、`run_experiment` |

`legacy_d3.py` 移植自 `paper1/bench/designs.py` 的原 D3 部分；全库测试对 48 格的节点顺序、边权、角色、真值和同种子样本逐一比对。运行时无需导入原发现管线。

## 三种数据来源

| `source.mode` | 输入 | 固定什么、重抽什么 |
|---|---|---|
| `saved` | 生成输出根目录，含 `manifest.json`、`graph_*/design.json` 和 NPZ | 使用已有观测重复，不自动补样本 |
| `generate` | 当前生成器 TOML | 固定图/机制/校准，按实验种子生成新的观测重复 |
| `legacy_d3` | 原 D3 格子编号和实验规模 | 复用原 48 格结构，在新随机流上重复抽样 |

TOML 中 `source.path`、`source.generator_config` 和 `output.directory` 的相对路径，均以**实验 TOML 所在目录**为基准。只有命令行 `--out` 以当前工作目录为基准。

- `graph_ids=[]`：选全部图；旧 D3 编号为 `0..47`，来自 `legacy_d3.cells()`。
- `sample_sizes=[500,1000]`：同一重复先取最大样本，再使用前 500 行和前 1000 行。各方案与不同 n 保持配对。
- `n_replicates=0`：saved 使用全部已有重复；generate 沿用生成器的重复数。legacy_d3 必须明确填写正数。
- generate 的图/机制/校准/真值种子仍来自生成 TOML；**新的观测数据种子来自 `execution.seed`**，派生标签为 44。没有混用生成器审阅数据的 `data_seed`。
- 暴露对比从输入真值的 `x0/x1` 读取；legacy_d3 使用 0 与 1。不会只修改拟合对比而沿用另一对比的真值。
- saved 校验样本哈希、形状与列顺序，使用 `allow_pickle=False`。超出已有重复或行数时报错。

## 原对照方案与自定义方案

| 名称 | 中介段 | 结局段 | 来源 |
|---|---|---|---|
| `O` | 本段 O 集 | 本段 O 集 | 原实验分段方案 |
| `FULL` | 全部允许调整的非后代池 | 同左 | 原实验全调 |
| `SHARED` | 两段 O 集的并集 | 同左 | 原实验共用并集 |
| `SHAREDm` | 并集中去掉专门的纯 M 预测节点 | 同左 | 原实验 Diop 式共用 |
| `OMISS` | O 集去掉明确标记的弱父节点 | 同左 | 原实验弱父点省略诊断 |

`OMISS` 需要 `weak_M_parent/weak_Y_parent` 角色，旧 D3 有弱父点的格子具备该标签。当前随机生成器的 `weak_draw` 是边的初始弱化标签，不能代替这个角色；缺少角色时记为 `unavailable`，而不是偷偷生成另一种 OMISS。没有纯 M 预测节点时，SHAREDm 与 SHARED 相同；共享方案优劣终点仅在有纯 M 预测节点的面板评价。

完整图重新计算标准方案并复核合法性，不能只相信输入 JSON 中的名字。自定义集合使用新名字：

```toml
[[arms]]
name = "CUSTOM_PAIR"
kind = "custom"
CM = ["C001", "C003"]
CY = ["C001", "C004"]
```

原五个名称保留给原策略，不能用 `custom` 改写其含义。默认不拟合不合法的调整对，记录 `skipped_non_admissible`。显式打开 `execution.allow_non_admissible=true` 才做有偏诊断，并保留不合法标记。

## 拟合、标准误和估计器扩展

内置 `linear_ols` 用两次带截距 OLS 估计：中介方程的 X 系数记为 a；结局方程的 M、X 系数分别记为 b、d。暴露变化量是 Δx=x₁−x₀，则 NIE=Δx×a×b、NDE=Δx×d、TE=NIE+NDE。

这些点估计与旧 `product_estimate` 一致。旧入口默认输出每单位 X，本接口显式应用暴露对比，标准误随 |Δx| 缩放。

| `estimator.se_method` | 标准误与区间 |
|---|---|
| `sandwich`（默认） | 两段回归按同一人的残差贡献堆叠，分别进行 HC1 自由度校正，保留跨方程协方差；乘积用 delta 法，区间用正态近似 |
| `bootstrap` | 对行有放回抽样，同一重复内各方案使用同一组抽样索引；标准误为 bootstrap SD，区间为百分位区间 |
| `none` | 只返回点估计；覆盖率与区间宽度保留缺失，不写成 0 |

默认标准误是这里新增的推断实现；没有直接沿用旧 `product_se` 忽略两段协方差的近似。它仍是渐近近似，区间实际覆盖率由重复实验评价。bootstrap 任一次拟合失败会使该次观测重复的该方案失败，避免静默删除坏的 bootstrap 抽样。

OLS 要求 n 大于两段各自的系数数量（含截距）、设计矩阵满秩且条件数不超过 `max_condition_number`；使用 SVD 求解与诊断。p≥n 的 FULL 会在复制大预测矩阵之前明确失败，不自动改成岭回归或伪逆方法。

生成机制非线性或含 X×M 交互时，默认拒绝内置线性估计器。`allow_misspecification=true` 允许用于错设诊断；实际错设标记会传到汇总并被原线性终点评价排除。线性非高斯模型可以做 OLS 实验，但不能据此认证生成器的非参数可分条件。

单次拟合 API：

```python
from paper1.efficiency_experiment import EstimatorConfig, fit_estimate

result = fit_estimate(
    data, columns, CM=["C001"], CY=["C001", "PY001"],
    x="X", m="M", y="Y", x0=0.0, x1=1.0,
    config=EstimatorConfig(se_method="sandwich"),
)
```

估计器接收观测数据、列名、调整集和暴露对比；不接收图、真实边权或效应真值。返回 JSON 兼容字典，包含 NIE/NDE/TE、各自 `se_*` 和 `ci_*`、回归系数、两段秩/条件数/自由度、耗时。缺失不确定性使用 `None`；95% 水平额外提供 `ci95_*` 兼容键。

扩展新估计方法时，用新名字注册，修改自己的拟合实现即可：

```python
from paper1.efficiency_experiment import register_estimator, load_config, run_experiment

@register_estimator("my_estimator")
def my_estimator(data, columns, CM, CY, *, x, m, y, x0, x1, config):
    options = config.options  # 来自 [estimator.options]，供自定义超参数使用
    # 在这里拟合自己的模型，并计算同一种自然效应分解。
    # 必须返回有限的 NIE/NDE/TE，且 TE=NIE+NDE；不返回 numpy 数组等不可序列化对象。
    # 还应返回 se_NIE/ci_NIE 等不确定性字段，无此功能时明确用 None。
    raise NotImplementedError("替换为自己的拟合实现")

config = load_config("my_experiment.toml")  # estimator.name = "my_estimator"
run_experiment(config, config_directory="配置文件所在目录")
```

注册在进程内进行；CLI 不自动导入任意第三方代码。需要插件时使用这样的 Python 驱动脚本，在调用运行器前 import/注册插件。`options` 支持 JSON 兼容的 TOML 参数；多线程模式下插件应避免共享可变状态。

## 汇总指标与原实验终点

统计单元是“固定图/机制 + 样本数 + 估计器”，不同图的估计结果不混算方差。每个方案分别报告：均值、偏差、经验 SD/方差、MSE/RMSE、MAE、平均模型标准误、区间覆盖率/宽度、耗时、失败/跳过比例，以及相应 Monte Carlo 标准误。

相对 O 的方差比使用两方案共同成功的重复，保留配对依赖计算 delta 近似 MCSE。若有拟合失败，这个比值是“条件于两方案成功”的统计，不能掩盖失败率。分母方差为 0、重复不足或缺少区间均明确记录缺失原因。

原五终点保留到 `endpoints.json`，按 n、估计器及 NIE/NDE 分开评价：

| 终点 | 原 A2 数值诊断标准 |
|---|---|
| E1 全调与 O | 各格 FULL/O 方差比最小值 ≥0.95 |
| E2 近工具收益 | 有近工具格中，O 相对 FULL 的方差降幅中位数 ≥20% |
| E3 理论预测相等格 | 图条件事先预测的相等格，FULL/O 方差比在 [0.95,1.05] |
| E4 与较好共用方案比较 | 有纯 M 预测变量的格，O/min(SHARED,SHAREDm) 方差比最大值 ≤1.05 |
| E5 合法性与 O 偏差 | 主方案合法，O 的最大绝对偏差不超过配置倍数的均值 MCSE（原 A2 取 3） |

不满足最小重复数、配对记录不完整、缺少适用格或处于错设诊断时，不会给出不加限定的“通过”。输出状态为 `passed/failed/inconclusive/not_evaluated`，并逐项列原因。即使部分终点 `passed`，整个运行仍记录 `confirmation=false`；新试运行没有复现旧预登记随机序列，也没有检验论文一的非参数有效影响函数方差定理。

## 输出与可追溯性

```text
output/
├── config.toml                 # 原实验配置（从文件运行时）
├── resolved_config.json        # 补全默认值后的配置
├── manifest.json              # 状态、代码哈希、环境、种子协议、输入追溯、警告
├── replicates.jsonl            # 每次拟合一行（可关闭）
├── summary.json / summary.csv  # 每个固定模型/样本量/方案的统计
├── comparisons.csv            # 配对方差比和MCSE
├── endpoints.json             # 原5终点的数值诊断
├── REPORT.md                  # 可阅读的运行报告
└── graph_000/
    ├── design.json            # 固定机制与真值
    └── arms.json              # 两段变量名单、合法性、图论预测
```

数据按重复生成与释放，最多预取与 workers 成比例的样本；拟合记录按图汇总后释放，JSONL 逐条写入，不把全部 48 格的大数据保存在内存。改变 workers 不改变样本种子和结果输出顺序，耗时字段当然会变化。

`execution.on_error="continue"` 保存错误并继续其他方案/重复；`"raise"` 停止并在 manifest 中标记 failed。整个流程完成不等于所有拟合成功：成功、失败、不可用和非法跳过分别记录。

## 配置样例

| 文件 | 用途 | 本次是否运行 |
|---|---|---|
| [linear_pilot.toml](examples/linear_pilot.toml) | 新生成器，2 图 × 2 个 n ×100 重复 ×4方案 | 已运行 |
| [saved_review.toml](examples/saved_review.toml) | 读取原审阅文件，每图只有2重复 | 已运行，仅接口检查 |
| [legacy_d3_pilot.toml](examples/legacy_d3_pilot.toml) | 原D3选8格，100重复，含原5方案 | 已运行 |
| [legacy_d3_full.toml](examples/legacy_d3_full.toml) | 原D3全部48格、两个n、2000重复 | 仅校验配置，未执行 |
| [boundary_diagnostic.toml](examples/boundary_diagnostic.toml) | 中间混杂，不合法方案默认跳过 | 已运行 |

## 独立仓库与测试

要复制到先前的生成器临时仓库，只需添加本目录。最小结构如下；`paper1/__init__.py` 使用空文件，避免导入完整 CHARM 管线：

```text
paper1/
├── __init__.py
├── graphs.py
├── data_generation/
└── efficiency_experiment/
```

运行依赖沿用 `data_generation/requirements.txt`。旧D3适配已自含，无需复制 `bench/`、`estimation.py` 或其他学图代码。独立测试时与完整原实现对比的可选 parity 测试会跳过；数值、配置、生成与运行器测试仍执行。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s paper1/efficiency_experiment/tests -v
```

测试涵盖旧D3逐边/同种子对照、旧乘积点估计一致性、非单位与反向暴露对比、联合 sandwich 与 delete-one jackknife、异方差重复抽样、bootstrap 重现、满秩检查、NPZ哈希/形状/列序、并发确定性、非法边界、缺失方案、插件契约、汇总公式与原终点适用范围。
