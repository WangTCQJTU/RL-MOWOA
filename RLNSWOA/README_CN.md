# RL-NSWOA 核心代码（MATLAB）

本仓库保留强化学习增强的非排序鲸鱼优化算法（RL-NSWOA，论文中称 RLMOWOA）的最小可运行核心代码，去除了对比算法、可视化与后处理脚本以及所有结果数据，便于期刊开源与复现。

## Quick Start

- 运行环境：MATLAB R2021a+，无需第三方工具箱
- 将本文件夹作为工作目录，执行入口脚本

### 运行 RL-NSWOA（RLMOWOA）

- 入口脚本：`Main_RL_NSWOA.m`
- 使用方法：
  - 在 MATLAB 命令行执行：`Main_RL_NSWOA`
  - 默认配置运行 RSM 问题，独立运行 30 次，并将每次的 Pareto 前沿保存为 CSV 文件
- 输出目录：
  - `results/RSM/pareto_run_XX.csv`
- 可复现实验：
  - 随机种子按运行序号固定（默认 `rng(33 + run - 1)`）
  - 如需修改问题规模与参数，在 `run_rsm_bt1_save_pareto_runs()` 中调整：
    - `D`（决策变量维数）、`M`（目标数）、`LB/UB`（上下界）
    - `SearchAgents_no`（种群大小）、`Max_iteration`（迭代次数）
    - `num_runs`（独立运行次数）、`Max_evals`（最大评价次数）

## 文件说明（保留的最小集合）

- `Main_RL_NSWOA.m`：入口脚本（默认批量运行并保存 CSV）
- `RL_NSWOA.m`：RL 增强的 NSWOA 核心实现
- `QLearningAgent.m`：Q-learning 智能体（选择参数组合）
- `RL_Utils.m`：RL 辅助（状态/指标/奖励）
- `initialize_variables.m`、`non_domination_sort_mod.m`、`replace_chromosome.m`、`replace_chromosome_uniform.m`：初始化、非支配排序与选择
- `evaluate_objective.m`、`bound_with_step.m`：目标函数与边界/步长处理
- `hv2d.m`、`hv2d_norm.m`、`hv_contrib_2d.m`、`metrics_utils.m`：HV/IGD/Spread 等必要度量


## 参考真实帕累托（测试用）

- 已提供四个标准测试函数的参考真实帕累托，用于离线指标评估：
  - 路径：`results/reference_pareto/`
  - 文件：`zdt1_true_pf.csv`、`zdt2_true_pf.csv`、`zdt3_true_pf.csv`、`dtlz7_true_pf.csv`
- 用途：
  - 可用于计算 IGD/Spread 等对比指标的离线评估
  - 若需在代码中自动使用，请在`RL_Utils.m:109-118`将`pf_path`指向相应的 CSV（或根据问题类型选择对应文件）

## 注意事项

- 本仓库不包含可视化、对比算法与后处理脚本，也不包含任何历史结果数据
- 仅保留可复现 RLMOWOA 的核心代码与入口，输出为 CSV，便于论文复现与评审

## 复现建议

- 固定随机种子以获得稳定的对比
- 调整 `LB/UB` 与 `Max_evals` 可控制搜索范围与评价成本
- 如需将输出改到其他相对路径，可修改 `Main_RL_NSWOA.m` 中的 `out_dir`

## How to Cite

如在学术工作中使用本代码，请引用对应论文。可添加 `CITATION.cff` 或在此处提供 BibTeX。
