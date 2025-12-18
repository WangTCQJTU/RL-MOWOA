# RL‑NSWOA Core (MATLAB)

Reinforcement-Learning enhanced Non-dominated Sorting Whale Optimization Algorithm (RL-NSWOA),referred to as RL-MOWOA in the paper.This repository provides the minimal, runnable core of the Reinforcement‑Learning enhanced Non‑Sorted Whale Optimization Algorithm (RL‑NSWOA; referred to as RLMOWOA in the paper). It focuses on the essential components required to reproduce results, and excludes baselines, visualization utilities and historical outputs for clean open‑sourcing.

## Requirements
- MATLAB R2021a or newer
- No third‑party toolboxes required

## Quick Start
- Set this folder as the MATLAB working directory
- Run the entry script:
  - `Main_RL_NSWOA` (Command Window)
- Default behavior:
  - Runs the RSM problem for 30 independent trials
  - Optionally saves each trial’s Pareto front to CSV at results/RSM/pareto_run_XX.csv when enabled.
- To change problem size or runtime:
  - Edit function `run_rsm_bt1_save_pareto_runs()` inside `Main_RL_NSWOA.m`
  - Parameters to adjust: `D, M, LB, UB, SearchAgents_no, Max_iteration, num_runs, Max_evals`

## What’s Included
- `Main_RL_NSWOA.m`: entry and batch run with CSV export
- `RL_NSWOA.m`: core optimizer integrating RL with NSWOA
- `QLearningAgent.m`: ε-greedy Q-learning over a discretized multi-parameter action space
- `RL_Utils.m`: state construction, metrics and reward aggregation
- `initialize_variables.m`, `non_domination_sort_mod.m`, `replace_chromosome.m`, `replace_chromosome_uniform.m`: initialization, ranking and selection
- `evaluate_objective.m`, `bound_with_step.m`: RSM objectives and fixed‑step boundary handling
- `hv2d.m`, `hv2d_norm.m`, `hv_contrib_2d.m`, `metrics_utils.m`: HV/IGD/Spread metrics


## Highlights

- Mixed policy+parameter control via RL:
  - Action vector `[SF, b, p, mutation_rate]` with ε‑greedy selection
  - Progress‑adjusted `p_eff` switches between encircling and spiral updates

## Configuration Reference

- Key settings:
  - Seeds per run: `rng(33 + run - 1)`
  - Termination by evaluations: `Max_evals` (progress and WOA coefficients use `Max_iteration` for scheduling)
  - RL: `learning_rate=0.12`, `discount_factor=0.95`, `epsilon=0.35→0.05` with `epsilon_decay=0.9975`
  - Action space: `SF=[1.15,1.25,1.35,1.45]`, `b=[1.2,1.4,1.6,1.8]`, `p=[0.60,0.65,0.70,0.75]`, `mutation=[0.08,0.10,0.12,0.14]`
  - Reward terms: Reward is constructed from multiple convergence and diversity indicators,with adaptive weighting guided by search progress (see paper for details).
  - Reference PF files: `results/reference_pareto/` (`zdt1_true_pf.csv`, `zdt2_true_pf.csv`, `zdt3_true_pf.csv`, `dtlz7_true_pf.csv`

## Outputs
- CSV per run: `results/RSM/pareto_run_XX.csv`
- Each file contains objective rows of the first Pareto front extracted from the final population

## Reference True Pareto Fronts (Tests)
- Provided for four standard test functions, to support offline metric evaluation:
  - Location: `results/reference_pareto/`
  - Files: `zdt1_true_pf.csv`, `zdt2_true_pf.csv`, `zdt3_true_pf.csv`, `dtlz7_true_pf.csv`, `rsm_true_pf.csv`
- Usage:
  - Use for IGD/Spread comparisons in external analysis
  - To enable automatic IGD in code, set `pf_path` in `RL_Utils.m:109-118` to the desired CSV (or select based on the problem type)

## Reproducibility Tips
- Keep seeds fixed for fair comparison across runs
- Tune `LB/UB` and `Max_evals` to control search range and budget
- Adjust `SearchAgents_no` and `num_runs` based on desired statistical confidence

## Algorithm Outline (brief)
- Initialize population (stratified sampling), rank by non‑domination and crowding
- Per iteration:
  - Build 4‑dim state `[convergence, diversity, progress, quality]` and pick an action by ε‑greedy Q‑learning
  - Pareto‑aware leader sampling for direction
  - Encircling or spiral update (chosen by progress‑adjusted `p_eff`), then fixed‑step bounding
  - Adaptive polynomial mutation and archive update
  - Compute metrics and reward; update Q‑table
  - Structured uniformization and elite injection; record convergence curve
- Stop when evaluation budget is exhausted; extract first front and save

## Citation
If you use this code in academic work, please cite the corresponding paper. Consider adding a `CITATION.cff` or provide a BibTeX entry.
