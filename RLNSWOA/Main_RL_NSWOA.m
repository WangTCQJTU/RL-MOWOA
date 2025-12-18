function Main_RL_NSWOA()
%% RL-NSWOA 主脚本

clc;
clear;
close all;

fprintf('\n');
fprintf('========================================\n');
fprintf('  RL-NSWOA 强化学习优化算法\n');
fprintf('========================================\n');
fprintf('\n');

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
addpath(genpath(script_dir));
if isempty(which('RL_NSWOA'))
    error('未找到RL_NSWOA函数文件，请确认RL_NSWOA.m位于: %s', script_dir);
end

%% 运行
run_rsm_bt1_save_pareto_runs();

end

function run_rsm_bt1_save_pareto_runs()
params = struct();
params.D = 3;
params.M = 2;
params.LB = [0.3, 80, 0];
params.UB = [1, 400, 1];
params.Max_iteration = 1000;
params.SearchAgents_no = 100;
params.ishow = 10;
params.num_runs = 30;
params.Max_evals = 50000;

base_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(base_dir,'results','RSM');
if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

for run = 1:params.num_runs
    rng(33 + run - 1);
    chromosome = initialize_variables(params.SearchAgents_no, params.M, params.D, params.LB, params.UB);
    intermediate_chromosome = non_domination_sort_mod(chromosome, params.M, params.D);
    Population = replace_chromosome(intermediate_chromosome, params.M, params.D, params.SearchAgents_no);

    rng(33 + run - 1);
    chromosome = initialize_variables(params.SearchAgents_no, params.M, params.D, params.LB, params.UB);
    intermediate_chromosome = non_domination_sort_mod(chromosome, params.M, params.D);
    Population = replace_chromosome(intermediate_chromosome, params.M, params.D, params.SearchAgents_no);

    [rl_nswoa_result, ~] = RL_NSWOA(params.D, params.M, params.LB, params.UB, Population, params.SearchAgents_no, params.Max_iteration, params.ishow, params.Max_evals);
    rl_front = extract_pareto_front(rl_nswoa_result, params.M, params.D);
    out_file = fullfile(out_dir, sprintf('pareto_run_%02d.csv', run));
    writematrix(rl_front, out_file);
end

fprintf('\n已保存30次RSM(BT1)独立运行的Pareto前沿到目录: %s\n', out_dir);
end

function pareto_front = extract_pareto_front(population, M, D)
%% 提取Pareto前沿

K = M + D;
first_front_indices = find(population(:, K+1) == 1);
if isempty(first_front_indices)
    first_front_indices = 1:min(10, size(population, 1));
end

pareto_front = population(first_front_indices, D+1:D+M);

end
