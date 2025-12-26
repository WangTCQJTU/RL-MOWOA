classdef RL_Utils < handle
%% RL_Utils - 强化学习辅助函数集合
% 包含状态获取、性能指标计算、奖励计算等功能

methods (Static)
        
function state = get_algorithm_state(Whale_pos, iteration, max_iteration, M, D)
%% 获取算法当前状态
% 返回标准化的状态向量: [收敛程度, 多样性, 迭代进度, 种群质量]

% 获取第一个非支配前沿
K = M + D;
first_front_indices = find(Whale_pos(:, K+1) == 1);
if isempty(first_front_indices)
    first_front_indices = 1:size(Whale_pos, 1);
end

if ~isempty(first_front_indices)
    first_front = Whale_pos(first_front_indices, D+1:D+M);
else
    first_front = [];
end

% 1. 收敛程度 - 基于目标函数值的标准差
if ~isempty(first_front) && size(first_front, 1) > 1
    convergence = 1 / (1 + mean(std(first_front, 0, 1)));
else
    convergence = 0.5;
end

% 2. 多样性 - 基于解的分布
if ~isempty(first_front)
    diversity = RL_Utils.calculate_diversity_metric(first_front);
else
    diversity = 0;
end

% 3. 迭代进度
progress = iteration / max_iteration;

% 4. 种群质量 - 基于非支配解的比例
quality = length(first_front_indices) / size(Whale_pos, 1);

% 组合状态向量
state = [convergence, diversity, progress, quality];

% 确保状态在[0,1]范围内
state = max(0, min(1, state));

end
        
function diversity = calculate_diversity_metric(front)
%% 计算多样性指标

if isempty(front) || size(front, 1) <= 1
    diversity = 0;
    return;
end

% 计算目标空间的跨度，进一步增加多样性权重
if size(front, 1) > 1
    ranges = max(front) - min(front);
    % 极大增加多样性计算的权重因子
     diversity = sum(ranges) * 2.5;  % 从2.0进一步增加到2.5倍权重
     
     % 增强额外的多样性奖励机制
     if size(front, 1) > 2
         % 计算解之间的平均距离作为额外多样性指标
         distances = pdist(front);
         avg_distance = mean(distances);
         diversity = diversity + avg_distance * 0.8;  % 从0.5增加到0.8
         
         % 添加基于方差的多样性奖励
         obj_variance = sum(var(front));
         diversity = diversity + obj_variance * 0.3;
     end
else
    diversity = 0;
end

end
        
function metrics = calculate_performance_metrics(Whale_pos, M, D)
%% 计算性能指标

K = M + D;
first_front_indices = find(Whale_pos(:, K+1) == 1);
if isempty(first_front_indices)
    first_front_indices = 1:min(10, size(Whale_pos, 1));
end

first_front = Whale_pos(first_front_indices, D+1:D+M);

metrics = struct();

% 1. 超体积指标（归一化）
metrics.hypervolume = hv2d_norm(first_front);

% 2. 间距指标
metrics.spacing = RL_Utils.calculate_spacing_metric(first_front);

% 3. 收敛性指标
metrics.convergence = RL_Utils.calculate_convergence_metric(first_front);

% 4. 多样性指标
metrics.diversity = RL_Utils.calculate_diversity_metric(first_front);
% IGD（若已有经验真实PF）或GD到理想点
try
    pf_path = 'D:\carstructure\WT\小论文\NSWOA_3\ruslt_parot\RSM\data\empirical_true_pf_thin.csv';
    if exist(pf_path,'file')
        R = readmatrix(pf_path);
        metrics.igd = metrics_utils.metric_igd(first_front, R);
    else
        metrics.igd = RL_Utils.calculate_gd_to_ideal(first_front);
    end
catch
    metrics.igd = RL_Utils.calculate_gd_to_ideal(first_front);
end
metrics.spread = metrics_utils.metric_spread(first_front, []);

end
        
function hv = calculate_hypervolume_simple(front)
%% 简化的超体积计算

% 输入验证
if isempty(front) || size(front, 1) == 0 || size(front, 2) < 2
    hv = 0;
    return;
end

% 使用简化的超体积计算方法
% 对于2目标问题，使用梯形积分
if size(front, 2) == 2
    % 排序
    [sorted_front, ~] = sortrows(front(:,1:2), 1);
    max1 = max(sorted_front(:,1));
    min1 = min(sorted_front(:,1));
    max2 = max(sorted_front(:,2));
    min2 = min(sorted_front(:,2));
    range1 = max1 - min1; if range1 == 0, range1 = 1; end
    range2 = max2 - min2; if range2 == 0, range2 = 1; end
    if all(sorted_front(:,1) >= 0) && all(sorted_front(:,1) <= 1.05) && all(sorted_front(:,2) >= 0) && all(sorted_front(:,2) <= 1.05)
        ref_point = [1.1, 1.1];
    else
        ref_point = [max1 + 0.1 * range1, max2 + 0.1 * range2];
    end
    hv = 0;
    n = size(sorted_front, 1);
    for i = 1:n
        if i == n
            width = ref_point(1) - sorted_front(i, 1);
        else
            width = sorted_front(i+1, 1) - sorted_front(i, 1);
        end
        height = ref_point(2) - sorted_front(i, 2);
        hv = hv + max(0, width) * max(0, height);
    end
else
    % 对于多目标，使用简化方法
    if size(front, 1) == 1
        ref_point = front + 1;  % 单个点时的简单处理
    else
        range_vals = max(front) - min(front);
        range_vals(range_vals == 0) = 1;  % 避免除零
        ref_point = max(front) + 0.1 * range_vals;
    end
    hv = 0;
    for i = 1:size(front, 1)
        volume = prod(max(0, ref_point - front(i, :)));
        hv = hv + volume;
    end
    hv = hv / max(1, size(front, 1));
end

% 确保非负
hv = max(0, hv);

end
        
function spacing = calculate_spacing_metric(front)
%% 计算间距指标

if isempty(front) || size(front, 1) <= 1
    spacing = 0;
    return;
end

% 计算每个解到其最近邻的距离
distances = [];
for i = 1:size(front, 1)
    min_dist = inf;
    for j = 1:size(front, 1)
        if i ~= j
            dist = norm(front(i, :) - front(j, :));
            min_dist = min(min_dist, dist);
        end
    end
    distances = [distances; min_dist];
end

% 间距指标为距离的标准差
spacing = std(distances);

end
        
function convergence = calculate_convergence_metric(front)
%% 计算收敛性指标

if isempty(front)
    convergence = inf;
    return;
end

% 简化的收敛性指标：到原点的平均距离
convergence = mean(sqrt(sum(front.^2, 2)));

% 标准化
convergence = 1 / (1 + convergence);

end

function gd = calculate_gd_to_ideal(front)
%% 计算到理想点的生成距离（归一化）

if isempty(front)
    gd = inf;
    return;
end
ideal = min(front, [], 1);
ranges = max(front, [], 1) - min(front, [], 1);
ranges(ranges == 0) = 1;
diffs = (front - ideal) ./ ranges;
d = sqrt(sum(diffs.^2, 2));
gd = mean(d);
end
        
        
function reward = calculate_rl_reward(current_metrics, iteration, max_iteration)
%% 计算强化学习奖励

% 奖励权重（HV优先，IGD次优，Spacing与Spread辅助，多样性轻权）
    w1_base = 0.80; w6_base = 0.42; w7_base = 0.26; w2 = 0.26; w3 = 0.03; w4_base = 0.02; w5 = 0.06;
prog = iteration / max_iteration;
    w1 = w1_base + 0.4 * prog;
    w6 = w6_base + 0.2 * prog;
    w7 = w7_base + 0.15 * prog;
    w4 = w4_base * (1 - prog) + 0.03 * prog;

% 标准化指标到[0,1]范围
    hv_reward = min(1, current_metrics.hypervolume);
    hv_reward = hv_reward.^1.22;
    sp_reward = 1 / (1 + current_metrics.spacing);
conv_reward = current_metrics.convergence;
% 进一步增强多样性奖励的敏感度
    div_reward = min(1, current_metrics.diversity / 120);
    spread_reward = 1 / (1 + current_metrics.spread);
igd_reward = 1 / (1 + current_metrics.igd);

% 进度奖励
progress_reward = iteration / max_iteration;

% 综合奖励
    reward = w1 * hv_reward + w2 * sp_reward + w3 * conv_reward + w4 * div_reward + w5 * progress_reward + w6 * igd_reward + w7 * spread_reward;

% 确保奖励在合理范围内
reward = max(-1, min(2, reward));

end
        
end

end
