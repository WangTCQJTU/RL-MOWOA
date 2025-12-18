%% Cited from NSGA-II All rights reserved.
function f = initialize_variables(NP, M, D, LB, UB)

%% function f = initialize_variables(N, M, D, LB, UB) 
% This function initializes the population. Each individual has the
% following at this stage
%       * set of decision variables 决策变量集
%       * objective function values 目标函数值
% 
% where,
% NP - Population size
% M - Number of objective functions
% D - Number of decision variables
% min_range - A vector of decimal values which indicate the minimum value
% for each decision variable.
% max_range - Vector of maximum possible values for decision variables.

min = LB;
max = UB;
K = M + D;
f=zeros(NP,K);

%% Initialize each individual in population
% For each chromosome perform the following (N is the population size)
% 使用分层抽样（类LHS）提升初始覆盖度
bins = (0:NP-1)'/NP;
X = zeros(NP, D);
for j = 1:D
    u = rand(NP,1)/NP;
    samples01 = bins + u;
    perm = randperm(NP);
    samples01 = samples01(perm);
    X(:,j) = min(j) + samples01*(max(j)-min(j));
end

for i = 1:NP
    f(i,1:D) = X(i,1:D);
    f(i,D + 1: K) = evaluate_objective(f(i,1:D));
end
