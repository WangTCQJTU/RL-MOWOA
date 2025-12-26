%% Cited from NSGA-II All rights reserved.
function f = non_domination_sort_mod(x, M, D)

%% function f = non_domination_sort_mod(x, M, D)
% This function sort the current popultion based on non-domination. All the
% individuals in the first front are given a rank of 1, the second front
% individuals are assigned rank 2 and so on. After assigning the rank the
% crowding in each front is calculated.

[N, ~] = size(x);%获取x的行数和列数

% Initialize the front number to 1.
front = 1;

% There is nothing to this assignment, used only to manipulate easily in
% MATLAB.
F(front).f = [];%将空数组 [] 赋值给 F(front).f。在鲸鱼优化算法中，F是一个结构体数组，其中每个元素代表一个Pareto前沿（Pareto Front）中的解集。这里通过将空数组赋值给 F(front).f，实际上是初始化了一个空的前沿集合。
individual = [];%将空数组 [] 赋值给 individual。individual 是一个用于存储鲸鱼个体的变量。通过将空数组赋值给 individual，实际上是初始化了一个空的个体集合。

%% Non-Dominated sort. 
% The initialized population is sorted based on non-domination. 
for i = 1 : N
    % Number of individuals that dominate this individual
    individual(i).n = 0;%被支配的数量
    % Individuals which this individual dominate
    individual(i).p = [];%支配的个体
    for j = 1 : N   %对于其他个体，统计个体i和个体j在目标函数空间中的支配关系
        dom_less = 0;   %统计个体i在该目标函数上取值小于个体j的目标函数值的数量；
        dom_equal = 0;  %统计个体i在该目标函数上取值等于个体j的目标函数值的数量
        dom_more = 0;   %统计个体i在该目标函数上取值大于个体j的目标函数值的数量。
        for k = 1 : M   %对于每个目标函数
            if (x(i,D + k) < x(j,D + k))    %
                dom_less = dom_less + 1;
            elseif (x(i,D + k) == x(j,D + k))
                dom_equal = dom_equal + 1;
            else
                dom_more = dom_more + 1;
            end
        end
        if dom_less == 0 && dom_equal ~= M
            individual(i).n = individual(i).n + 1;  %如果dom_less等于0且dom_equal不等于M（M为目标函数的数量），则个体i被个体j所支配，因此个体i的被支配数量加1
        elseif dom_more == 0 && dom_equal ~= M
            individual(i).p = [individual(i).p j];  %如果dom_more等于0且dom_equal不等于M，则个体i支配个体j，因此将个体j添加到个体i的支配个体列表中。
        end
    end   % end for j (N)
    if individual(i).n == 0
        x(i,M + D + 1) = 1;
        F(front).f = [F(front).f i];    %如果个体i的被支配数量为0，则将其标记为第front个前沿，并将其索引i添加到相应的前沿中（F(front).f = [F(front).f i]）。
    end     %谁都支配不了i，那么i就是前沿 
end % end for i (N)
% Find the subsequent fronts
while ~isempty(F(front).f)
   Q = [];%初始化一个空数组 Q，用于存储下一个前沿的个体。
   for i = 1 : length(F(front).f)   %对当前前沿中的每一个个体进行遍历。
       if ~isempty(individual(F(front).f(i)).p)    %对于前沿i所支配的个体
        	for j = 1 : length(individual(F(front).f(i)).p)
            	individual(individual(F(front).f(i)).p(j)).n = ...
                	individual(individual(F(front).f(i)).p(j)).n - 1;%将当前个体的父代的后代计数 n 减去 1，表示该父代多了一个后代。
        	   	if individual(individual(F(front).f(i)).p(j)).n == 0%判断当前父代的后代计数是否为 0，即判断该父代是否成为下一个前沿的候选个体。
               		x(individual(F(front).f(i)).p(j),M + D + 1) = ...
                        front + 1;%如果当前父代成为下一个前沿的候选个体，则在一个记录矩阵 x 中的特定位置（通常用来记录个体所属的前沿）更新该父代的信息。
                    Q = [Q individual(F(front).f(i)).p(j)];%将当前父代加入到数组 Q 中，以便在下一轮迭代中形成新的前沿。
                end
            end
       end
   end
   front =  front + 1;
   F(front).f = Q;
end

sorted_based_on_front = sortrows(x,M+D+1); % sort follow front M+D+1这一列记录的是前沿层数。按照前沿层数对个体排序
current_index = 0;

%% Crowding distance
% Find the crowding distance for each individual in each front
for front = 1 : (length(F) - 1) %循环遍历所有前沿
    y = [];
    previous_index = current_index + 1; %记录当前前沿开始的索引位置。
    for i = 1 : length(F(front).f)
        y(i,:) = sorted_based_on_front(current_index + i,:);    %将当前前沿中的个体按照其在排序后的前沿集合中的顺序存储到数组 y 中
    end
    current_index = current_index + i;  %i表示某一个前沿的个数，current_index 是上一个前沿的最后一个个体的索引。            
    for i = 1 : M  %对于每个目标函数，计算拥挤度。
        [sorted_based_on_objective, index_of_objectives] = sortrows(y,D + i);%将矩阵 y 按照当前目标函数的值进行排序，并记录排序后的索引     
        f_max = ...
            sorted_based_on_objective(length(index_of_objectives), D + i);
        f_min = sorted_based_on_objective(1, D + i);    %计算当前目标函数的最大值 f_max 和最小值 f_min。
        y(index_of_objectives(length(index_of_objectives)),M + D + 1 + i)...
            = Inf;
        y(index_of_objectives(1),M + D + 1 + i) = Inf;  %根据拥挤度定义，将边界个体的拥挤度设置为无穷大。
         for j = 2 : length(index_of_objectives) - 1    %对于中间个体，计算其拥挤度值，并存储到矩阵 y 中。
            next_obj  = sorted_based_on_objective(j + 1,D + i); %获取排序后目标值中，当前个体的下一个个体在第 D + i 个目标函数上的值
            previous_obj  = sorted_based_on_objective(j - 1,D + i);%获取排序后目标值中，当前个体的上一个个体在第 D + i 个目标函数上的值
            if (f_max - f_min == 0)%判断最大值和最小值之差是否为零，即判断目标值范围是否为零。
                y(index_of_objectives(j),M + D + 1 + i) = Inf;%如果目标值范围为零，表示所有个体在该目标上的值都相等，则将当前个体的拥挤距离设为无穷大，以避免除以零的情况。
            else
                y(index_of_objectives(j),M + D + 1 + i) = ...
                     (next_obj - previous_obj)/(f_max - f_min);%归一化处理，统一量纲 都在0到1之间。
            end
         end % end for j
    end % end for i
    distance = [];
    distance(:,1) = zeros(length(F(front).f),1);
    for i = 1 : M
        distance(:,1) = distance(:,1) + y(:,M + D + 1 + i); %计算拥挤度的总和，即个体在多个目标函数上的拥挤度之和。
    end
    y(:,M + D + 2) = distance;
    y = y(:,1 : M + D + 2); %将计算出的拥挤度存储在矩阵 y 的最后一列中。
    z(previous_index:current_index,:) = y;  %将拥挤度信息更新到种群矩阵 z 的对应位置。
end
f = z;