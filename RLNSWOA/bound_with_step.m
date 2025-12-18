function x = bound_with_step(x, UB, LB, step_sizes)
%% 带步长约束的边界处理函数
% 输入:
%   x - 决策变量向量
%   UB - 上界向量
%   LB - 下界向量  
%   step_sizes - 步长向量 [壁厚步长, 密度步长, 比例步长]
% 输出:
%   x - 处理后的决策变量向量

% 默认步长设置
if nargin < 4
    step_sizes = [0.01, 1, 0.01];  % [壁厚, 密度, 比例]
end

% 确保输入为行向量
if size(x, 1) > size(x, 2)
    x = x';
end

% 对每个变量进行边界和步长约束
for i = 1:length(x)
    % 边界反射处理（先反射，再量化）
    if x(i) < LB(i)
        x(i) = LB(i) + (LB(i) - x(i));
    elseif x(i) > UB(i)
        x(i) = UB(i) - (x(i) - UB(i));
    end

    % 步长量化
    if step_sizes(i) > 0
        offset = x(i) - LB(i);
        adjusted_offset = round(offset / step_sizes(i)) * step_sizes(i);
        x(i) = LB(i) + adjusted_offset;
    end

    % 最终夹取，防止反射和量化后越界
    if x(i) > UB(i)
        x(i) = UB(i);
    elseif x(i) < LB(i)
        x(i) = LB(i);
    end
end

end
