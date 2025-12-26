classdef QLearningAgent < handle
    %% Q-Learning智能体类
    % 用于自适应调整NSWOA算法参数
    % 学习最优的参数组合以提高算法性能
    
    properties
        q_table          % Q表
        learning_rate    % 学习率
        discount_factor  % 折扣因子
        epsilon          % 探索率
        epsilon_decay    % 探索率衰减
        epsilon_min      % 最小探索率
        state_size       % 状态空间大小
        action_size      % 动作空间大小
        actions          % 动作集合（参数组合）
        current_episode  % 当前回合数
    end
    
    methods
        function obj = QLearningAgent()
            %% 构造函数 - 初始化Q-learning智能体
            
            % 超参数设置
            obj.learning_rate = 0.12;
            obj.discount_factor = 0.95;
            obj.epsilon = 0.35;
            obj.epsilon_decay = 0.9975;
            obj.epsilon_min = 0.05;
            obj.current_episode = 0;
            
            % 定义状态空间 (离散化)
            % 状态包括: [收敛程度, 多样性, 迭代进度, 种群质量]
            obj.state_size = 4^4; % 每个状态维度4个离散值
            
            % 定义动作空间 (参数组合)
            % 动作包括: [缩放因子SF, 螺旋参数b, 探索概率p, 变异率]
            obj.actions = obj.define_action_space();
            obj.action_size = size(obj.actions, 1);
            
            % 初始化Q表
            obj.q_table = zeros(obj.state_size, obj.action_size);
            
            fprintf('Q-Learning智能体初始化完成\n');
            fprintf('状态空间大小: %d, 动作空间大小: %d\n', obj.state_size, obj.action_size);
        end
        
        function actions = define_action_space(obj)
            %% 定义动作空间 - 不同的参数组合
            % 每个动作是一个参数向量: [SF, b, p, mutation_rate]
            
            SF_values = [1.15, 1.25, 1.35, 1.45];
            b_values = [1.2, 1.4, 1.6, 1.8];
            p_values = [0.60, 0.65, 0.70, 0.75];
            mutation_values = [0.08, 0.10, 0.12, 0.14];
            
            % 生成所有参数组合
            [SF_grid, b_grid, p_grid, mut_grid] = ndgrid(SF_values, b_values, p_values, mutation_values);
            
            actions = [SF_grid(:), b_grid(:), p_grid(:), mut_grid(:)];
        end
        
        function state_index = discretize_state(obj, state_vector)
            %% 将连续状态向量离散化为状态索引
            % state_vector: [收敛程度, 多样性, 迭代进度, 种群质量]
            
            % 将每个状态维度离散化为4个区间
            discrete_state = zeros(1, 4);
            
            for i = 1:4
                if state_vector(i) <= 0.25
                    discrete_state(i) = 1;
                elseif state_vector(i) <= 0.5
                    discrete_state(i) = 2;
                elseif state_vector(i) <= 0.75
                    discrete_state(i) = 3;
                else
                    discrete_state(i) = 4;
                end
            end
            
            % 将多维离散状态转换为一维索引
            state_index = sub2ind([4, 4, 4, 4], discrete_state(1), discrete_state(2), ...
                                 discrete_state(3), discrete_state(4));
        end
        
        function [action_index, params] = select_action(obj, state_vector)
            %% 选择动作 - 使用ε-贪婪策略
            
            state_index = obj.discretize_state(state_vector);
            
            if rand() < obj.epsilon
                % 探索：随机选择动作
                action_index = randi(obj.action_size);
            else
                % 利用：选择Q值最大的动作
                [~, action_index] = max(obj.q_table(state_index, :));
            end
            
            % 返回对应的参数组合
            params = obj.actions(action_index, :);
        end
        
        function update_q_table(obj, state_vector, action_index, reward, next_state_vector)
            %% 更新Q表 - 使用Q-learning更新规则
            
            state_index = obj.discretize_state(state_vector);
            next_state_index = obj.discretize_state(next_state_vector);
            
            % Q-learning更新公式
            current_q = obj.q_table(state_index, action_index);
            max_next_q = max(obj.q_table(next_state_index, :));
            
            new_q = current_q + obj.learning_rate * (reward + obj.discount_factor * max_next_q - current_q);
            obj.q_table(state_index, action_index) = new_q;
            
            % 更新探索率
            if obj.epsilon > obj.epsilon_min
                obj.epsilon = obj.epsilon * obj.epsilon_decay;
            end
            
            obj.current_episode = obj.current_episode + 1;
        end
        
        function save_agent(obj, filename)
            %% 保存训练好的智能体
            save(filename, 'obj');
            fprintf('Q-Learning智能体已保存到: %s\n', filename);
        end
        
        function load_agent(obj, filename)
            %% 加载预训练的智能体
            if exist(filename, 'file')
                loaded_data = load(filename);
                obj.q_table = loaded_data.obj.q_table;
                obj.epsilon = loaded_data.obj.epsilon;
                obj.current_episode = loaded_data.obj.current_episode;
                fprintf('Q-Learning智能体已从文件加载: %s\n', filename);
            else
                fprintf('警告: 文件 %s 不存在，使用默认初始化\n', filename);
            end
        end
        
        function visualize_q_table(obj)
            %% 可视化Q表 - 显示学习到的策略
            
            figure('Name', 'Q-Table可视化', 'Position', [100, 100, 400, 300]);
            
            % 显示每个状态的最优动作
            [~, optimal_actions] = max(obj.q_table, [], 2);
            plot(optimal_actions, 'o-', 'LineWidth', 2);
            title('Optimal Action for Each State');
            xlabel('State Index');
            ylabel('Optimal Action Index');
            grid on;
        end
        
        function stats = get_learning_stats(obj)
            %% 获取学习统计信息
            
            stats = struct();
            stats.total_episodes = obj.current_episode;
            stats.current_epsilon = obj.epsilon;
            stats.q_table_mean = mean(obj.q_table(:));
            stats.q_table_std = std(obj.q_table(:));
            stats.q_table_max = max(obj.q_table(:));
            stats.q_table_min = min(obj.q_table(:));
            
            % 计算每个动作被选择为最优的次数
            [~, optimal_actions] = max(obj.q_table, [], 2);
            stats.action_preferences = histcounts(optimal_actions, 1:obj.action_size+1);
            
            fprintf('=== Q-Learning智能体学习统计 ===\n');
            fprintf('总回合数: %d\n', stats.total_episodes);
            fprintf('当前探索率: %.4f\n', stats.current_epsilon);
            fprintf('Q表均值: %.4f\n', stats.q_table_mean);
            fprintf('Q表标准差: %.4f\n', stats.q_table_std);
            fprintf('Q表最大值: %.4f\n', stats.q_table_max);
            fprintf('Q表最小值: %.4f\n', stats.q_table_min);
        end
    end
end
