function f  = replace_chromosome(intermediate_chromosome, M,D,NP)

%% function f  = replace_chromosome(intermediate_chromosome,M,D,NP)
% This function replaces the chromosomes based on rank and crowding
% distance. Initially until the population size is reached each front is
% added one by one until addition of a complete front which results in
% exceeding the population size. At this point the chromosomes in that
% front is added subsequently to the population based on crowding distance.
%该函数根据等级和拥挤距离来替换染色体。最初，直到达到种群规模，每个前沿被一个接一个地添加，
%直到添加一个完整的前沿，导致超过种群规模。在这一点上，前面的染色体随后根据拥挤距离添加到种群中。
[~,m]=size(intermediate_chromosome);
f=zeros(NP,m);%创建一个大小为 NP×m 的空矩阵 f，用于保存新的染色体。

% Now sort the individuals based on the index
sorted_chromosome = sortrows(intermediate_chromosome,M + D + 1);%将 intermediate_chromosome根据第M+D+1列（即等级）进行排序，得到sorted_chromosome。

% Find the maximum rank in the current population
max_rank = max(intermediate_chromosome(:,M + D + 1));%找到当前种群中等级的最大值，存储在 max_rank 中。

% Start adding each front based on rank and crowing distance until the
% whole population is filled.开始逐个添加每个等级的前沿，直到填满整个种群。
previous_index = 0;%使用变量 previous_index 来记录上一个等级的最后一个染色体的索引，初始值为 0。
for i = 1 : max_rank %优先要等级低的，也就是被支配少的。
    % Get the index for current rank i.e the last the last element in the
    % sorted_chromosome with rank i. 对于每个等级 i，在 sorted_chromosome 中找到等级为 i 的最后一个染色体的索引 current_index。
    current_index = find(sorted_chromosome(:,M + D + 1) == i, 1, 'last' );
    % Check to see if the population is filled if all the individuals with
    % rank i is added to the population. 
    if current_index > NP
        % If so then find the number of individuals with in with current
        % rank i.如果 current_index 大于 NP（种群大小），说明将添加等级为 i 的染色体会超过种群容量，此时需要进行处理。
        remaining = NP - previous_index;%计算剩余可添加的个体数量 remaining = NP - previous_index。
        % Get information about the individuals in the current rank i.
        temp_pop = ...
            sorted_chromosome(previous_index + 1 : current_index, :);%获取等级为 i 的染色体信息，存储在 temp_pop 中。
        % Sort the individuals with rank i in the descending order based on
        % the crowding distance.
        [~,temp_sort_index] = ...
            sort(temp_pop(:, M + D + 2),'descend');%根据拥挤距离对等级为 i 的染色体进行降序排序，得到 temp_sort_index。
        % Start filling individuals into the population in descending order
        % until the population is filled.
        for j = 1 : remaining
            f(previous_index + j,:) = temp_pop(temp_sort_index(j),:);
        end %从降序排列的染色体中依次选取个体，填充到种群中，直到种群达到容量 NP。
        return;
    elseif current_index < NP %如果 current_index 小于 NP，则将等级为 i 的所有染色体全部添加到种群中。
        % Add all the individuals with rank i into the population.
        f(previous_index + 1 : current_index, :) = ...
            sorted_chromosome(previous_index + 1 : current_index, :);
    else
        % Add all the individuals with rank i into the population.
        f(previous_index + 1 : current_index, :) = ...
            sorted_chromosome(previous_index + 1 : current_index, :);
        return;%如果 current_index 等于 NP，意味着刚好填满种群，将等级为 i 的所有染色体添加到种群中，并结束循环。
    end % end if current_index
    % Get the index for the last added individual.
    previous_index = current_index;%更新 previous_index 为 current_index，进入下一个等级的处理。
end
%选择个体，对每个个体按照等级排序，每个等级内部再按照拥挤度距离排序，优先选择等级低并且拥挤度距离大的个体存入pop中，依次添加各个等级
