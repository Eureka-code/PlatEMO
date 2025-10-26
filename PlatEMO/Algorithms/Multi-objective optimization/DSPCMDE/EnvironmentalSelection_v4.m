function [Population, FrontNo, CrowdDis] = EnvironmentalSelection_v4(Population, N, a, strategy, progress)
% Environmental selection with Strategy v4 RL-based weight adjustment
%
% strategy: Strategyv4 object for RL-based weight selection
% progress: current progress (FE/maxFE)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
     
    [N1, ~] = size(Population.objs);   
    [FrontNo1, ~] = NDSort(Population.objs, Population.cons, inf);
    
    CrowdDis1 = CrowdingDistance(Population.objs, FrontNo1);
  
    [~, r1] = sortrows([FrontNo1', -CrowdDis1']);
    Rc(r1) = 1 : N1;

    [FrontNo2, ~] = NDSort(Population.objs, 0, inf);
    
    CrowdDis2 = CrowdingDistance(Population.objs, FrontNo2);
    
    [~, r2] = sortrows([FrontNo2', -CrowdDis2']);
    Rp(r2) = 1 : N1;
    
    % Use RL-selected weight instead of fixed formula
    % a is passed in but can be overridden by strategy
    if ~isempty(strategy)
        % Get current state
        state = strategy.GetState(Population(1:min(N, N1)), struct('FE', 0, 'maxFE', 1));
        
        % Get weight action from previous step or select new one
        if isempty(strategy.prev_action_w)
            [action_w, ~] = strategy.SelectActions(state, progress);
        else
            action_w = strategy.prev_action_w;
        end
        
        % Apply weight action to get new weight
        a = strategy.ApplyWeightAction(action_w, progress);
    end
    
    R_sum = (1 - a) * Rc + a * Rp;
    
    [~, Rank] = sort(R_sum);

    Population = Population(Rank(1:N));
    FrontNo = FrontNo1(Rank(1:N));
    CrowdDis = CrowdDis1(Rank(1:N));  
end
