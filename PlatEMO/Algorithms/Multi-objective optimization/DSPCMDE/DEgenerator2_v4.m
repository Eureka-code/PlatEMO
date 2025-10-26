function [Offspring] = DEgenerator2_v4(Problem, Population, strategy)
% DE operator with Strategy v4 RL-based operator selection
%
% strategy: Strategyv4 object for RL-based action selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    cv = sum(max(0, Population.cons), 2);       

    FrontNo = NDSort(Population.objs, Population.cons, 1);   
    index1  = find(FrontNo == 1);
    r       = floor(rand * length(index1)) + 1;
    best    = index1(r);

    [N, D] = size(Population(1).decs);       
    trial = zeros(1 * Problem.N, D);
    
    % Get current state from strategy
    progress = Problem.FE / Problem.maxFE;
    state = strategy.GetState(Population, Problem);
    
    % Select actions
    [action_w, action_op] = strategy.SelectActions(state, progress);
    
    % Get operator parameters from action
    [F, CR] = strategy.GetOperatorParams(action_op);
    
    % Determine if using diversity or convergence operator
    use_diversity = (action_op <= 3);
    
    for i = 1 : Problem.N   
        if use_diversity
            % DE/current-to-rand/1--Diversity (actions 1-3)
            indexset = 1:Problem.N;
            indexset(i) = [];
            r1 = floor(rand * (Problem.N - 1)) + 1;
            xr1 = indexset(r1);
            indexset(r1) = [];
            r2 = floor(rand * (Problem.N - 2)) + 1;
            xr2 = indexset(r2);
            indexset(r2) = [];
            r3 = floor(rand * (Problem.N - 3)) + 1;
            xr3 = indexset(r3);
            
            v = Population(i).decs + rand * (Population(xr1).decs - Population(i).decs) + ...
                F * (Population(xr2).decs - Population(xr3).decs);
            
            Lower = repmat(Problem.lower, N, 1);
            Upper = repmat(Problem.upper, N, 1); 
            trial(i, :) = min(max(v, Lower), Upper);
        else
            % DE/rand-to-best/1/bin--Convergence (actions 4-12)
            indexset = 1 : Problem.N;
            indexset(i) = [];
            r1 = floor(rand * (Problem.N - 1)) + 1;
            xr1 = indexset(r1);
            indexset(r1) = [];
            r2 = floor(rand * (Problem.N - 2)) + 1;
            xr2 = indexset(r2)  ;
            r3 = floor(rand * (Problem.N - 3)) + 1;
            xr3 = indexset(r3);
            
            Best_index = Population(best).decs;
            v = Population(xr1).decs + rand * (Best_index - Population(xr1).decs) + ...
                F * (Population(xr2).decs - Population(xr3).decs);
            
            Lower = repmat(Problem.lower, N, 1);
            Upper = repmat(Problem.upper, N, 1);
            v = min(max(v, Lower), Upper);
            
            % Binomial crossover
            Site = rand(N, D) < CR;
            j_rand = floor(rand * D) + 1;
            Site(1, j_rand) = 1;
            Site_ = 1 - Site;
            trial(i, :) = Site .* v + Site_ .* Population(i).decs;
        end
    end
    
    Offspring = trial;
    Offspring = Problem.Evaluation(Offspring);
end
