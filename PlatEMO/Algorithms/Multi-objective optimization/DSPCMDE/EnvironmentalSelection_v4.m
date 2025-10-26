function [Population, FrontNo, CrowdDis] = EnvironmentalSelection_v4(Population, N, a, strategy, progress)
% Environmental selection with Strategy v4 RL-based weight adjustment
%
% strategy: Strategyv4 object for RL-based weight selection
% progress: current progress (FE/maxFE)
% Note: The weight 'a' is already calculated by Strategy v4 in the main algorithm

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
    
    % Use RL-selected weight 'a' which was already calculated
    R_sum = (1 - a) * Rc + a * Rp;
    
    [~, Rank] = sort(R_sum);

    Population = Population(Rank(1:N));
    FrontNo = FrontNo1(Rank(1:N));
    CrowdDis = CrowdDis1(Rank(1:N));  
end
