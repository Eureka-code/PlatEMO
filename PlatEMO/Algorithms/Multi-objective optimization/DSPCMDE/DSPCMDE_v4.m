classdef DSPCMDE_v4 < ALGORITHM
% <2022> <multi> <real/integer> <constrained>
% Dynamic selection preference-assisted constrained multiobjective 
% differential evolution with Strategy v4
%
% This version implements Strategy v4 with:
% - Dual-head RL architecture for operator and weight selection
% - Phase-based reward functions
% - Experience replay
% - IGD-based callback modifier

%------------------------------- Reference --------------------------------
% K. Yu, J. Liang, B. Qu, Y. Luo, and C. Yue. Dynamic selection
% preference-assisted constrained multiobjective differential evolution.
% IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2022, 52(5):
% 2954-2965.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm, Problem)
            %% Initialize Strategy v4
            strategy = Strategyv4();
            
            %% Generate random population
            Population = Problem.Initialization();
            
            %% Initialize state
            state = strategy.GetState(Population, Problem);
            strategy.prev_state = state;
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Get current progress
                progress = Problem.FE / Problem.maxFE;
                
                % Get current state
                current_state = strategy.GetState(Population, Problem);
                
                % Select actions (operator and weight)
                [action_w, action_op] = strategy.SelectActions(current_state, progress);
                
                % Store actions for use in offspring generation
                strategy.prev_action_w = action_w;
                strategy.prev_action_op = action_op;
                
                % Generate offspring using RL-selected operator
                Offspring = DEgenerator2_v4(Problem, Population, strategy);
                
                % Calculate weight using RL
                a = strategy.ApplyWeightAction(action_w, progress);
                
                % Environmental selection
                [Population, ~, ~] = EnvironmentalSelection_v4([Population, Offspring], ...
                                                               Problem.N, a, strategy, progress);
                
                % Get next state
                next_state = strategy.GetState(Population, Problem);
                
                % Calculate reward
                reward = strategy.CalculateReward(Population, Problem, progress);
                
                % Store experience
                strategy.StoreExperience(state, action_w, action_op, reward, next_state);
                
                % Train neural network
                strategy.Train();
                
                % Check IGD deviation and activate callback if needed
                if progress > 2/3  % Only in late phase
                    strategy.CheckIGDDeviation(Population, Problem);
                end
                
                % Update state for next iteration
                state = next_state;
            end
        end
    end
end
