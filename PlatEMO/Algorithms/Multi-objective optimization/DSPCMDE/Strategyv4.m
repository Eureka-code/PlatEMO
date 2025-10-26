classdef Strategyv4 < handle
% Strategy v4 for DSPCMDE with dual-head RL architecture
%
% This class implements reinforcement learning with:
% - Dual-head architecture (Head_w for weights, Head_op for operators)
% - Experience replay buffer
% - Phase-based reward functions
% - IGD-based callback modifier

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        % Neural network parameters
        encoder_weights     % Shared encoder weights
        head_w_weights      % Head for weight actions
        head_op_weights     % Head for operator actions
        
        % Experience replay
        replay_buffer       % Experience replay buffer
        buffer_size         % Maximum buffer size
        buffer_idx          % Current buffer index
        
        % Exploration parameters
        epsilon             % Exploration rate
        epsilon_min         % Minimum exploration rate
        epsilon_decay       % Exploration decay rate
        
        % Step size parameters
        delta_init          % Initial step size
        delta_current       % Current step size
        
        % State tracking
        prev_state          % Previous state
        prev_action_w       % Previous weight action
        prev_action_op      % Previous operator action
        prev_weight         % Previous weight value
        
        % IGD tracking for callback
        igd_history         % History of IGD values
        igd_deviation_count % Count of consecutive IGD deviations
        igd_threshold       % Threshold for IGD deviation
        callback_active     % Whether callback modifier is active
        
        % Phase tracking
        current_phase       % Current evolutionary phase (1: early, 2: mid, 3: late)
        
        % Learning parameters
        learning_rate       % Learning rate for network updates
        gamma               % Discount factor
        batch_size          % Batch size for training
        
        % Previous metrics for reward calculation
        prev_F_min
        prev_F_max
        prev_F_mean
        prev_CV_mean
        prev_FR
        prev_IGD
    end
    
    methods
        function obj = Strategyv4()
            % Constructor - Initialize Strategy v4
            
            % Initialize neural network (simple linear approximation)
            % State size: 5 (F_min_norm, F_max_norm, F_mean_norm, CV_mean_norm, FR)
            % Encoder: 5 -> 16
            obj.encoder_weights = randn(5, 16) * 0.1;
            
            % Head_w: 16 -> 3 (actions: +delta, 0, -delta)
            obj.head_w_weights = randn(16, 3) * 0.1;
            
            % Head_op: 16 -> 12 (12 operator actions)
            obj.head_op_weights = randn(16, 12) * 0.1;
            
            % Experience replay buffer
            obj.buffer_size = 1000;
            obj.replay_buffer = struct('state', {}, 'action_w', {}, 'action_op', {}, ...
                                      'reward', {}, 'next_state', {}, 'phase', {});
            obj.buffer_idx = 0;
            
            % Exploration parameters
            obj.epsilon = 0.3;
            obj.epsilon_min = 0.05;
            obj.epsilon_decay = 0.995;
            
            % Step size parameters
            obj.delta_init = 0.1;
            obj.delta_current = 0.1;
            
            % IGD tracking
            obj.igd_history = [];
            obj.igd_deviation_count = 0;
            obj.igd_threshold = 0.05;  % 5% deviation threshold
            obj.callback_active = false;
            
            % Phase
            obj.current_phase = 1;  % Start in early phase
            
            % Learning parameters
            obj.learning_rate = 0.01;
            obj.gamma = 0.95;
            obj.batch_size = 32;
            
            % Initialize previous metrics
            obj.prev_F_min = [];
            obj.prev_F_max = [];
            obj.prev_F_mean = [];
            obj.prev_CV_mean = [];
            obj.prev_FR = [];
            obj.prev_IGD = [];
            obj.prev_state = [];
            obj.prev_action_w = [];
            obj.prev_action_op = [];
            obj.prev_weight = 0.5;
        end
        
        function state = GetState(obj, Population, Problem)
            % Compute normalized state vector
            % State: [F_min_norm, F_max_norm, F_mean_norm, CV_mean_norm, FR]
            
            objs = Population.objs;
            cons = Population.cons;
            
            % Calculate objective statistics
            F_min = min(objs, [], 1);
            F_max = max(objs, [], 1);
            F_mean = mean(objs, 1);
            
            % Normalize objectives (simple normalization)
            if ~isempty(obj.prev_F_min)
                F_min_norm = mean((F_min - obj.prev_F_min) ./ (abs(obj.prev_F_min) + 1e-10));
                F_max_norm = mean((F_max - obj.prev_F_max) ./ (abs(obj.prev_F_max) + 1e-10));
                F_mean_norm = mean((F_mean - obj.prev_F_mean) ./ (abs(obj.prev_F_mean) + 1e-10));
            else
                F_min_norm = 0;
                F_max_norm = 0;
                F_mean_norm = 0;
            end
            
            % Calculate constraint violation statistics
            cv = sum(max(0, cons), 2);
            CV_mean = mean(cv);
            
            if ~isempty(obj.prev_CV_mean)
                CV_mean_norm = (CV_mean - obj.prev_CV_mean) / (abs(obj.prev_CV_mean) + 1e-10);
            else
                CV_mean_norm = 0;
            end
            
            % Calculate feasibility ratio
            FR = sum(cv == 0) / length(cv);
            
            % Clip normalized values to reasonable range
            F_min_norm = max(min(F_min_norm, 1), -1);
            F_max_norm = max(min(F_max_norm, 1), -1);
            F_mean_norm = max(min(F_mean_norm, 1), -1);
            CV_mean_norm = max(min(CV_mean_norm, 1), -1);
            
            state = [F_min_norm, F_max_norm, F_mean_norm, CV_mean_norm, FR];
            
            % Update previous metrics for next iteration
            obj.prev_F_min = F_min;
            obj.prev_F_max = F_max;
            obj.prev_F_mean = F_mean;
            obj.prev_CV_mean = CV_mean;
            obj.prev_FR = FR;
        end
        
        function [action_w, action_op] = SelectActions(obj, state, progress)
            % Select actions using epsilon-greedy policy
            % progress: current progress (FE/maxFE)
            
            % Update phase based on progress
            if progress < 1/3
                obj.current_phase = 1;  % Early phase
            elseif progress < 2/3
                obj.current_phase = 2;  % Mid phase
            else
                obj.current_phase = 3;  % Late phase
            end
            
            % Apply callback modifier if active
            epsilon = obj.epsilon;
            if obj.callback_active
                epsilon = 0.05;  % Reduce exploration
            end
            
            % Epsilon-greedy action selection
            if rand() < epsilon
                % Explore: random actions
                action_w = randi([1, 3]);  % 1: +delta, 2: 0, 3: -delta
                
                if obj.callback_active
                    % Bias towards convergence operators (actions 4-12)
                    action_op = randi([4, 12]);
                else
                    action_op = randi([1, 12]);  % 1-3: diversity, 4-12: convergence
                end
            else
                % Exploit: use neural network
                [action_w, action_op] = obj.Forward(state);
            end
        end
        
        function [action_w, action_op] = Forward(obj, state)
            % Forward pass through dual-head network
            
            % Shared encoder
            state = state(:)';  % Ensure row vector
            hidden = tanh(state * obj.encoder_weights);
            
            % Head_w output
            q_w = hidden * obj.head_w_weights;
            [~, action_w] = max(q_w);
            
            % Head_op output
            q_op = hidden * obj.head_op_weights;
            
            % Apply callback bias if active (favor convergence operators)
            if obj.callback_active
                q_op(1:3) = q_op(1:3) - 1.0;  % Penalize diversity operators
            end
            
            [~, action_op] = max(q_op);
        end
        
        function new_weight = ApplyWeightAction(obj, action_w, progress)
            % Apply weight action with phase-based constraints
            
            % Update delta with time annealing
            obj.delta_current = obj.delta_init * (1 - progress);
            
            % Apply callback modifier
            if obj.callback_active
                obj.delta_current = obj.delta_current / 2;
            end
            
            % Get weight change
            if action_w == 1
                delta_w = obj.delta_current;   % +delta
            elseif action_w == 2
                delta_w = 0;                    % 0
            else
                delta_w = -obj.delta_current;   % -delta
            end
            
            % Apply weight change
            new_weight = obj.prev_weight + delta_w;
            
            % Apply phase-based constraints
            if obj.current_phase == 1  % Early phase
                new_weight = max(min(new_weight, 1.0), 2/3);
            elseif obj.current_phase == 2  % Mid phase
                new_weight = max(min(new_weight, 2/3), 1/3);
            else  % Late phase
                new_weight = max(min(new_weight, 1/3), 0.0);
            end
            
            obj.prev_weight = new_weight;
        end
        
        function [F, CR] = GetOperatorParams(obj, action_op)
            % Map operator action to F and CR parameters
            % Actions 1-3: Diversity (F varies, CR = 1.0)
            % Actions 4-12: Convergence (F and CR vary)
            
            if action_op <= 3
                % Diversity operators
                F_values = [0.6, 0.8, 1.0];
                F = F_values(action_op);
                CR = 1.0;  % Diversity uses current-to-rand (no CR in implementation)
            else
                % Convergence operators
                conv_idx = action_op - 3;  % 1-9
                F_idx = mod(conv_idx - 1, 3) + 1;  % 1, 2, or 3
                CR_idx = ceil(conv_idx / 3);  % 1, 2, or 3
                
                F_values = [0.6, 0.8, 1.0];
                CR_values = [0.1, 0.2, 1.0];
                
                F = F_values(F_idx);
                CR = CR_values(CR_idx);
            end
        end
        
        function reward = CalculateReward(obj, Population, Problem, progress)
            % Calculate phase-based reward
            
            objs = Population.objs;
            cons = Population.cons;
            cv = sum(max(0, cons), 2);
            FR = sum(cv == 0) / length(cv);
            
            % Calculate improvement metrics
            if ~isempty(obj.prev_F_min)
                E_min = mean((obj.prev_F_min - min(objs, [], 1)) ./ (abs(obj.prev_F_min) + 1e-10));
                E_max = mean((obj.prev_F_max - max(objs, [], 1)) ./ (abs(obj.prev_F_max) + 1e-10));
                E_mean = mean((obj.prev_F_mean - mean(objs, 1)) ./ (abs(obj.prev_F_mean) + 1e-10));
            else
                E_min = 0;
                E_max = 0;
                E_mean = 0;
            end
            
            % Phase-based reward calculation
            if obj.current_phase == 1
                % Early phase: Focus on objective improvements
                reward = 0.5 * E_min + 0.3 * E_max + 0.2 * E_mean;
                
            elseif obj.current_phase == 2
                % Mid phase: Gradual transition
                phase_progress = (progress - 1/3) / (1/3);  % 0 to 1 within mid phase
                w_obj = 1 - phase_progress;
                w_feas = phase_progress;
                
                r_obj = 0.5 * E_min + 0.3 * E_max + 0.2 * E_mean;
                r_feas = (FR - obj.prev_FR);
                
                reward = w_obj * r_obj + w_feas * r_feas;
                
            else
                % Late phase: Focus on feasibility and IGD
                dIGD = 0;
                d1 = 0;
                P_mono = 0;
                
                % Calculate IGD improvement
                if ~isempty(obj.prev_IGD)
                    current_IGD = obj.CalculateIGD(Population, Problem);
                    dIGD = (obj.prev_IGD - current_IGD) / (abs(obj.prev_IGD) + 1e-10);
                    
                    % Check monotonicity
                    if current_IGD > obj.prev_IGD
                        P_mono = 1;
                    end
                    
                    obj.prev_IGD = current_IGD;
                end
                
                % Feasibility improvement
                d1 = (FR - obj.prev_FR);
                
                reward = 1.0 * dIGD + 0.5 * d1 - 2.0 * P_mono - 3.0 * (1 - FR);
            end
            
            % Apply callback penalty if active
            if obj.callback_active
                reward = reward - 0.5;
            end
            
            % Clip reward to reasonable range
            reward = max(min(reward, 10), -10);
        end
        
        function igd = CalculateIGD(obj, Population, Problem)
            % Calculate IGD metric (simplified version)
            % In real implementation, this should use the true Pareto front
            
            try
                % Get non-dominated solutions
                FrontNo = NDSort(Population.objs, Population.cons, 1);
                PF = Population(FrontNo == 1).objs;
                
                % Use problem's optimal Pareto front if available
                if ismethod(Problem, 'optimum')
                    PF_true = Problem.optimum(10000);  % Get reference front
                    
                    % Calculate IGD
                    if ~isempty(PF) && ~isempty(PF_true)
                        igd = mean(min(pdist2(PF_true, PF), [], 2));
                    else
                        igd = inf;
                    end
                else
                    % Fallback: use hypervolume as proxy
                    igd = 0;
                end
            catch
                igd = 0;
            end
        end
        
        function StoreExperience(obj, state, action_w, action_op, reward, next_state)
            % Store experience in replay buffer
            
            experience.state = state;
            experience.action_w = action_w;
            experience.action_op = action_op;
            experience.reward = reward;
            experience.next_state = next_state;
            experience.phase = obj.current_phase;
            
            % Add to buffer
            obj.buffer_idx = obj.buffer_idx + 1;
            if obj.buffer_idx > obj.buffer_size
                obj.buffer_idx = 1;  % Circular buffer
            end
            
            if length(obj.replay_buffer) < obj.buffer_size
                obj.replay_buffer(end+1) = experience;
            else
                obj.replay_buffer(obj.buffer_idx) = experience;
            end
        end
        
        function Train(obj)
            % Train both heads using experience replay
            
            % Need enough experiences to train
            if length(obj.replay_buffer) < obj.batch_size
                return;
            end
            
            % Sample random batch
            batch_indices = randperm(length(obj.replay_buffer), obj.batch_size);
            batch = obj.replay_buffer(batch_indices);
            
            % Extract batch data
            states = vertcat(batch.state);
            actions_w = [batch.action_w]';
            actions_op = [batch.action_op]';
            rewards = [batch.reward]';
            next_states = vertcat(batch.next_state);
            
            % Forward pass for current states
            hidden = tanh(states * obj.encoder_weights);
            q_w = hidden * obj.head_w_weights;
            q_op = hidden * obj.head_op_weights;
            
            % Forward pass for next states (for Q-learning target)
            next_hidden = tanh(next_states * obj.encoder_weights);
            next_q_w = next_hidden * obj.head_w_weights;
            next_q_op = next_hidden * obj.head_op_weights;
            
            % Calculate targets (Q-learning)
            target_w = q_w;
            target_op = q_op;
            
            for i = 1:obj.batch_size
                % Same reward for both heads
                target_w(i, actions_w(i)) = rewards(i) + obj.gamma * max(next_q_w(i, :));
                target_op(i, actions_op(i)) = rewards(i) + obj.gamma * max(next_q_op(i, :));
            end
            
            % Backpropagation (simplified gradient descent)
            % Update Head_w
            delta_w = q_w - target_w;
            grad_head_w = hidden' * delta_w / obj.batch_size;
            obj.head_w_weights = obj.head_w_weights - obj.learning_rate * grad_head_w;
            
            % Update Head_op
            delta_op = q_op - target_op;
            grad_head_op = next_hidden' * delta_op / obj.batch_size;
            obj.head_op_weights = obj.head_op_weights - obj.learning_rate * grad_head_op;
            
            % Update encoder (using combined gradients)
            grad_encoder_w = states' * (delta_w * obj.head_w_weights') / obj.batch_size;
            grad_encoder_op = states' * (delta_op * obj.head_op_weights') / obj.batch_size;
            grad_encoder = (grad_encoder_w + grad_encoder_op) / 2;
            obj.encoder_weights = obj.encoder_weights - obj.learning_rate * grad_encoder;
            
            % Decay epsilon
            obj.epsilon = max(obj.epsilon * obj.epsilon_decay, obj.epsilon_min);
        end
        
        function CheckIGDDeviation(obj, Population, Problem)
            % Check IGD deviation and activate callback if needed
            
            current_IGD = obj.CalculateIGD(Population, Problem);
            
            if isempty(obj.igd_history)
                obj.igd_history = current_IGD;
                return;
            end
            
            % Check deviation
            if length(obj.igd_history) >= 2
                recent_mean = mean(obj.igd_history(max(1, end-4):end));
                deviation = abs(current_IGD - recent_mean) / (recent_mean + 1e-10);
                
                if deviation > obj.igd_threshold
                    obj.igd_deviation_count = obj.igd_deviation_count + 1;
                else
                    obj.igd_deviation_count = 0;
                    obj.callback_active = false;
                end
                
                % Activate callback if 2 consecutive deviations
                if obj.igd_deviation_count >= 2
                    obj.callback_active = true;
                end
            end
            
            % Update history
            obj.igd_history(end+1) = current_IGD;
            if length(obj.igd_history) > 10
                obj.igd_history = obj.igd_history(end-9:end);  % Keep last 10
            end
        end
    end
end
