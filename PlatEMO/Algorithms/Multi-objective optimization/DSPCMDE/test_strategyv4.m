%% Test script for DSPCMDE Strategy v4
% This script demonstrates how to use DSPCMDE_v4 with Strategy v4
%
% Note: This is a demonstration script. To actually run it, you need:
% - MATLAB or GNU Octave installed
% - PlatEMO framework properly set up
% - A constrained multi-objective test problem

%% Example Usage

% Add PlatEMO paths (adjust as needed)
% addpath(genpath('/path/to/PlatEMO'));

% Create a constrained multi-objective problem
% Example: MW1 from PlatEMO's problem suite
% Problem = MW1('N', 100, 'M', 2, 'D', 30);

% Create DSPCMDE_v4 algorithm instance
% Algorithm = DSPCMDE_v4();

% Run the algorithm
% Algorithm.Solve(Problem);

%% Verification Tests (Conceptual - would need MATLAB to run)

% Test 1: Verify Strategyv4 initialization
% strategy = Strategyv4();
% assert(~isempty(strategy.encoder_weights), 'Encoder weights should be initialized');
% assert(size(strategy.encoder_weights, 1) == 5, 'Encoder input should be 5');
% assert(size(strategy.encoder_weights, 2) == 16, 'Encoder output should be 16');
% assert(size(strategy.head_w_weights, 2) == 3, 'Head_w should output 3 actions');
% assert(size(strategy.head_op_weights, 2) == 12, 'Head_op should output 12 actions');

% Test 2: Verify operator action mapping
% strategy = Strategyv4();
% for i = 1:12
%     [F, CR] = strategy.GetOperatorParams(i);
%     if i <= 3
%         assert(ismember(F, [0.6, 0.8, 1.0]), 'F should be 0.6, 0.8, or 1.0');
%         assert(CR == 1.0, 'Diversity operators should have CR=1.0');
%     else
%         assert(ismember(F, [0.6, 0.8, 1.0]), 'F should be 0.6, 0.8, or 1.0');
%         assert(ismember(CR, [0.1, 0.2, 1.0]), 'CR should be 0.1, 0.2, or 1.0');
%     end
% end

% Test 3: Verify phase-based weight constraints
% strategy = Strategyv4();
% % Early phase
% strategy.current_phase = 1;
% strategy.prev_weight = 0.5;
% w = strategy.ApplyWeightAction(1, 0.1); % action=+delta, progress=0.1
% assert(w >= 2/3 && w <= 1.0, 'Early phase weight should be in [2/3, 1]');
% 
% % Mid phase
% strategy.current_phase = 2;
% strategy.prev_weight = 0.5;
% w = strategy.ApplyWeightAction(1, 0.5);
% assert(w >= 1/3 && w <= 2/3, 'Mid phase weight should be in [1/3, 2/3]');
% 
% % Late phase
% strategy.current_phase = 3;
% strategy.prev_weight = 0.2;
% w = strategy.ApplyWeightAction(1, 0.9);
% assert(w >= 0.0 && w <= 1/3, 'Late phase weight should be in [0, 1/3]');

fprintf('Strategy v4 implementation is ready for testing!\n');
fprintf('To use it:\n');
fprintf('1. Ensure PlatEMO is in your MATLAB path\n');
fprintf('2. Create a constrained multi-objective problem\n');
fprintf('3. Instantiate DSPCMDE_v4 and call Solve(Problem)\n');
fprintf('\nExample:\n');
fprintf('  Algorithm = DSPCMDE_v4();\n');
fprintf('  Problem = MW1(''N'', 100, ''M'', 2, ''D'', 30);\n');
fprintf('  Algorithm.Solve(Problem);\n');
