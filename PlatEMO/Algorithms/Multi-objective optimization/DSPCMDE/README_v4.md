# DSPCMDE Strategy v4 Implementation

## Overview
This directory contains the implementation of Strategy v4 for the DSPCMDE (Dynamic Selection Preference-assisted Constrained Multiobjective Differential Evolution) algorithm.

## Files
- **DSPCMDE_v4.m**: Main algorithm file integrating Strategy v4
- **Strategyv4.m**: Core RL strategy class implementing dual-head architecture
- **DEgenerator2_v4.m**: DE operator generator using RL-selected actions
- **EnvironmentalSelection_v4.m**: Environmental selection with RL-based weight adjustment

## Strategy v4 Features

### 1. Dual-Head RL Architecture
- **Head_w**: Manages weight adjustment actions (a_w ∈ {+δ, 0, −δ})
- **Head_op**: Manages operator selection actions (a_op ∈ {1..12})
- **Shared Encoder**: Both heads share a common feature encoder

### 2. Operator Actions (12 total)
- **Diversity Actions (1-3)**: DE/current-to-rand/1 with F ∈ {0.6, 0.8, 1.0}
- **Convergence Actions (4-12)**: DE/rand-to-best/1/bin with:
  - F ∈ {0.6, 0.8, 1.0}
  - CR ∈ {0.1, 0.2, 1.0}

### 3. Phase-Based Reward Functions

#### Early Phase ([0, 1/3))
- Focus: Objective improvements only
- Reward: r_E = 0.5 * E_min + 0.3 * E_max + 0.2 * E_mean

#### Mid Phase ([1/3, 2/3))
- Focus: Gradual transition from objectives to feasibility
- Reward: Weighted combination with shifting weights

#### Late Phase ([2/3, 1])
- Focus: Strict feasibility (FR=1) and monotonically decreasing IGD
- Reward: r_L = 1.0 * dIGD + 0.5 * d1 - 2.0 * P_mono - 3.0 * (1 - FR)

### 4. State Representation
Normalized state vector: [F_min_norm, F_max_norm, F_mean_norm, CV_mean_norm, FR]

### 5. IGD-Based Callback Modifier
When IGD deviation exceeds threshold for 2 consecutive generations:
- Reduce exploration: ε = 0.05
- Halve step size: δ_t /= 2
- Bias towards convergence operators
- Add penalty to reward

### 6. Time-Annealed Step Size
- δ_t = δ_init * (1 - progress)
- Phase-based constraints on weight values:
  - Early: [2/3, 1]
  - Mid: [1/3, 2/3]
  - Late: [0, 1/3]

## Usage

```matlab
% Create algorithm instance
Algorithm = DSPCMDE_v4();

% Create problem instance
Problem = Problem_Name('N', 100, 'M', 2, 'D', 30);

% Solve
Algorithm.Solve(Problem);
```

## Implementation Details

### Neural Network Architecture
- Input layer: 5 neurons (state features)
- Encoder: 16 neurons with tanh activation
- Head_w: 3 output neurons (weight actions)
- Head_op: 12 output neurons (operator actions)

### Experience Replay
- Buffer size: 1000 experiences
- Batch size: 32 for training
- Circular buffer implementation

### Learning Parameters
- Learning rate: 0.01
- Discount factor (γ): 0.95
- Initial epsilon: 0.3
- Minimum epsilon: 0.05
- Epsilon decay: 0.995

## References
K. Yu, J. Liang, B. Qu, Y. Luo, and C. Yue. Dynamic selection preference-assisted constrained multiobjective differential evolution. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2022, 52(5): 2954-2965.
