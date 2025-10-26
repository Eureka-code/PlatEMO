# Strategy v4 Technical Specification

## Architecture Overview

### Component Diagram
```
DSPCMDE_v4 (Main Algorithm)
    │
    ├── Strategyv4 (RL Controller)
    │   ├── Neural Network
    │   │   ├── Shared Encoder (5 → 16)
    │   │   ├── Head_w (16 → 3)  [Weight actions]
    │   │   └── Head_op (16 → 12) [Operator actions]
    │   ├── Experience Replay Buffer
    │   └── IGD Callback Modifier
    │
    ├── DEgenerator2_v4 (Offspring Generation)
    │   ├── Diversity Operators (Actions 1-3)
    │   └── Convergence Operators (Actions 4-12)
    │
    └── EnvironmentalSelection_v4 (Selection)
        └── CDP/Pareto Ranking with RL Weight
```

## State Space

### State Vector (5 dimensions)
- `F_min_norm`: Normalized change in minimum objective values
- `F_max_norm`: Normalized change in maximum objective values
- `F_mean_norm`: Normalized change in mean objective values
- `CV_mean_norm`: Normalized change in mean constraint violation
- `FR`: Feasibility ratio (fraction of feasible solutions)

### Normalization
```matlab
F_min_norm = mean((F_min - prev_F_min) / (|prev_F_min| + ε))
# Clipped to [-1, 1]
```

## Action Space

### Weight Actions (Head_w)
| Action | Description | Effect on Weight |
|--------|-------------|------------------|
| 1      | Increase    | w = w + δ_t      |
| 2      | Maintain    | w = w            |
| 3      | Decrease    | w = w - δ_t      |

### Operator Actions (Head_op)

#### Diversity Operators (Actions 1-3)
| Action | Operator | F   | CR  | Description |
|--------|----------|-----|-----|-------------|
| 1      | Diversity| 0.6 | 1.0 | DE/current-to-rand/1 |
| 2      | Diversity| 0.8 | 1.0 | DE/current-to-rand/1 |
| 3      | Diversity| 1.0 | 1.0 | DE/current-to-rand/1 |

#### Convergence Operators (Actions 4-12)
| Action | Operator     | F   | CR  | Description |
|--------|--------------|-----|-----|-------------|
| 4      | Convergence  | 0.6 | 0.1 | DE/rand-to-best/1/bin |
| 5      | Convergence  | 0.8 | 0.1 | DE/rand-to-best/1/bin |
| 6      | Convergence  | 1.0 | 0.1 | DE/rand-to-best/1/bin |
| 7      | Convergence  | 0.6 | 0.2 | DE/rand-to-best/1/bin |
| 8      | Convergence  | 0.8 | 0.2 | DE/rand-to-best/1/bin |
| 9      | Convergence  | 1.0 | 0.2 | DE/rand-to-best/1/bin |
| 10     | Convergence  | 0.6 | 1.0 | DE/rand-to-best/1/bin |
| 11     | Convergence  | 0.8 | 1.0 | DE/rand-to-best/1/bin |
| 12     | Convergence  | 1.0 | 1.0 | DE/rand-to-best/1/bin |

## Reward Functions

### Early Phase (t ∈ [0, 1/3))
**Objective**: Maximize objective improvements

```matlab
r_E = 0.5 * E_min + 0.3 * E_max + 0.2 * E_mean
```

where:
- `E_min = (prev_F_min - F_min) / |prev_F_min|`
- `E_max = (prev_F_max - F_max) / |prev_F_max|`
- `E_mean = (prev_F_mean - F_mean) / |prev_F_mean|`

### Mid Phase (t ∈ [1/3, 2/3))
**Objective**: Gradual transition to feasibility

```matlab
phase_progress = (t - 1/3) / (1/3)
w_obj = 1 - phase_progress
w_feas = phase_progress

r_obj = 0.5 * E_min + 0.3 * E_max + 0.2 * E_mean
r_feas = FR - prev_FR

r_M = w_obj * r_obj + w_feas * r_feas
```

### Late Phase (t ∈ [2/3, 1])
**Objective**: Achieve FR=1 and monotonically decreasing IGD

```matlab
r_L = 1.0 * dIGD + 0.5 * d1 - 2.0 * P_mono - 3.0 * (1 - FR)
```

where:
- `dIGD = (prev_IGD - current_IGD) / |prev_IGD|`
- `d1 = FR - prev_FR`
- `P_mono = 1 if IGD increased, 0 otherwise`

### Callback Penalty
```matlab
if callback_active:
    r = r - 0.5
```

## Time-Annealed Step Size

```matlab
δ_t = δ_init * (1 - progress)

if callback_active:
    δ_t = δ_t / 2
```

## Phase-Based Weight Constraints

| Phase | Progress Range | Weight Range |
|-------|----------------|--------------|
| Early | [0, 1/3)       | [2/3, 1]     |
| Mid   | [1/3, 2/3)     | [1/3, 2/3]   |
| Late  | [2/3, 1]       | [0, 1/3]     |

## IGD Callback Modifier

### Trigger Conditions
1. Progress > 2/3 (Late phase only)
2. IGD deviation > threshold (5%) for 2 consecutive generations

### Callback Effects
When activated:
- `ε = 0.05` (reduced exploration)
- `δ_t = δ_t / 2` (halved step size)
- Bias Q-values: `Q_op[1:3] -= 1.0` (penalize diversity)
- Random action selection: `a_op ∈ [4, 12]` (favor convergence)
- Reward penalty: `r -= 0.5`

### Deactivation
Callback deactivates when deviation drops below threshold.

## Neural Network Training

### Q-Learning Update
```matlab
target = r + γ * max(Q(s'))

For each action in batch:
    Q_target[a] = target
    
Loss = MSE(Q, Q_target)
```

### Gradient Descent
```matlab
# Head updates
δ_w = Q_w - Q_target_w
∇_head_w = hidden^T * δ_w / batch_size
head_w_weights -= α * ∇_head_w

# Encoder update (combined gradients)
∇_encoder = (∇_encoder_w + ∇_encoder_op) / 2
encoder_weights -= α * ∇_encoder
```

### Hyperparameters
- Learning rate (α): 0.01
- Discount factor (γ): 0.95
- Batch size: 32
- Buffer size: 1000
- Initial ε: 0.3
- Min ε: 0.05
- ε decay: 0.995

## Integration Flow

### Initialization
1. Create Strategyv4 instance
2. Initialize random population
3. Compute initial state

### Main Loop (Each Generation)
1. Get current state from population
2. Select actions (weight + operator) via ε-greedy
3. Generate offspring using selected operator
4. Apply weight to environmental selection
5. Compute next state from new population
6. Calculate reward based on phase
7. Store experience in replay buffer
8. Train neural network (if buffer sufficient)
9. Check IGD deviation (if late phase)
10. Update state for next iteration

### Experience Replay
- Store: `(state, action_w, action_op, reward, next_state, phase)`
- Circular buffer with size 1000
- Random batch sampling of size 32
- Train when buffer has ≥ 32 experiences

## File Structure

```
DSPCMDE/
├── DSPCMDE.m                    # Original algorithm
├── DSPCMDE_v4.m                 # Strategy v4 algorithm
├── DEgenerator2.m               # Original DE operator
├── DEgenerator2_v4.m            # RL-based DE operator
├── EnvironmentalSelection.m     # Original selection
├── EnvironmentalSelection_v4.m  # RL-based selection
├── Strategyv4.m                 # Main RL strategy class
├── README_v4.md                 # User documentation
├── TECHNICAL_SPEC.md            # This file
└── test_strategyv4.m            # Test script
```

## Key Differences from Original DSPCMDE

| Aspect | Original | Strategy v4 |
|--------|----------|-------------|
| Operator Selection | Random 50/50 | RL-based (12 actions) |
| Weight Calculation | Fixed formula | RL-based adjustment |
| Parameter Adaptation | Random from discrete set | RL-selected from discrete set |
| Reward Function | None | Phase-based multi-objective |
| Callback Mechanism | None | IGD-based adaptive |
| Learning | None | Experience replay + Q-learning |

## Performance Considerations

### Computational Complexity
- State computation: O(N*M) where N=population size, M=objectives
- Action selection: O(1) forward pass
- Training: O(batch_size * hidden_size^2) per generation
- IGD calculation: O(N*PF_size) if true Pareto front available

### Memory Usage
- Replay buffer: ~1000 experiences × (state + actions + reward)
- Neural network: ~(5×16 + 16×3 + 16×12) = 320 parameters

## Future Enhancements

Potential improvements:
1. Adaptive buffer size based on problem complexity
2. Prioritized experience replay
3. Target network for stable training
4. Multi-layer encoder for complex problems
5. Transfer learning between similar problems
6. Adaptive IGD threshold based on problem characteristics
