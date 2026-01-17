# PPO Solver for VNR Placement

## Overview

The PPO (Proximal Policy Optimization) solver is a state-of-the-art reinforcement learning algorithm for Virtual Network Request (VNR) placement. It provides more stable training than vanilla policy gradients by using a clipped surrogate objective.

## Key Features

- **Clipped Objective**: Prevents excessively large policy updates
- **Multiple Epochs**: Trains multiple times on the same batch for better sample efficiency
- **Entropy Bonus**: Encourages exploration
- **GAE (Generalized Advantage Estimation)**: Better advantage computation
- **GPU Optimized**: Leverages GPU acceleration for faster training

## Configuration

To use PPO in your `parameters.json`, add a solver with `type: "GNNDRLPPO"`:

```json
{
    "name": "PPO_Solver",
    "type": "GNNDRLPPO",
    "sigma": 0.8,
    "gamma": 0.99,
    "rejection_penalty": -1,
    "learning_rate": 3e-4,
    "epsilon": 1.0,
    "memory_size": 1000000,
    "batch_size": 64,
    "num_inputs_sn": 8,
    "num_inputs_vnr": 7,
    "hidden_size": 64,
    "GCN_out": 32,
    "num_actions": 24,
    "max_itteration": 32,
    "eps_min": 0.01,
    "eps_dec": 2.5e-5,
    "clip_ratio": 0.2,
    "ppo_epochs": 4,
    "entropy_coef": 0.01
}
```

## Parameters

### Standard Parameters (shared with GNNDRL)
- `sigma`: Reward calculation parameter
- `gamma`: Discount factor (typically 0.99)
- `rejection_penalty`: Penalty for failed placements
- `learning_rate`: Learning rate (3e-4 is common for PPO)
- `epsilon`: Initial exploration rate
- `memory_size`: Replay buffer size
- `batch_size`: Batch size (64 or 128 recommended for PPO)
- `num_inputs_sn/vnr`: Input feature dimensions
- `hidden_size`: Hidden layer size
- `GCN_out`: GCN output dimension
- `num_actions`: Number of substrate nodes
- `max_itteration`: Max placement attempts
- `eps_min/eps_dec`: Epsilon decay parameters

### PPO-Specific Parameters
- `clip_ratio` (default: 0.2): PPO clipping parameter
  - Controls how much the policy can change in one update
  - Typical values: 0.1 to 0.3
  
- `ppo_epochs` (default: 4): Number of training epochs per batch
  - Higher values = better sample efficiency but slower
  - Typical values: 3 to 10
  
- `entropy_coef` (default: 0.01): Entropy bonus coefficient
  - Encourages exploration
  - Typical values: 0.001 to 0.1

## Advantages over A2C

1. **Stability**: Clipped objective prevents destructive policy updates
2. **Sample Efficiency**: Multiple epochs on same data
3. **Better Exploration**: Entropy bonus
4. **State-of-the-art**: PPO is currently one of the best policy gradient methods

## Recommended Hyperparameters

### Conservative (stable, slower learning)
```json
{
    "learning_rate": 1e-4,
    "batch_size": 128,
    "clip_ratio": 0.1,
    "ppo_epochs": 10,
    "entropy_coef": 0.01
}
```

### Balanced (good default)
```json
{
    "learning_rate": 3e-4,
    "batch_size": 64,
    "clip_ratio": 0.2,
    "ppo_epochs": 4,
    "entropy_coef": 0.01
}
```

### Aggressive (faster learning, less stable)
```json
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "clip_ratio": 0.3,
    "ppo_epochs": 3,
    "entropy_coef": 0.001
}
```

## Performance Tips

1. **Learning Rate**: Start with 3e-4, reduce if training is unstable
2. **Batch Size**: Larger is better (64-128), but requires more memory
3. **PPO Epochs**: 4-10 is typical, more can overfit to old data
4. **Clip Ratio**: 0.2 works well, reduce for more stability
5. **Entropy**: Start with 0.01, reduce over time if needed

## Monitoring

Watch for:
- **Acceptance Rate**: Should increase over time
- **Average Reward**: Should increase and stabilize
- **Epsilon**: Should decay to eps_min
- **Training Loss**: Should decrease initially

## GPU Optimizations

The PPO implementation includes all GPU optimizations:
- Cached VNR graphs
- Fast SN copying (`copy_for_placement`)
- Tensors kept on GPU
- Vectorized operations
- Minimal CPU-GPU transfers

Expected speedup: **2-3x faster** than baseline implementations.

## Example Usage

```python
# In parameters.json, add PPO solver
{
    "solvers": [
        {
            "name": "PPO_Primary",
            "type": "GNNDRLPPO",
            "sigma": 0.8,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "batch_size": 64,
            "clip_ratio": 0.2,
            "ppo_epochs": 4,
            ...
        }
    ]
}
```

Then run:
```bash
python main.py
```

## Troubleshooting

**Training is unstable:**
- Reduce `learning_rate` (try 1e-4)
- Reduce `clip_ratio` (try 0.1)
- Increase `batch_size`

**Learning too slow:**
- Increase `learning_rate` (try 1e-3)
- Increase `ppo_epochs` (try 6-8)
- Reduce `entropy_coef`

**GPU out of memory:**
- Reduce `batch_size`
- Reduce `hidden_size` or `GCN_out`

## References

- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
