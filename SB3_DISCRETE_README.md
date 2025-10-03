# Stable-Baselines3 (SB3) Integration for FRE25 - DISCRETE Actions

## Overview
Switched from skrl to Stable-Baselines3 due to skrl's MultiCategorical action space bugs.
SB3 has excellent support for MultiDiscrete actions and is much more mature and stable.

## Files Created

### 1. Training & Evaluation Scripts
- `scripts/sb3/train.py` - SB3 training script
- `scripts/sb3/play.py` - SB3 evaluation/play script
- `RUN_SB3_TRAIN.sh` - Shell script to start training
- `RUN_SB3_PLAY.sh` - Shell script to evaluate trained model

### 2. Configuration
- `source/FRE25_IsaacLabSym/FRE25_IsaacLabSym/tasks/direct/fre25_isaaclabsym/agents/sb3_ppo_cfg.yaml`
  - PPO hyperparameters optimized for discrete actions
  - Network architecture: [512, 256, 256, 128, 128]
  - Higher entropy coefficient (0.01) for exploration with discrete actions

### 3. Environment Registration
- Updated `__init__.py` to register `sb3_cfg_entry_point`

## Action Space
**MultiDiscrete([3, 3, 3, 3, 3, 3])**
- 6 action dimensions (steering, throttle, 4 hidden state accumulators)
- Each action: 3 categories representing {-1, 0, +1}
- Total: 3^6 = 729 possible discrete action combinations

## Key Advantages of SB3

✅ **Native MultiDiscrete Support** - No bugs, well-tested
✅ **Mature Library** - Industry standard, used by thousands
✅ **Better Documentation** - Clear examples and guides
✅ **Simpler API** - Less boilerplate than skrl
✅ **IsaacLab Integration** - Official wrappers provided

## Usage

### Training
```bash
./RUN_SB3_TRAIN.sh

# With custom parameters
./RUN_SB3_TRAIN.sh --num_envs 128 --max_iterations 10000
```

### Evaluation
```bash
# Use best checkpoint
./RUN_SB3_PLAY.sh

# Use specific checkpoint
./RUN_SB3_PLAY.sh --checkpoint logs/sb3/Fre25-Isaaclabsym-Direct-v0/2025-10-03_12-34-56/model.zip

# With video recording
./RUN_SB3_PLAY.sh --video --video_length 500
```

### Tensorboard
```bash
tensorboard --logdir logs/sb3/Fre25-Isaaclabsym-Direct-v0
```

## Configuration Tuning

Edit `sb3_ppo_cfg.yaml` to adjust:
- **n_steps**: Steps per environment per update (default: 2048)
- **batch_size**: Minibatch size (default: 512)
- **learning_rate**: LR for optimizer (default: 0.0003)
- **ent_coef**: Entropy coefficient - increase for more exploration (default: 0.01)
- **gamma**: Discount factor (default: 0.996, horizon ~250 steps)
- **policy_kwargs/net_arch**: Network layer sizes

## Expected Behavior

SB3 automatically:
- Handles MultiDiscrete action space output layers
- Computes proper categorical distributions
- Samples discrete actions during training
- Uses argmax for deterministic evaluation

## Comparison: skrl vs SB3

| Feature | skrl | SB3 |
|---------|------|-----|
| MultiDiscrete Support | Buggy (entropy/stddev errors) | Excellent, well-tested |
| Documentation | Good | Excellent |
| Community | Growing | Large, established |
| GPU Memory | Efficient | CPU-based (slower with many envs) |
| Ease of Use | Moderate | Very Easy |

## Notes

- SB3 uses CPU for replay buffers, so use fewer environments (64-128) compared to skrl
- For MultiDiscrete, SB3 automatically creates separate categorical distributions
- Entropy bonus (ent_coef) is important for exploration in discrete action spaces
- Consider adjusting ent_coef if agent gets stuck in suboptimal policies

## Next Steps

1. Run training: `./RUN_SB3_TRAIN.sh`
2. Monitor with Tensorboard
3. Evaluate checkpoints: `./RUN_SB3_PLAY.sh`
4. Tune hyperparameters if needed

## Troubleshooting

**If training is slow:**
- Reduce `num_envs` (SB3 is CPU-based for memory)
- Use fewer environments (64 is good balance)

**If not learning:**
- Increase `ent_coef` for more exploration
- Check reward shaping in environment
- Verify action space mapping is correct

**If unstable:**
- Reduce `learning_rate`
- Increase `n_steps` for more data per update
- Adjust `clip_range`
