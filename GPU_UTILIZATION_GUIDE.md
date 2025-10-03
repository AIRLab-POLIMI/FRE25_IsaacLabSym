# GPU Utilization Guide - SB3 vs skrl vs RSL-RL

## Current Situation
- **Training with**: Stable-Baselines3 (SB3)
- **GPU Usage**: ~32%
- **Reason**: SB3 stores replay buffers on CPU (design limitation)

## Why Low GPU Usage?

### SB3 Architecture:
1. **Replay buffer**: CPU-based (numpy arrays)
2. **Data transfer**: Constant CPU→GPU copies
3. **Policy forward**: GPU (only part using GPU)
4. **Value forward**: GPU
5. **Gradient computation**: GPU
6. **Buffer storage**: CPU ❌

This CPU bottleneck is **by design** in SB3 for memory efficiency.

## Solutions to Increase GPU Usage

### ✅ Solution 1: Increase Environments & Batch Size (Implemented)
**Changes made:**
- Increased `num_envs`: 64 → 128
- Increased `batch_size`: 512 → 2048

**Expected improvement:**
- GPU usage: 32% → 50-60%
- More parallel simulation on GPU
- Larger batches during training

**Try it:**
```bash
./RUN_SB3_TRAIN.sh
```

### ⚡ Solution 2: Use skrl with Continuous Actions
**GPU usage**: 70-90%

skrl keeps everything on GPU, but has the MultiCategorical bug we encountered.

**Workaround**: Temporarily use continuous actions [-1, 1] without discretization.

### 🚀 Solution 3: Use RSL-RL (Isaac Lab's Native)
**GPU usage**: 80-95%
**Best for**: Maximum performance, continuous actions

RSL-RL is Isaac Lab's native RL library, fully GPU-accelerated.

### 📊 Solution 4: Use rl_games
**GPU usage**: 75-90%
**Supports**: Both continuous and discrete actions well

Another GPU-accelerated option with better discrete action support than skrl.

## Comparison Table

| Framework | GPU Usage | Discrete Actions | Stability | Speed |
|-----------|-----------|------------------|-----------|-------|
| **SB3** | 30-50% | ✅ Excellent | ✅ Very Stable | ⚠️ Moderate |
| **skrl** | 70-90% | ❌ Buggy | ⚠️ Issues | ✅ Fast |
| **RSL-RL** | 80-95% | ⚠️ Limited | ✅ Stable | ✅ Very Fast |
| **rl_games** | 75-90% | ✅ Good | ✅ Stable | ✅ Fast |

## My Recommendations

### For Your Use Case (Discrete Actions):

**Option 1: Stick with SB3 (Current)** ✅ RECOMMENDED
- Pros: Stable, works with discrete, proven
- Cons: Lower GPU usage (but training still works!)
- Improvement: Use 128+ envs, larger batches
- **This is the pragmatic choice**

**Option 2: Try rl_games** ⚡ ALTERNATIVE
- Pros: High GPU usage, supports discrete well
- Cons: Different API, needs setup
- **Good if you need maximum speed**

**Option 3: Go back to Continuous** 🔄 IF NEEDED
- Use continuous actions with skrl
- Remove discretization
- High GPU usage, but changes your control scheme

### For Maximum GPU Utilization:

**Immediate wins with SB3:**
1. ✅ Increase environments: 128-256
2. ✅ Increase batch_size: 2048-4096
3. ✅ Reduce n_steps if memory issues
4. Monitor with `nvidia-smi`

**Expected results:**
- 128 envs + 2048 batch → ~50-60% GPU
- 256 envs + 4096 batch → ~60-70% GPU

## GPU Usage Monitoring

### Check current usage:
```bash
watch -n 1 nvidia-smi
```

### During training look for:
- **GPU-Util**: Target 50-70% with SB3
- **Memory-Usage**: Should be high (80%+)
- **Power**: Should be near TDP limit

## Important Notes

### Why not 100% GPU?
- SB3 is CPU-bottlenecked by design
- Data transfer overhead
- CPU does environment resets
- 50-70% is actually **good for SB3**

### Is low GPU usage bad?
**NO!** What matters:
- ✅ Training is progressing
- ✅ Episodes completing
- ✅ Rewards improving
- ✅ Fast enough for your needs

GPU utilization alone doesn't determine training quality!

### Real bottleneck:
- **Wall-clock time**: How long to train?
- **Sample efficiency**: Reward per timestep?
- **Stability**: Does it learn reliably?

SB3 wins on stability, even with lower GPU usage.

## Try This Now

### 1. Test with increased settings:
```bash
./RUN_SB3_TRAIN.sh
```

### 2. Monitor GPU:
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### 3. Expected GPU usage:
- Before: ~32%
- After: ~50-60%

### 4. If still want more:

**Increase further in RUN_SB3_TRAIN.sh:**
```bash
--num_envs 256  # More environments
```

**Increase in sb3_ppo_cfg.yaml:**
```yaml
batch_size: 4096  # Larger batches
n_steps: 4096     # More steps per update
```

## Alternative: Quick rl_games Setup

If you want to try rl_games for better GPU usage:

```bash
# I can help you set this up
# rl_games has good discrete action support
# Similar to SB3 but GPU-accelerated
```

Let me know if you want to explore this option!

## Bottom Line

**Current state**: 32% GPU is normal for SB3
**After changes**: Should see 50-60% GPU
**Is this enough?**: Yes, if training progresses well
**Want more?**: Consider rl_games or accept SB3's trade-off

The question is: **Is training fast enough for your needs?**
If yes, GPU % doesn't matter. If no, we can explore rl_games.
