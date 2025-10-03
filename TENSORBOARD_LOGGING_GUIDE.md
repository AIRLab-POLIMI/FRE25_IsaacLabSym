# TensorBoard Logging Guide - SB3 Training

## Overview
The SB3 training script now includes enhanced logging with custom callbacks that track additional metrics beyond the defaults.

## Metrics Logged to TensorBoard

### Standard SB3 Metrics (Automatic)
- **`rollout/ep_rew_mean`** - Mean episode reward (default SB3)
- **`rollout/ep_len_mean`** - Mean episode length (default SB3)
- **`train/entropy_loss`** - Policy entropy loss
- **`train/policy_gradient_loss`** - Policy gradient loss
- **`train/value_loss`** - Value function loss
- **`train/approx_kl`** - Approximate KL divergence
- **`train/clip_fraction`** - Fraction of clipped samples
- **`train/loss`** - Total loss
- **`train/explained_variance`** - Explained variance of value function
- **`train/learning_rate`** - Current learning rate
- **`time/fps`** - Frames per second
- **`time/total_timesteps`** - Total timesteps completed

### Enhanced Custom Metrics (Added by Callbacks)

#### Episode Reward Statistics
- **`rollout/ep_rew_max`** - Maximum reward in recent episodes
- **`rollout/ep_rew_min`** - Minimum reward in recent episodes
- **`rollout/ep_rew_std`** - Standard deviation of episode rewards
- **`rollout/max_reward_ever`** ⭐ - **All-time maximum reward achieved**

#### Episode Length Statistics
- **`rollout/ep_len_mean`** - Mean episode length
- **`rollout/ep_len_std`** - Standard deviation of episode lengths

#### Detailed Performance Stats (Every 1000 steps)
- **`stats/reward_p25`** - 25th percentile of recent rewards
- **`stats/reward_p50`** - Median reward (50th percentile)
- **`stats/reward_p75`** - 75th percentile of recent rewards
- **`stats/reward_p90`** - 90th percentile of recent rewards
- **`stats/recent_reward_mean`** - Mean of last 10 episodes
- **`stats/best_mean_reward`** - Best rolling mean reward achieved

#### Progress Tracking (Every 10000 steps)
- Prints to console:
  - Current timesteps
  - Mean, max, min rewards
  - Best mean reward so far

## Viewing in TensorBoard

### Start TensorBoard
```bash
tensorboard --logdir logs/sb3/Fre25-Isaaclabsym-Direct-v0
```

Or use the provided script:
```bash
./RUN_TENSORBOARD.sh
```

### Navigate to Metrics

1. **SCALARS Tab** - View all metrics over time
2. **Custom Scalars** - Create custom layouts

### Recommended Views

#### Training Progress
- Plot: `rollout/ep_rew_mean`, `rollout/max_reward_ever`
- Shows: Learning curve and peak performance

#### Reward Distribution
- Plot: `rollout/ep_rew_min`, `rollout/ep_rew_mean`, `rollout/ep_rew_max`
- Shows: Variance in episode outcomes

#### Performance Percentiles
- Plot: `stats/reward_p25`, `stats/reward_p50`, `stats/reward_p75`, `stats/reward_p90`
- Shows: Distribution of performance

#### Training Stability
- Plot: `rollout/ep_rew_std`, `train/clip_fraction`, `train/approx_kl`
- Shows: Learning stability

#### Loss Curves
- Plot: `train/policy_gradient_loss`, `train/value_loss`, `train/entropy_loss`
- Shows: Optimization progress

## Key Metrics to Watch

### 🎯 Success Indicators
1. **`rollout/max_reward_ever`** - Should increase over time
2. **`rollout/ep_rew_mean`** - Should show upward trend
3. **`train/explained_variance`** - Should be close to 1.0
4. **`stats/reward_p90`** - Top performance improving

### ⚠️ Warning Signs
1. **`train/approx_kl`** - Too high (>0.03) may indicate unstable training
2. **`rollout/ep_rew_std`** - Very high indicates inconsistent performance
3. **`train/clip_fraction`** - Too high (>0.5) suggests learning rate too high
4. **`train/entropy_loss`** - Dropping too fast means exploration stopped

### 🔧 Tuning Hints

**If reward not improving:**
- Increase `ent_coef` (exploration) - try 0.02 or 0.05
- Reduce `learning_rate` - try 1e-4
- Check `rollout/ep_rew_max` - if it's high, agent can solve it but not consistently

**If training unstable:**
- Reduce `learning_rate` - try 1e-4 or 5e-5
- Reduce `clip_range` - try 0.1
- Check `train/approx_kl` - should stay below 0.03

**If no exploration:**
- Increase `ent_coef` - try 0.02, 0.05, or even 0.1
- Check `train/entropy_loss` - should decrease slowly, not drop immediately

## Console Output

The `ProgressCallback` prints every 10,000 steps:
```
Steps: 10000
  Mean reward: 145.32
  Max reward: 890.45
  Min reward: -52.10
----------------------------------------
```

The `EnhancedLoggingCallback` prints when a new max reward is achieved:
```
New max reward: 1205.67
```

## Custom Callback Details

### `EnhancedLoggingCallback`
- Tracks all-time maximum reward
- Logs min/max/std of episode rewards
- Runs every step (low overhead)

### `DetailedStatsCallback`
- Logs percentile statistics
- Runs every 1000 steps
- Tracks last 100 episodes for percentiles

### `ProgressCallback`
- Prints progress to console
- Runs every 10,000 steps
- Tracks best mean reward

## Files
- **Callbacks**: `scripts/sb3/callbacks.py`
- **Training script**: `scripts/sb3/train.py`
- **Logs**: `logs/sb3/Fre25-Isaaclabsym-Direct-v0/<timestamp>/`

## Tips

1. **Compare runs**: TensorBoard can overlay multiple runs
2. **Smoothing**: Use smoothing slider in TensorBoard for clearer trends
3. **Export data**: Download CSV from TensorBoard for custom analysis
4. **Real-time monitoring**: Keep TensorBoard open during training
5. **Checkpoint correlation**: Note timestep of best `max_reward_ever` to load that checkpoint

## Example TensorBoard Commands

```bash
# Basic view
tensorboard --logdir logs/sb3

# Compare multiple tasks
tensorboard --logdir logs/sb3 --port 6006

# Reload faster
tensorboard --logdir logs/sb3 --reload_interval 30
```
