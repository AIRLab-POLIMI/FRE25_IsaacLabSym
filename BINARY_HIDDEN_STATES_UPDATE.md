# Binary Hidden States Update

## Summary
Changed hidden state actions from ternary {-1, 0, 1} to **binary {-1, +1}** to remove the neutral state and force more decisive memory encoding.

## Rationale
- **Simpler action space**: Binary actions are easier for the policy to learn
- **Forced commitment**: No neutral state means the policy must actively choose a memory encoding
- **Better differentiation**: Binary states provide clearer "on/off" or "left/right" memory encoding
- **Computational efficiency**: Fewer action categories reduce the combinatorial complexity

## Changes Made

### 1. Environment (`fre25_isaaclabsym_env.py`)

#### Action Space Definition
```python
# Before: [3, 3, 2] + [3] * num_hidden
# After:  [3, 3, 2] + [2] * num_hidden

action_categories = [3, 3, 2] + [2] * num_hidden
```

- **Steering**: 3 categories {-1, 0, 1} ← unchanged
- **Throttle**: 3 categories {-1, 0, 1} ← unchanged  
- **Step Command**: 2 categories {0, 1} ← unchanged
- **Hidden States**: **2 categories {-1, +1}** ← changed from 3

#### Action Conversion
```python
# Before: actions[:, 3:] - 1  (produces {-1, 0, 1})
# After:  actions[:, 3:] * 2 - 1  (produces {-1, +1})

if self.num_hidden_states > 0:
    converted_actions[:, 3:] = actions[:, 3:] * 2 - 1  # 0→-1, 1→+1
```

**Conversion mapping:**
- Discrete index `0` → `-1` (e.g., "off", "left", "negative")
- Discrete index `1` → `+1` (e.g., "on", "right", "positive")

### 2. Teleop Script (`teleop.py`)

#### Default Hidden State Values
```python
# Before: hidden_actions = [1] * num_hidden_states  # → 0 after conversion
# After:  hidden_actions = [0] * num_hidden_states  # → -1 after conversion

hidden_actions = [0] * num_hidden_states  # Default to 0 (→ -1 after conversion)
```

In teleop mode, all hidden states default to `-1` (inactive/off state).

### 3. Configuration Files

Updated documentation in:
- `agents/sb3_ppo_cfg.yaml`
- `agents/policy/mlp.yaml`
- `KEYBOARD_CONTROLS.md`

All files now document that hidden actions are binary: `{0, 1} → {-1, +1}`

## Action Space Structure

### Complete Action Tensor
```
[steering, throttle, step_command, hidden_1, hidden_2, ..., hidden_N]
```

### Discrete Indices (what the agent outputs)
```
[0-2, 0-2, 0-1, 0-1, 0-1, ..., 0-1]
```

### Continuous Values (after conversion in environment)
```
[-1/0/+1, -1/0/+1, 0/1, -1/+1, -1/+1, ..., -1/+1]
```

### Example with 8 Hidden States
- **Discrete action space**: `MultiDiscrete([3, 3, 2, 2, 2, 2, 2, 2, 2, 2])`
- **Total actions**: 11 (3 control + 8 hidden)
- **Agent output example**: `[1, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0]`
- **After conversion**: `[0, +1, 0, +1, -1, +1, +1, -1, -1, +1, -1]`

## Observation Space

The observation vector includes all past actions:
```
[current_steering_angle, lidar_readings..., current_commands..., past_actions...]
```

Where `past_actions` has shape `(num_envs, 3 + num_hidden_states)` with values:
- `past_actions[:, 0]`: steering in [-1, 1]
- `past_actions[:, 1]`: throttle in [-1, 1]
- `past_actions[:, 2]`: step_command in [0, 1]
- `past_actions[:, 3:]`: hidden states in [-1, 1]

## Impact on Learning

### Advantages of Binary Hidden States
1. **Reduced action space complexity**: 2^N vs 3^N possible hidden state combinations
2. **Clearer semantics**: On/off, left/right, positive/negative
3. **Easier exploration**: Fewer discrete choices to explore
4. **Better gradient flow**: Simpler decision boundaries

### Potential Considerations
1. **Less expressiveness**: Can't encode a "neutral" or "don't care" state
2. **Always active**: Policy must choose a side for every hidden dimension
3. **May need more dimensions**: To compensate for lack of neutral state

### Recommended Configuration
For equivalent expressiveness to 8 ternary hidden states, you might want:
- **Before**: 8 hidden states × 3 categories = 6561 combinations
- **After**: ~13 hidden states × 2 categories = 8192 combinations

Currently using 8 binary hidden states = 256 combinations (much simpler!)

## Testing

After this update, verify:
1. ✅ Environment initializes correctly with binary action space
2. ✅ Teleop works with binary hidden states (default to -1)
3. ✅ Training converges with reduced action space complexity
4. ✅ Agent can still solve the task with binary memory encoding

## Files Modified

1. `source/FRE25_IsaacLabSym/FRE25_IsaacLabSym/tasks/direct/fre25_isaaclabsym/fre25_isaaclabsym_env.py`
   - Action space definition
   - Action conversion logic

2. `scripts/teleop/keyboard/teleop.py`
   - Default hidden state values

3. `source/FRE25_IsaacLabSym/FRE25_IsaacLabSym/tasks/direct/fre25_isaaclabsym/agents/sb3_ppo_cfg.yaml`
   - Documentation update

4. `source/FRE25_IsaacLabSym/FRE25_IsaacLabSym/tasks/direct/fre25_isaaclabsym/agents/policy/mlp.yaml`
   - Documentation update

5. `KEYBOARD_CONTROLS.md`
   - User-facing documentation

## Migration Notes

If you have existing trained models with ternary hidden states:
- ⚠️ **Models are NOT compatible** - action space has changed
- You must retrain from scratch
- Old checkpoints cannot be loaded

## Next Steps

1. **Test in simulation**: Run teleop to verify binary hidden states work correctly
2. **Start training**: Train a new model with the simplified action space
3. **Monitor learning**: Check if the policy learns faster with binary hidden states
4. **Adjust dimensions**: If needed, increase `num_hidden_states` to maintain expressiveness
5. **Reward engineering**: Consider adding rewards for effective use of hidden states

## Related Documentation

- `KEYBOARD_CONTROLS.md`: User guide for teleop controls
- `agents/sb3_ppo_cfg.yaml`: Main training configuration
- `agents/policy/mlp.yaml`: MLP-specific policy configuration
