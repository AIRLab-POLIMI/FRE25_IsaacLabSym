# Keyboard Teleop Controls

## Overview
The teleop script allows you to manually control the robot using keyboard inputs. This is useful for debugging and understanding the robot's behavior.

## Key Bindings

| Key | Action | Description |
|-----|--------|-------------|
| **W** | Throttle Forward | Increases forward velocity |
| **S** | Throttle Backward | Increases backward velocity |
| **A** | Steer Left | Turns the robot left |
| **D** | Steer Right | Turns the robot right |
| **E** | Step Command Buffer | **Hold down** to set step_command to 1 (advances command buffer on rising edge) |
| **L** | Reset | Resets the keyboard controller state |

## Action Space Details

The keyboard inputs are converted to discrete actions that match the agent's action space:

### Discrete Action Format
```
[steering, throttle, step_command, hidden_state_1, ..., hidden_state_N]
```

- **Steering**: Discrete values {0, 1, 2} → {-1, 0, +1}
  - 0 (left): Press A
  - 1 (no change): Default
  - 2 (right): Press D

- **Throttle**: Discrete values {0, 1, 2} → {-1, 0, +1}
  - 0 (backward): Press S
  - 1 (no change): Default
  - 2 (forward): Press W

- **Step Command**: Binary {0, 1}
  - 0: E key not held
  - 1: E key held down
  - The environment uses rising edge detection (transition from 0→1) to advance the command buffer

- **Hidden States**: All default to 0 (→ -1) in teleop mode
  - Binary actions: {0, 1} → {-1, +1} (no neutral state)
  - The number of hidden states is read dynamically from the environment config (`num_hidden_states`)
  - These are automatically populated with default values (0 → -1) during teleop

## How the Step Command Works

1. **Hold 'E' key**: Sets `step_command = 1`
2. **Release 'E' key**: Sets `step_command = 0`
3. **Rising Edge Detection**: The environment detects when step_command transitions from 0→1 and advances the command buffer
4. **This means**: To step the command buffer once, press and release 'E'. Holding 'E' continuously will only step once (on the initial press).

## Running Teleop

Execute the teleop script using:
```bash
./RUN_TELEOP.sh
```

Or manually:
```bash
python scripts/teleop/keyboard/teleop.py --num_envs 1
```

## Tips

- The robot expects commands in a "steel crab" kinematic configuration (can move in any direction without changing bearing)
- Press **L** to reset if the controller gets into a strange state
- The `--printRewards` flag can be added to see real-time reward feedback
- Only one environment is used for teleop (forced to `num_envs=1`)

## Implementation Files

- **KeyboardDevice**: `source/FRE25_IsaacLabSym/FRE25_IsaacLabSym/utils/KeyboardDevice.py`
  - Handles keyboard event listening and state management
  - Manages the step_command boolean based on 'E' key state

- **Teleop Script**: `scripts/teleop/keyboard/teleop.py`
  - Converts keyboard inputs to discrete actions
  - Interfaces with the gym environment

## Related Documentation

See `REFACTOR_SUMMARY.md` for details on how the step_command action integrates with the environment and agent architecture.
