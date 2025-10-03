#!/bin/bash
# Play/evaluate trained FRE25 agent using Stable-Baselines3

# Get Isaac Lab path
ISAAC_LAB_PATH="${HOME}/Desktop/PaoloGinefraMultidisciplinaryProject/IsaacLab"

# Run play with SB3
# Uses --use_last_checkpoint to automatically find the latest checkpoint
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/sb3/play.py \
    --task Fre25-Isaaclabsym-Direct-v0 \
    --num_envs 1 \
    --use_last_checkpoint \
    "$@"
