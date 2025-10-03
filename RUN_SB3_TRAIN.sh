#!/bin/bash
# Train FRE25 agent using Stable-Baselines3 PPO with DISCRETE actions

# Get Isaac Lab path
ISAAC_LAB_PATH="${HOME}/Desktop/PaoloGinefraMultidisciplinaryProject/IsaacLab"

# Run training with SB3
# Increased num_envs to 128 for better GPU utilization
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/sb3/train.py \
    --task Fre25-Isaaclabsym-Direct-v0 \
    --num_envs 128 \
    --headless \
    "$@"
