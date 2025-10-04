#!/bin/bash
# Train FRE25 agent using Stable-Baselines3 PPO with DISCRETE actions
# Supports both MLP and LSTM policies via Hydra configuration

# Get Isaac Lab path
ISAAC_LAB_PATH="${HOME}/Desktop/PaoloGinefraMultidisciplinaryProject/IsaacLab"

echo "========================================="
echo "FRE25 SB3 TRAINING (HYDRA CONFIG)"
echo "========================================="
echo ""
echo "Policy Configuration:"
echo "  Edit sb3_ppo_cfg.yaml to switch between MLP and LSTM"
echo "  defaults:"
echo "    - policy: mlp   # or 'lstm' for temporal memory"
echo ""
echo "Hidden accumulator options (override from command line):"
echo "  Default:        (uses mlp.yaml default: 8 accumulators)"
echo "  No memory:      num_hidden_accumulators=0"
echo "  Light memory:   num_hidden_accumulators=4"
echo "  Medium memory:  num_hidden_accumulators=8"
echo "  Rich memory:    num_hidden_accumulators=16"
echo ""
echo "Examples:"
echo "  Default (8):    ./RUN_SB3_TRAIN.sh"
echo "  No memory:      ./RUN_SB3_TRAIN.sh num_hidden_accumulators=0"
echo "  Rich memory:    ./RUN_SB3_TRAIN.sh num_hidden_accumulators=16"
echo ""
echo "Current settings: 128 environments"
echo "Override with: --num_envs <N>"
echo ""

# Run training with SB3
# Hydra args are passed after script args (after --)
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/sb3/train.py \
    --task Fre25-Isaaclabsym-Direct-v0 \
    --num_envs 128 \
    --headless \
    "$@"
