#!/bin/bash
# Quick test of SB3 discrete training (short run to verify everything works)

# Get Isaac Lab path
ISAAC_LAB_PATH="${HOME}/Desktop/PaoloGinefraMultidisciplinaryProject/IsaacLab"

echo "========================================="
echo "Testing SB3 with Discrete Actions"
echo "========================================="
echo ""
echo "This will run a short training test to verify:"
echo "  ✓ Environment loads correctly"
echo "  ✓ MultiDiscrete action space works"
echo "  ✓ SB3 PPO trains without errors"
echo ""
echo "Running for 5000 steps with 16 environments..."
echo ""

# Run short test
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/sb3/train.py \
    --task Fre25-Isaaclabsym-Direct-v0 \
    --num_envs 16 \
    --max_iterations 5 \
    --headless \
    "$@"

echo ""
echo "========================================="
echo "Test complete!"
echo "If this ran without errors, you're ready to train for real:"
echo "  ./RUN_SB3_TRAIN.sh"
echo "========================================="
