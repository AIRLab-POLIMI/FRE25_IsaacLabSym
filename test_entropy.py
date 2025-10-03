#!/usr/bin/env python3
"""Test script to verify DiscretePolicy entropy computation works correctly."""

import torch
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete

# Add source to path
import sys
sys.path.append('/home/airlab/Desktop/PaoloGinefraMultidisciplinaryProject/FRE25_IsaacLabSym/source/FRE25_IsaacLabSym')

from FRE25_IsaacLabSym.models import DiscretePolicy


def test_entropy():
    """Test that DiscretePolicy can compute entropy without errors."""

    # Create dummy spaces
    observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,))
    action_space = MultiDiscrete([3, 3, 3, 3, 3, 3])  # 6 actions, 3 categories each

    device = "cpu"

    # Create policy
    print("Creating DiscretePolicy...")
    policy = DiscretePolicy(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        reduction="sum",
        hidden_sizes=[64, 64],
    )

    # Create dummy input
    batch_size = 4
    dummy_states = torch.randn(batch_size, 10)

    # Forward pass
    print("Running forward pass...")
    policy.act({"states": dummy_states}, role="policy")

    # Test entropy computation
    print("Computing entropy...")
    try:
        entropy = policy.get_entropy(role="policy")
        print(f"✓ Entropy computed successfully: {entropy.item():.4f}")
        print(f"✓ Entropy shape: {entropy.shape}")
        print(f"✓ Entropy is scalar: {entropy.dim() == 0}")

        # Verify it's a valid scalar
        assert entropy.dim() == 0, f"Entropy should be 0-d tensor, got {entropy.dim()}-d"
        assert not torch.isnan(entropy), "Entropy should not be NaN"
        assert entropy.item() >= 0, f"Entropy should be non-negative, got {entropy.item()}"

        print("\n✅ All entropy tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Entropy computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_entropy()
    sys.exit(0 if success else 1)
