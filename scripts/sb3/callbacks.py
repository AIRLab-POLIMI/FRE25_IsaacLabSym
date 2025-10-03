"""Custom callbacks for SB3 training with enhanced logging."""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EnhancedLoggingCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard.

    Logs:
    - Max episode reward
    - Min episode reward
    - Episode length statistics
    - Success rate (if applicable)
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Check if there are any episode infos (completed episodes)
        if len(self.model.ep_info_buffer) > 0:
            # Get all episode rewards from the buffer
            rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]

            if len(rewards) > 0:
                # Log statistics
                self.logger.record("rollout/ep_rew_max", np.max(rewards))
                self.logger.record("rollout/ep_rew_min", np.min(rewards))
                self.logger.record("rollout/ep_rew_std", np.std(rewards))
                self.logger.record("rollout/ep_len_mean", np.mean(lengths))
                self.logger.record("rollout/ep_len_std", np.std(lengths))

                # Track all-time max reward
                if not hasattr(self, 'max_reward_ever'):
                    self.max_reward_ever = -np.inf

                current_max = np.max(rewards)
                if current_max > self.max_reward_ever:
                    self.max_reward_ever = current_max
                    if self.verbose > 0:
                        print(f"New max reward: {self.max_reward_ever:.2f}")

                self.logger.record("rollout/max_reward_ever", self.max_reward_ever)

        return True


class ProgressCallback(BaseCallback):
    """
    Simple callback that prints progress information.
    """

    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get recent episode rewards
            if len(self.model.ep_info_buffer) > 0:
                rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
                mean_reward = np.mean(rewards)

                if self.verbose > 0:
                    print(f"Steps: {self.num_timesteps}")
                    print(f"  Mean reward: {mean_reward:.2f}")
                    print(f"  Max reward: {np.max(rewards):.2f}")
                    print(f"  Min reward: {np.min(rewards):.2f}")
                    print("-" * 40)

                # Track best mean reward
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.logger.record("stats/best_mean_reward", self.best_mean_reward)

        return True
