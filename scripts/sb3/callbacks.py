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
    - Custom episode statistics (waypoints, collisions, etc.)
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

            # DEBUG: Print what keys are available in episode info
            if self.verbose > 0 and self.n_calls % 1000 == 0:
                sample_ep = self.model.ep_info_buffer[0]
                print(f"\n[DEBUG] Episode info keys: {list(sample_ep.keys())}")
                print(f"[DEBUG] Episode info sample: {sample_ep}")

            # Extract custom episode statistics if available
            waypoints_reached = []
            plant_collisions = []
            out_of_bounds = []
            timeouts = []

            for ep_info in self.model.ep_info_buffer:
                # Extract from extras["log"] - these are already computed as means for resetting envs
                if 'waypoints_reached' in ep_info:
                    val = ep_info['waypoints_reached']
                    waypoints_reached.append(float(val) if hasattr(val, 'item') else val)
                if 'plant_collisions' in ep_info:
                    val = ep_info['plant_collisions']
                    plant_collisions.append(float(val) if hasattr(val, 'item') else val)
                if 'out_of_bounds' in ep_info:
                    val = ep_info['out_of_bounds']
                    out_of_bounds.append(float(val) if hasattr(val, 'item') else val)
                if 'timeouts' in ep_info:
                    val = ep_info['timeouts']
                    timeouts.append(float(val) if hasattr(val, 'item') else val)

            # Debug: print what we extracted
            if self.verbose > 0 and self.n_calls % 1000 == 0 and len(waypoints_reached) > 0:
                print(f"\n[DEBUG] Extracted {len(waypoints_reached)} episode stats")
                print(f"  Waypoints: {waypoints_reached[:5]}...")
                print(f"  Collisions: {plant_collisions[:5]}...")
                print(f"  Out of bounds: {out_of_bounds[:5]}...")
                print(f"  Timeouts: {timeouts[:5]}...")

            if len(rewards) > 0:
                # Log standard statistics
                self.logger.record("rollout/ep_rew_max", np.max(rewards))
                self.logger.record("rollout/ep_rew_min", np.min(rewards))
                self.logger.record("rollout/ep_rew_std", np.std(rewards))
                self.logger.record("rollout/ep_len_mean", np.mean(lengths))
                self.logger.record("rollout/ep_len_std", np.std(lengths))

                # Log custom episode statistics
                # Note: These are means computed across environments that reset in each episode
                # So waypoints_reached_mean is the mean of means (average waypoints per episode batch)
                if len(waypoints_reached) > 0:
                    self.logger.record("episode/waypoints_reached_mean", np.mean(waypoints_reached))
                    self.logger.record("episode/waypoints_reached_max", np.max(waypoints_reached))
                    self.logger.record("episode/waypoints_reached_min", np.min(waypoints_reached))
                    self.logger.record("episode/waypoints_reached_std", np.std(waypoints_reached))
                    # Sum tells us total waypoints across recent episode completions
                    self.logger.record("episode/waypoints_reached_sum", np.sum(waypoints_reached))

                if len(plant_collisions) > 0:
                    self.logger.record("episode/plant_collisions_mean", np.mean(plant_collisions))
                    self.logger.record("episode/plant_collisions_max", np.max(plant_collisions))
                    # Sum of means tells us aggregate collision rate
                    self.logger.record("episode/plant_collisions_sum", np.sum(plant_collisions))

                if len(out_of_bounds) > 0:
                    self.logger.record("episode/out_of_bounds_mean", np.mean(out_of_bounds))
                    self.logger.record("episode/out_of_bounds_max", np.max(out_of_bounds))
                    # Sum of means tells us aggregate out-of-bounds rate
                    self.logger.record("episode/out_of_bounds_sum", np.sum(out_of_bounds))

                if len(timeouts) > 0:
                    self.logger.record("episode/timeouts_mean", np.mean(timeouts))
                    self.logger.record("episode/timeouts_max", np.max(timeouts))
                    # Sum tells us how many episodes timed out
                    self.logger.record("episode/timeouts_sum", np.sum(timeouts))
                    # Timeout rate: fraction of episodes that timed out
                    timeout_rate = np.mean([1.0 if t > 0.5 else 0.0 for t in timeouts])
                    self.logger.record("episode/timeout_rate", timeout_rate)

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
