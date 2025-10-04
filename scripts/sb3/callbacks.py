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

            # Extract custom episode statistics from aggregate sums and counts
            # Environment now provides only sum and count (efficient, no per-episode lists)
            # We compute accurate means from these aggregates
            total_episodes = 0
            waypoints_sum = 0.0
            collisions_sum = 0.0
            oob_sum = 0.0
            timeouts_sum = 0.0

            waypointsAverge = []

            for ep_info in self.model.ep_info_buffer:
                # Extract aggregate sums and count from each buffer entry
                if 'episode_count' in ep_info:
                    total_episodes += ep_info['episode_count']
                    waypoints_sum += ep_info.get('waypoints_reached_sum', 0.0)
                    collisions_sum += ep_info.get('plant_collisions_sum', 0.0)
                    oob_sum += ep_info.get('out_of_bounds_sum', 0.0)
                    timeouts_sum += ep_info.get('timeouts_sum', 0.0)
                    waypointsAverge.append(ep_info.get('waypoints_reached_sum', 0.0) / ep_info['episode_count'] if ep_info['episode_count'] > 0 else 0.0)

            # Debug: print what we extracted
            if self.verbose > 0 and self.n_calls % 1000 == 0 and total_episodes > 0:
                print(f"\n[DEBUG] Extracted stats from {total_episodes} episodes")
                print(f"  Total episodes in buffer: {len(self.model.ep_info_buffer)}")
                print(f"  Waypoints sum: {waypoints_sum:.1f}, mean: {waypoints_sum/total_episodes:.2f}")
                print(f"  Collisions sum: {collisions_sum:.1f}, mean: {collisions_sum/total_episodes:.2f}")
                print(f"  Out of bounds sum: {oob_sum:.1f}, mean: {oob_sum/total_episodes:.2f}")
                print(f"  Timeouts sum: {timeouts_sum:.1f}, mean: {timeouts_sum/total_episodes:.2f}")

            if len(rewards) > 0:
                # Log standard statistics
                self.logger.record("rollout/ep_rew_max", np.max(rewards))
                self.logger.record("rollout/ep_rew_min", np.min(rewards))
                self.logger.record("rollout/ep_rew_std", np.std(rewards))
                self.logger.record("rollout/ep_len_mean", np.mean(lengths))
                self.logger.record("rollout/ep_len_std", np.std(lengths))

                # Log custom episode statistics from aggregate sums
                # These are accurate per-episode means and rates computed from aggregate data
                if total_episodes > 0:
                    waypoints_mean = waypoints_sum / total_episodes
                    collisions_mean = collisions_sum / total_episodes
                    oob_mean = oob_sum / total_episodes
                    timeouts_mean = timeouts_sum / total_episodes
                    waypoints_std = np.std(waypointsAverge) if waypointsAverge else 0.0
                    waypopints_max = np.max(waypointsAverge) if waypointsAverge else 0.0

                    self.logger.record("episode/waypoints_reached_max", waypopints_max)
                    self.logger.record("episode/waypoints_reached_mean", waypoints_mean)
                    self.logger.record("episode/waypoints_reached_std", waypoints_std)

                    self.logger.record("episode/rate_plant_collisions", collisions_mean)
                    # self.logger.record("episode/plant_collisions_total", collisions_sum)

                    self.logger.record("episode/rate_out_of_bounds", oob_mean)
                    # self.logger.record("episode/out_of_bounds_total", oob_sum)

                    self.logger.record("episode/rate_timeouts", timeouts_mean)
                    # self.logger.record("episode/timeouts_total", timeouts_sum)

                    # Also log rates (as percentages of total episodes)
                    # self.logger.record("episode/collision_rate", (collisions_sum / total_episodes) * 100)
                    # self.logger.record("episode/oob_rate", (oob_sum / total_episodes) * 100)
                    # self.logger.record("episode/timeout_rate", (timeouts_sum / total_episodes) * 100)

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
