"""Custom PPO agent with fix for MultiCategorical action space tracking."""

from skrl.agents.torch.ppo import PPO
import torch


class PPO_MultiCategorical(PPO):
    """PPO agent with fixed tracking for MultiCategorical action spaces.

    Skips standard deviation tracking which is only applicable to Gaussian (continuous) policies.
    """

    def _update(self, timestep: int, timesteps: int):
        """Update method with fixed tracking for discrete actions."""

        # Call parent update but catch the stddev tracking error
        try:
            return super()._update(timestep, timesteps)
        except (AttributeError, RuntimeError) as e:
            # If error is related to stddev (MultiCategorical doesn't have it), patch and retry
            if "stddev" in str(e) or "0-d tensor" in str(e):
                # Monkey-patch the track_data call to skip stddev tracking
                original_track_data = self.track_data

                def patched_track_data(tag, value, **kwargs):
                    # Skip standard deviation tracking for discrete actions
                    if "Standard deviation" not in tag:
                        original_track_data(tag, value, **kwargs)

                self.track_data = patched_track_data
                result = super()._update(timestep, timesteps)
                self.track_data = original_track_data  # Restore
                return result
            else:
                raise

    def post_interaction(self, timestep: int, timesteps: int):
        """Override to patch tracking before calling parent."""

        # Temporarily replace track_data to skip stddev tracking
        original_track_data = self.track_data

        def patched_track_data(tag, value, **kwargs):
            # Skip standard deviation tracking for discrete actions
            if "Standard deviation" not in tag:
                try:
                    original_track_data(tag, value, **kwargs)
                except (AttributeError, RuntimeError, TypeError) as e:
                    # Silently skip if value causes issues (like 0-d tensor iteration)
                    if "0-d tensor" in str(e) or "stddev" in str(e):
                        pass
                    else:
                        raise

        self.track_data = patched_track_data

        try:
            result = super().post_interaction(timestep, timesteps)
        finally:
            self.track_data = original_track_data  # Always restore

        return result
