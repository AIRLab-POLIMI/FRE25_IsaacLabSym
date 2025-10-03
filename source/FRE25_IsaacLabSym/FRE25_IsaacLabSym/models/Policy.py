import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin, MultiCategoricalMixin


# Define custom policy model with configurable architecture
class CustomPolicy(GaussianMixin, Model):
    """Custom policy network with configurable architecture for continuous action spaces."""

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20, max_log_std=2,
                 initial_log_std=0, reduction="sum",
                 hidden_sizes=[512, 256, 256, 128, 128],
                 activation="relu"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        # Map activation string to activation function
        activation_map = {
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'selu': nn.SELU
        }
        activation_fn = activation_map.get(activation.lower(), nn.ReLU)

        # Build network layers
        layers = []
        input_size = self.num_observations

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation_fn())
            input_size = hidden_size

        # Output layer for mean
        layers.append(nn.Linear(input_size, self.num_actions))
        layers.append(nn.Tanh())  # Assuming actions are in range [-1, 1]

        self.net = nn.Sequential(*layers)

        # Log standard deviation (learned parameter)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * initial_log_std)

    def compute(self, inputs, role=""):
        # Compute mean from network
        mean = self.net(inputs["states"])
        # Return mean and log_std
        return mean, self.log_std_parameter, {}


# Define custom value model
class CustomValue(DeterministicMixin, Model):
    """Custom value network with configurable architecture."""

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False,
                 hidden_sizes=[512, 256, 256, 128, 128],
                 activation="relu"):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # Map activation string to activation function
        activation_map = {
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'selu': nn.SELU
        }
        activation_fn = activation_map.get(activation.lower(), nn.ReLU)

        # Build network layers
        layers = []
        input_size = self.num_observations

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation_fn())
            input_size = hidden_size

        # Output layer for value (single scalar)
        layers.append(nn.Linear(input_size, 1))

        self.net = nn.Sequential(*layers)

    def compute(self, inputs, role=""):
        # Compute value from network
        return self.net(inputs["states"]), {}


# Define discrete policy model for multi-categorical actions
class DiscretePolicy(MultiCategoricalMixin, Model):
    """Custom discrete policy network for multi-categorical action spaces.

    Outputs logits for 6 discrete actions, each with 3 categories {-1, 0, 1}.
    Total output: 18 logits (6 actions × 3 categories).

    Includes workaround for skrl PPO entropy computation bug with MultiCategorical actions.
    """

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False,
                 unnormalized_log_prob=True,
                 reduction="sum",  # Keep "sum" for log_prob storage
                 hidden_sizes=[512, 256, 256, 128, 128],
                 activation="relu",
                 num_actions=6,
                 num_categories=3):
        Model.__init__(self, observation_space, action_space, device)
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob, reduction)

        # Map activation string to activation function
        activation_map = {
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'selu': nn.SELU
        }
        activation_fn = activation_map.get(activation.lower(), nn.ReLU)

        # Build network layers
        layers = []
        input_size = self.num_observations

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation_fn())
            input_size = hidden_size

        # Output layer - use self.num_actions (set by Model from action_space)
        # For MultiDiscrete([3]*6), self.num_actions will be sum([3]*6) = 18
        layers.append(nn.Linear(input_size, self.num_actions))

        self.net = nn.Sequential(*layers)

    def compute(self, inputs, role):
        # Simple forward pass - mixin handles the rest
        return self.net(inputs["states"]), {}

    def get_entropy(self, role=""):
        """Override entropy computation to handle skrl PPO bug.

        Computes entropy manually to avoid the 0-d tensor iteration issue in skrl's PPO.
        The issue occurs because skrl tries to iterate over entropy when it's a 0-d tensor
        with reduction="sum".
        """
        if self._distribution is None:
            return torch.tensor(0.0, device=self.device)

        # Compute entropy per action dimension (no reduction)
        # For MultiDiscrete, entropy() returns [batch_size, num_action_dims]
        entropy_per_action = self._distribution.entropy()

        # Sum across action dimensions, then mean across batch
        # This gives us a scalar value that won't cause iteration issues
        if entropy_per_action.dim() > 0:
            # Sum across action dimensions (dim=-1 if 2D), then mean across batch
            if entropy_per_action.dim() > 1:
                entropy = entropy_per_action.sum(dim=-1).mean()
            else:
                entropy = entropy_per_action.mean()
        else:
            # Fallback if somehow we get a 0-d tensor
            entropy = entropy_per_action

        return entropy
