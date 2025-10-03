import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


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
