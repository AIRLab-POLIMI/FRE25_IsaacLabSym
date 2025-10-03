# Copyright (c) 2022-2025, Paolo Ginefra.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script will allow to control the robot with keyboard inputs.
It will use just one environment.

This has been developed for FRE25_IsaacLabSym environment, and it is not meant to work with other environments.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher  # type: ignore

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)

parser.add_argument(
    "--printRewards",
    action="store_true",
    default=False,
    help="Print rewards during training.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Set the number of environments to 1
args_cli.num_envs = 1

# Set the task to FRE25_IsaacLabSym
args_cli.task = "Fre25-Isaaclabsym-Direct-v0"

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # type: ignore # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg  # type: ignore
from isaaclab.devices.device_base import DeviceBase

import FRE25_IsaacLabSym.tasks  # noqa: F401

# Resolve KeyboardManager import: prefer package-relative import, fallback to absolute import when run as a script
from FRE25_IsaacLabSym.utils.KeyboardDevice import KeyboardManager


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    keyboardManager = KeyboardManager()
    keyboardManager.reset()
    totalReward = 0.0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            kinematicCommands, stepCommand = keyboardManager.advance()            # For discrete action space: convert continuous keyboard input to discrete {-1, 0, 1}
            # kinematicCommands contains [steering, throttle] as continuous values
            # We need to discretize them and also handle the 4 hidden state accumulators

            # Convert to discrete actions for all 6 dimensions
            # Dimensions 0-1: steering and throttle (from keyboard)
            # Dimensions 2-5: hidden state accumulators (set to 0 - no change)

            # Steering: negative = left (index 0 → -1), positive = right (index 2 → +1)
            steering_discrete = 1  # Default: no change (index 1 → 0)
            if kinematicCommands[0] > 0.1:
                steering_discrete = 2  # Right (index 2 → +1)
            elif kinematicCommands[0] < -0.1:
                steering_discrete = 0  # Left (index 0 → -1)

            # Throttle: positive = forward (index 2 → +1), negative = backward (index 0 → -1)
            throttle_discrete = 1  # Default: no change (index 1 → 0)
            if kinematicCommands[1] > 0.1:
                throttle_discrete = 2  # Forward (index 2 → +1)
            elif kinematicCommands[1] < -0.1:
                throttle_discrete = 0  # Backward (index 0 → -1)

            # Create discrete action tensor: [steering, throttle, acc1, acc2, acc3, acc4]
            # All indices are in [0, 1, 2] which the environment converts to [-1, 0, +1]
            discrete_actions = torch.tensor(
                [[steering_discrete, throttle_discrete, 1, 1, 1, 1]],  # 1 = no change (0)
                device=env.unwrapped.device,  # type: ignore
                dtype=torch.long
            )

            # apply actions
            observations, rewards, terminations, truncations, infos = env.step(discrete_actions)

            totalReward += rewards.item()

            if args_cli.printRewards:
                print(f"Total reward: {totalReward}")

            if terminations.any() or truncations.any():
                print(f"Episode finished with reward {totalReward}")
                totalReward = 0.0

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()
