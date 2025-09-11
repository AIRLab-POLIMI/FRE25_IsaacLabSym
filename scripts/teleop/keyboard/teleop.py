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
            kinematicCommands, stepCommand = keyboardManager.advance()

            # sample actions from -1 to 1
            actions = env.action_space.sample()
            continousActions = torch.tensor([kinematicCommands], device=env.unwrapped.device)  # type: ignore
            discreteActions = torch.tensor([stepCommand], device=env.unwrapped.device)[:, None]  # type: ignore
            # actions = torch.cat([continousActions, discreteActions], dim=1)
            # apply actions
            observations, rewards, terminations, truncations, infos = env.step(
                continousActions
            )

            totalReward += rewards.item()

            # print(f"Total reward: {totalReward}")

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
