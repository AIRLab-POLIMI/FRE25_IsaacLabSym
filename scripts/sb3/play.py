# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3 - FRE25 Custom Version."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

parser.add_argument("--plotHiddenStates", action="store_true", default=False, help="Plot hidden state action values.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch

# Import both PPO variants - will select based on checkpoint
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import FRE25_IsaacLabSym.tasks  # noqa: F401


def find_latest_sb3_checkpoint(log_root_path):
    """Find the latest SB3 checkpoint in the log directory.

    SB3 saves checkpoints as: model_XXXXXX_steps.zip
    This function finds the most recent run and the highest numbered checkpoint.
    """
    import glob
    from pathlib import Path

    # Find all run directories (sorted by modification time, newest first)
    run_dirs = sorted(Path(log_root_path).glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not run_dirs:
        raise ValueError(f"No training runs found in: {log_root_path}")

    # Look for checkpoints in the most recent run first
    for run_dir in run_dirs:
        if run_dir.is_dir():
            # Find all checkpoint files
            checkpoints = list(run_dir.glob("model_*_steps.zip"))
            if checkpoints:
                # Sort by step number (extract from filename)
                def get_steps(path):
                    try:
                        return int(path.stem.split('_')[1])
                    except:
                        return 0

                latest_checkpoint = max(checkpoints, key=get_steps)
                print(f"[INFO] Found checkpoint: {latest_checkpoint}")
                print(f"[INFO] From run: {run_dir.name}")
                return str(latest_checkpoint)

    raise ValueError(f"No checkpoints found in any run directory under: {log_root_path}")


def main():
    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    # directory for logging into
    log_root_path = os.path.join("logs", "sb3", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)

    # checkpoint and log_dir stuff
    if args_cli.checkpoint is None:
        # Find the latest SB3 checkpoint automatically
        checkpoint_path = find_latest_sb3_checkpoint(log_root_path)
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(checkpoint_path)

    # Load saved environment configuration to match training setup
    saved_env_cfg_path = os.path.join(log_dir, "params", "env.pkl")
    if os.path.exists(saved_env_cfg_path):
        print(f"[INFO] Loading saved environment config from: {saved_env_cfg_path}")
        import pickle
        with open(saved_env_cfg_path, "rb") as f:
            saved_env_cfg = pickle.load(f)
        # Update critical settings that affect action/observation spaces
        # Try new terminology first, fall back to old for backward compatibility
        num_hidden_states = getattr(saved_env_cfg, "num_hidden_states",
                                    getattr(saved_env_cfg, "num_hidden_accumulators", 0))
        env_cfg.num_hidden_states = num_hidden_states
        env_cfg.action_space = getattr(saved_env_cfg, "action_space", env_cfg.action_space)
        env_cfg.observation_space = getattr(saved_env_cfg, "observation_space", env_cfg.observation_space)
        if num_hidden_states > 0:
            print(f"[INFO] 🧠 Using {num_hidden_states} hidden states (from saved config)")
            print(f"[INFO]    Action space: {env_cfg.action_space} (2 control + {num_hidden_states} hidden)")
            print(f"[INFO]    Observation space: {env_cfg.observation_space} (44 base + {2 + num_hidden_states} past actions)")
    else:
        print(f"[WARN] ⚠️  No saved environment config found at: {saved_env_cfg_path}")
        print(f"[WARN]    Using default environment config (may mismatch with checkpoint)")

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)

    # Determine algorithm type from config (Hydra composition)
    algorithm_name = agent_cfg.get("algorithm", "PPO")
    policy_type = agent_cfg.get("policy_type", "mlp")

    # Select correct algorithm class
    if algorithm_name == "RecurrentPPO":
        algorithm_class = RecurrentPPO
        print(f"[INFO] Loading RecurrentPPO checkpoint (LSTM policy)")
    else:
        algorithm_class = PPO
        print(f"[INFO] Loading PPO checkpoint (MLP policy)")

    print(f"[INFO] Policy type: {policy_type}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)

    # normalize environment (if needed)
    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # create agent from stable baselines (algorithm determined by config)
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    agent = algorithm_class.load(checkpoint_path, env, print_system_info=True)
    print(f"[INFO] {algorithm_name} agent loaded successfully")

    dt = env.unwrapped.step_dt

    # Prepare plotting for all actions if requested
    num_hidden_states = getattr(env_cfg, "num_hidden_states", 0)
    num_total_actions = 2 + num_hidden_states  # steering, throttle + hidden states

    if args_cli.plotHiddenStates and num_total_actions > 0:
        import matplotlib.pyplot as plt

        # Prepare for plotting ALL actions (control + hidden states)
        plt.ion()
        fig, axs = plt.subplots(num_total_actions, 1, figsize=(8, 2 * num_total_actions))
        if num_total_actions == 1:
            axs = [axs]
        lines = []
        data = [[] for _ in range(num_total_actions)]

        # Action titles
        action_titles = ["Steering Action", "Throttle Action"] + [f"Hidden State {i+1}" for i in range(num_hidden_states)]

        for i in range(num_total_actions):
            line, = axs[i].plot([], [])
            lines.append(line)
            axs[i].set_xlim(0, args_cli.video_length)
            axs[i].set_ylim(-1.5, 1.5)
            axs[i].set_ylabel("Action Value")
            axs[i].set_title(action_titles[i])
            axs[i].grid(True, alpha=0.3)
        axs[-1].set_xlabel("Timestep")
        plt.tight_layout()

    # reset environment
    obs = env.reset()
    timestep = 0
    # simulate environment
    print("[INFO] Starting evaluation...")
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, _ = agent.predict(obs, deterministic=True)
            # env stepping
            obs, _, _, _ = env.step(actions)

        if args_cli.plotHiddenStates and num_total_actions > 0:
            # Extract ALL past actions from observation
            # Observations structure: [steering(1), lidar(40), commands(3), past_actions(2+N)]
            # Past actions are the last (2 + num_hidden_states) elements
            if isinstance(obs, (list, tuple)):
                obs_array = obs[0]  # For VecEnvWrapper, obs is a list with one element
            else:
                obs_array = obs

            # Extract past actions (last num_total_actions observations)
            past_actions = obs_array[0, -num_total_actions:]

            # Update plot data for each action
            for i in range(num_total_actions):
                data[i].append(past_actions[i])
                lines[i].set_data(range(len(data[i])), data[i])
                axs[i].set_xlim(0, max(10, len(data[i])))
            plt.pause(0.001)
            plt.draw()

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
