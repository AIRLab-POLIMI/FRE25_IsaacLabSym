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
parser.add_argument("--plotObservations", action="store_true", default=False, help="Plot observations (lidar + other signals).")
parser.add_argument("--plotRewards", action="store_true", default=False, help="Plot discounted cumulative rewards over time.")

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
            print(f"[INFO]    Action space: {env_cfg.action_space} (3 control + {num_hidden_states} hidden)")
            print(f"[INFO]    Observation space: {env_cfg.observation_space} (44 base + {3 + num_hidden_states} past actions)")
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
    num_total_actions = 3 + num_hidden_states  # steering, throttle, step_command + hidden states

    if args_cli.plotHiddenStates and num_total_actions > 0:
        import matplotlib.pyplot as plt

        # Prepare for plotting ALL actions (control + hidden states)
        plt.ion()
        fig, axs = plt.subplots(num_total_actions, 1, figsize=(8, 2 * num_total_actions))
        if num_total_actions == 1:
            axs = [axs]
        lines = []
        data = [[] for _ in range(num_total_actions)]

        # Action titles (updated for 3 control actions)
        action_titles = ["Steering", "Throttle", "Step Command"] + [f"Hidden State {i+1}" for i in range(num_hidden_states)]

        for i in range(num_total_actions):
            line, = axs[i].plot([], [])
            lines.append(line)
            axs[i].set_xlim(0, args_cli.video_length)
            # Step command is binary {0, 1}, others are {-1, 1}
            if i == 2:  # step_command
                axs[i].set_ylim(-0.2, 1.2)
            else:
                axs[i].set_ylim(-1.5, 1.5)
            axs[i].set_ylabel("Action Value")
            axs[i].set_title(action_titles[i])
            axs[i].grid(True, alpha=0.3)
        axs[-1].set_xlabel("Timestep")
        plt.tight_layout()

    # Prepare plotting for observations if requested
    if args_cli.plotObservations:
        import matplotlib.pyplot as plt
        import math

        # Observation structure: steering (1) + lidar (40) + commands (3) + past_actions (1 + num_hidden_states)
        num_lidar_rays = 40
        num_non_lidar_obs = 1 + 3 + (1 + num_hidden_states)  # steering + commands + past_actions

        print(f"[INFO] Setting up observation plotting:")
        print(f"[INFO]   - Lidar rays: {num_lidar_rays}")
        print(f"[INFO]   - Other observations: {num_non_lidar_obs}")

        # Create separate figures for lidar and other observations
        plt.ion()

        # Figure 1: Lidar visualization (polar/spatial plot)
        fig_lidar = plt.figure(figsize=(8, 8))
        ax_lidar = fig_lidar.add_subplot(111)
        ax_lidar.set_aspect('equal')
        ax_lidar.set_title('Lidar Visualization (Robot View)', fontsize=14, fontweight='bold')
        ax_lidar.set_xlabel('X (meters)')
        ax_lidar.set_ylabel('Y (meters)')
        ax_lidar.grid(True, alpha=0.3)

        # Initialize lidar scatter plot
        lidar_scatter = ax_lidar.scatter([], [], c='red', s=50, alpha=0.6, label='Obstacles')
        lidar_line, = ax_lidar.plot([], [], 'r-', alpha=0.3, linewidth=1)

        # Add robot center marker
        ax_lidar.scatter([0], [0], c='blue', s=200, marker='o', label='Robot', zorder=5)
        ax_lidar.legend(loc='upper right')

        # Set fixed limits for lidar plot (assume max range ~10m)
        lidar_max_range = 1.5
        ax_lidar.set_xlim(-lidar_max_range, lidar_max_range)
        ax_lidar.set_ylim(-lidar_max_range, lidar_max_range)

        # Figure 2: Other observations (time series)
        fig_obs = plt.figure(figsize=(10, 8))
        num_other_plots = num_non_lidar_obs
        axs_obs = fig_obs.subplots(num_other_plots, 1)
        if num_other_plots == 1:
            axs_obs = [axs_obs]

        lines_obs = []
        data_obs = [[] for _ in range(num_other_plots)]

        # Observation titles
        obs_titles = ["Steering Angle"] + \
                     [f"Command {i+1}" for i in range(3)] + \
                     ["Past: Step Command"] + \
                     [f"Past: Hidden State {i+1}" for i in range(num_hidden_states)]

        for i in range(num_other_plots):
            line, = axs_obs[i].plot([], [])
            lines_obs.append(line)
            axs_obs[i].set_xlim(0, args_cli.video_length)
            # axs_obs[i].set_ylim(-1.5, 1.5)  # Most observations are normalized
            axs_obs[i].set_ylabel("Value")
            axs_obs[i].set_title(obs_titles[i])
            axs_obs[i].grid(True, alpha=0.3)
        axs_obs[-1].set_xlabel("Timestep")
        fig_obs.tight_layout()

        # Precompute lidar angles (evenly distributed around 360 degrees)
        lidar_angles = np.linspace(0, 2 * math.pi, num_lidar_rays, endpoint=False)

    # Prepare plotting for rewards if requested
    if args_cli.plotRewards:
        import matplotlib.pyplot as plt

        print(f"[INFO] Setting up reward plotting with gamma={agent_cfg.get('gamma', 0.99)}")

        # Get gamma from agent config
        gamma = agent_cfg.get('gamma', 0.99)

        # Create figure for reward visualization
        plt.ion()
        fig_reward = plt.figure(figsize=(12, 6))

        # Create two subplots: instant reward and discounted cumulative reward
        ax_instant = fig_reward.add_subplot(2, 1, 1)
        ax_cumulative = fig_reward.add_subplot(2, 1, 2)

        # Setup instant reward plot
        ax_instant.set_title('Instant Reward per Step', fontsize=12, fontweight='bold')
        ax_instant.set_ylabel('Reward')
        ax_instant.grid(True, alpha=0.3)
        line_instant, = ax_instant.plot([], [], 'b-', linewidth=1.5, label='Instant Reward')
        ax_instant.legend(loc='upper right')
        ax_instant.set_xlim(0, args_cli.video_length)

        # Setup cumulative reward plot
        ax_cumulative.set_title(f'Discounted Cumulative Reward (γ={gamma})', fontsize=12, fontweight='bold')
        ax_cumulative.set_xlabel('Timestep')
        ax_cumulative.set_ylabel('Cumulative Reward')
        ax_cumulative.grid(True, alpha=0.3)
        line_cumulative, = ax_cumulative.plot([], [], 'g-', linewidth=2, label='Cumulative Reward')
        ax_cumulative.legend(loc='upper right')
        ax_cumulative.set_xlim(0, args_cli.video_length)

        fig_reward.tight_layout()

        # Initialize reward tracking
        instant_rewards = []
        cumulative_reward = 0.0
        cumulative_rewards = []
        reward_step = 0

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
            obs, rewards, dones, infos = env.step(actions)

            if args_cli.plotHiddenStates and num_total_actions > 0:
                # Extract past actions (last num_total_actions observations)
                past_actions = actions[0]
                past_actions[:2] -= 1

                # Update plot data for each action
                for i in range(num_total_actions):
                    data[i].append(past_actions[i])
                    lines[i].set_data(range(len(data[i])), data[i])
                    axs[i].set_xlim(0, max(10, len(data[i])))
                plt.pause(0.001)
                plt.draw()

            if args_cli.plotObservations:
                # Extract observations from first environment
                # obs is a dict with 'policy' key containing the observation vector
                if isinstance(obs, dict):
                    obs_vector = obs['policy'][0]  # First environment
                else:
                    obs_vector = obs[0]  # First environment

                # Convert to numpy if it's a torch tensor
                if torch.is_tensor(obs_vector):
                    obs_vector = obs_vector.cpu().numpy()

                # Parse observation structure: steering (1) + lidar (40) + commands (3) + past_actions (1 + num_hidden)
                idx = 0
                steering_obs = obs_vector[idx]
                idx += 1

                lidar_distances = obs_vector[idx:idx + num_lidar_rays]
                idx += num_lidar_rays

                command_obs = obs_vector[idx:idx + 3]
                idx += 3

                past_action_obs = obs_vector[idx:]

                # Update lidar visualization (convert polar to cartesian)
                lidar_x = lidar_distances * np.cos(lidar_angles)
                lidar_y = lidar_distances * np.sin(lidar_angles)

                # Update scatter plot
                lidar_scatter.set_offsets(np.c_[lidar_x, lidar_y])

                # Update line plot (connect the points)
                # Close the loop by appending first point at the end
                lidar_x_closed = np.append(lidar_x, lidar_x[0])
                lidar_y_closed = np.append(lidar_y, lidar_y[0])
                lidar_line.set_data(lidar_x_closed, lidar_y_closed)

                # Update other observations time series
                other_obs = np.concatenate([[steering_obs], command_obs, past_action_obs])
                for i in range(len(other_obs)):
                    data_obs[i].append(other_obs[i])
                    lines_obs[i].set_data(range(len(data_obs[i])), data_obs[i])
                    axs_obs[i].set_xlim(0, max(10, len(data_obs[i])))

                    # Dynamic y-scale: adjust to data range with 10% padding
                    if len(data_obs[i]) > 0:
                        y_min = min(data_obs[i])
                        y_max = max(data_obs[i])
                        y_range = y_max - y_min
                        if y_range > 0:
                            padding = y_range * 0.1
                            axs_obs[i].set_ylim(y_min - padding, y_max + padding)
                        else:
                            # If all values are the same, use a default range
                            axs_obs[i].set_ylim(y_min - 0.5, y_max + 0.5)

                # Refresh plots
                fig_lidar.canvas.draw_idle()
                fig_lidar.canvas.flush_events()
                fig_obs.canvas.draw_idle()
                fig_obs.canvas.flush_events()

            if args_cli.plotRewards:
                # Extract reward from first environment
                if isinstance(rewards, np.ndarray):
                    current_reward = float(rewards[0])
                else:
                    current_reward = float(rewards)

                # Update instant reward
                instant_rewards.append(current_reward)

                # Update discounted cumulative reward
                # G_t = r_t + γ * G_{t-1}
                cumulative_reward = current_reward + gamma * cumulative_reward
                cumulative_rewards.append(cumulative_reward)

                # Update plots
                reward_step += 1
                timesteps_reward = list(range(len(instant_rewards)))

                # Update instant reward plot
                line_instant.set_data(timesteps_reward, instant_rewards)
                ax_instant.set_xlim(0, max(10, len(instant_rewards)))
                if len(instant_rewards) > 0:
                    y_min = min(instant_rewards)
                    y_max = max(instant_rewards)
                    y_range = y_max - y_min
                    if y_range > 0:
                        padding = y_range * 0.1
                        ax_instant.set_ylim(y_min - padding, y_max + padding)
                    else:
                        ax_instant.set_ylim(y_min - 1, y_max + 1)

                # Update cumulative reward plot
                line_cumulative.set_data(timesteps_reward, cumulative_rewards)
                ax_cumulative.set_xlim(0, max(10, len(cumulative_rewards)))
                if len(cumulative_rewards) > 0:
                    y_min = min(cumulative_rewards)
                    y_max = max(cumulative_rewards)
                    y_range = y_max - y_min
                    if y_range > 0:
                        padding = y_range * 0.1
                        ax_cumulative.set_ylim(y_min - padding, y_max + padding)
                    else:
                        ax_cumulative.set_ylim(y_min - 1, y_max + 1)

                # Refresh plot
                fig_reward.canvas.draw_idle()
                fig_reward.canvas.flush_events()

                # Check for environment reset and clear accumulators/plots
            if dones[0]:  # First environment done
                print("[INFO] Environment reset detected - clearing plots and accumulators")

                # # Reset hidden states plot data
                # if args_cli.plotHiddenStates and num_total_actions > 0:
                #     for i in range(num_total_actions):
                #         data[i].clear()
                #         lines[i].set_data([], [])

                # # Reset observations plot data
                # if args_cli.plotObservations:
                #     for i in range(num_other_plots):
                #         data_obs[i].clear()
                #         lines_obs[i].set_data([], [])

                # Reset rewards plot data and cumulative reward
                if args_cli.plotRewards:
                    instant_rewards.clear()
                    cumulative_rewards.clear()
                    cumulative_reward = 0.0
                    reward_step = 0
                    # line_instant.set_data([], [])
                    # line_cumulative.set_data([], [])

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
