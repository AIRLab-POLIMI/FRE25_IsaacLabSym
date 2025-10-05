# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3 - FRE25 Custom Version."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime

# Import both PPO variants - Hydra will select which to use
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

# Import custom callbacks
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from callbacks import EnhancedLoggingCallback, ProgressCallback

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import FRE25_IsaacLabSym.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""

    # Extract policy configuration from Hydra (set by policy/*.yaml)
    policy_type = agent_cfg.pop("policy_type", "mlp")
    algorithm_name = agent_cfg.pop("algorithm", "PPO")
    policy_class = agent_cfg.pop("policy_class", "MlpPolicy")

    # Extract hidden states configuration (for MLP with policy memory)
    num_hidden_states = agent_cfg.pop("num_hidden_states", 0)

    print(f"[DEBUG] Policy type: {policy_type}")
    print(f"[DEBUG] Algorithm: {algorithm_name}")
    print(f"[DEBUG] num_hidden_states: {num_hidden_states}")

    # Remove Hydra-specific keys that shouldn't be passed to the algorithm
    agent_cfg.pop("defaults", None)  # Hydra composition metadata

    # Configure environment for hidden states if requested
    if num_hidden_states > 0 and policy_type == "mlp":
        print(f"[INFO] 🔧 Enabling {num_hidden_states} hidden states for MLP")
        env_cfg.num_hidden_states = num_hidden_states
        # Update action and observation spaces
        # Actions: 3 control (steering, throttle, step_command) + N hidden states
        env_cfg.action_space = 3 + num_hidden_states
        # Observations: 44 base + (3 + N) past actions (ALL actions fed back)
        env_cfg.observation_space = 44 + (1 + num_hidden_states)
        print(f"[INFO]    Action space: {env_cfg.action_space} (3 control + {num_hidden_states} hidden states)")
        print(f"[INFO]      - Control: steering, throttle, step_command")
        print(f"[INFO]      - Hidden: {num_hidden_states} state dimensions")
        print(f"[INFO]    Observation space: {env_cfg.observation_space} (44 base + {3 + num_hidden_states} past actions)")
    elif num_hidden_states > 0 and policy_type == "lstm":
        print(f"[WARN] ⚠️  Hidden states ignored for LSTM policy (LSTM has built-in memory)")
        num_hidden_states = 0
        env_cfg.num_hidden_states = 0

    # Select algorithm class based on Hydra config
    if algorithm_name == "RecurrentPPO":
        from sb3_contrib import RecurrentPPO
        algorithm_class = RecurrentPPO
        print(f"[INFO] ✅ Using RecurrentPPO with {policy_class} (LSTM with temporal memory)")
    elif algorithm_name == "PPO":
        from stable_baselines3 import PPO
        algorithm_class = PPO
        memory_info = f" + {num_hidden_states} hidden states" if num_hidden_states > 0 else " (no memory)"
        print(f"[INFO] ✅ Using standard PPO with {policy_class} (feedforward MLP{memory_info})")
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Use 'PPO' or 'RecurrentPPO'")

    print(f"[INFO] Policy Type: {policy_type}")
    print(f"[INFO] n_steps: {agent_cfg.get('n_steps')}, batch_size: {agent_cfg.get('batch_size')}")

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    # Note: policy_class is already set by Hydra, no need to pop "policy"
    n_timesteps = agent_cfg.pop("n_timesteps")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)

    # Check action space type
    action_space = env.action_space
    print(f"[INFO] Action space: {action_space}")
    if hasattr(action_space, 'nvec'):
        print(f"[INFO] Detected MultiDiscrete action space with {len(action_space.nvec)} actions")
        print(f"[INFO] Categories per action: {action_space.nvec}")

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

    # create agent from stable baselines (algorithm selected by Hydra config)
    print(f"[INFO] Creating {algorithm_name} agent with {policy_class}")
    agent = algorithm_class(policy_class, env, verbose=1, **agent_cfg)
    # configure the logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # callbacks for agent
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)
    enhanced_logging = EnhancedLoggingCallback(verbose=1)
    progress_callback = ProgressCallback(check_freq=1000, verbose=1)

    # Combine all callbacks
    callback_list = CallbackList([checkpoint_callback, enhanced_logging, progress_callback])

    # train the agent
    print(f"[INFO] Starting training for {n_timesteps} timesteps...")
    print(f"[INFO] Enhanced logging enabled - tracking max reward, statistics, and progress")
    agent.learn(total_timesteps=n_timesteps, callback=callback_list)
    # save the final model
    agent.save(os.path.join(log_dir, "model"))
    print(f"[INFO] Training complete! Model saved to {log_dir}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
