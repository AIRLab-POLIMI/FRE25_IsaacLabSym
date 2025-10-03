# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to model checkpoint to resume training.",
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

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
import os
import random
from datetime import datetime

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    import torch
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory
    from skrl.trainers.torch import SequentialTrainer
    from skrl.resources.schedulers.torch import KLAdaptiveLR
elif args_cli.ml_framework.startswith("jax"):
    import jax
    from skrl.agents.jax.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.memories.jax import RandomMemory
    from skrl.trainers.jax import SequentialTrainer
    from skrl.resources.schedulers.jax import KLAdaptiveLR

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import FRE25_IsaacLabSym.tasks  # noqa: F401
from FRE25_IsaacLabSym.models import CustomPolicy, CustomValue

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = (
    "skrl_cfg_entry_point"
    if algorithm in ["ppo"]
    else f"skrl_{algorithm}_cfg_entry_point"
)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict
):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = (
            args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
        )
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = (
        args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    )
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join(
        "logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + f"_{algorithm}_{args_cli.ml_framework}"
    )
    print(f"Exact experiment name requested from command line {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = (
        retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    )

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
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

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(
        env, ml_framework=args_cli.ml_framework
    )  # same as: `wrap_env(env, wrapper="auto")`

    # Get device
    device = env.device

    # Instantiate custom models
    print("[INFO] Creating custom policy and value models...")

    # Get network configuration from agent_cfg
    policy_cfg = agent_cfg["models"]["policy"]
    value_cfg = agent_cfg["models"]["value"]

    # Extract network architecture
    policy_network = policy_cfg["network"][0]
    policy_hidden_sizes = policy_network["layers"]
    policy_activation = policy_network["activations"]

    value_network = value_cfg["network"][0]
    value_hidden_sizes = value_network["layers"]
    value_activation = value_network["activations"]

    # Create policy model
    policy = CustomPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        clip_actions=policy_cfg.get("clip_actions", False),
        clip_log_std=policy_cfg.get("clip_log_std", True),
        min_log_std=policy_cfg.get("min_log_std", -20),
        max_log_std=policy_cfg.get("max_log_std", 2),
        initial_log_std=policy_cfg.get("initial_log_std", 0),
        hidden_sizes=policy_hidden_sizes,
        activation=policy_activation
    )

    # Create value model
    value = CustomValue(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        clip_actions=value_cfg.get("clip_actions", False),
        hidden_sizes=value_hidden_sizes,
        activation=value_activation
    )

    print(f"[INFO] Policy model: {policy}")
    print(f"[INFO] Value model: {value}")

    # Create models dictionary
    models = {}
    if agent_cfg["models"].get("separate", False):
        models = {"policy": policy, "value": value}
    else:
        # Shared model (policy and value use same network)
        models = {"policy": policy, "value": value}

    # Configure PPO agent
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg.update(agent_cfg["agent"])

    # Convert learning rate scheduler from string to class
    if "learning_rate_scheduler" in ppo_cfg and isinstance(ppo_cfg["learning_rate_scheduler"], str):
        scheduler_name = ppo_cfg["learning_rate_scheduler"]
        if scheduler_name == "KLAdaptiveLR":
            ppo_cfg["learning_rate_scheduler"] = KLAdaptiveLR
        else:
            # If it's an unknown scheduler, set to None
            print(f"[WARNING] Unknown scheduler '{scheduler_name}', disabling scheduler")
            ppo_cfg["learning_rate_scheduler"] = None

    # Create memory
    memory_size = agent_cfg["agent"]["rollouts"]
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

    # Instantiate the PPO agent
    print("[INFO] Creating PPO agent...")
    agent = PPO(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )

    # Configure and instantiate the sequential trainer
    # https://skrl.readthedocs.io/en/latest/api/trainers/sequential.html
    trainer_cfg = {
        "timesteps": agent_cfg["trainer"]["timesteps"],
        "headless": True,
        "disable_progressbar": False,
        "close_environment_at_exit": agent_cfg["trainer"].get("close_environment_at_exit", False),
    }

    print("[INFO] Creating trainer...")
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)

    # run training
    print("[INFO] Starting training...")
    trainer.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
