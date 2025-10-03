# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play a checkpoint of an RL agent from skrl."
)
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
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to model checkpoint."
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
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
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

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
    from skrl.resources.schedulers.torch import KLAdaptiveLR
    # Import custom PPO for MultiCategorical action spaces
    from FRE25_IsaacLabSym.agents import PPO_MultiCategorical
elif args_cli.ml_framework.startswith("jax"):
    import jax
    from skrl.agents.jax.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.memories.jax import RandomMemory
    from skrl.resources.schedulers.jax import KLAdaptiveLR

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    get_checkpoint_path,
    load_cfg_from_registry,
    parse_env_cfg,
)

import FRE25_IsaacLabSym.tasks  # noqa: F401
from FRE25_IsaacLabSym.models import CustomPolicy, CustomValue, DiscretePolicy

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    try:
        experiment_cfg = load_cfg_from_registry(
            args_cli.task, f"skrl_{algorithm}_cfg_entry_point"
        )
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join(
        "logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print(
                "[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task."
            )
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
            other_dirs=["checkpoints"],
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

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

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(
        env, ml_framework=args_cli.ml_framework
    )  # same as: `wrap_env(env, wrapper="auto")`

    # Get device
    device = env.device

    # Instantiate custom models
    print("[INFO] Creating custom policy and value models...")

    # Get network configuration from experiment_cfg
    policy_cfg = experiment_cfg["models"]["policy"]
    value_cfg = experiment_cfg["models"]["value"]

    # Extract network architecture
    policy_network = policy_cfg["network"][0]
    policy_hidden_sizes = policy_network["layers"]
    policy_activation = policy_network["activations"]

    value_network = value_cfg["network"][0]
    value_hidden_sizes = value_network["layers"]
    value_activation = value_network["activations"]

    # Determine if using discrete or continuous policy based on action space
    from gymnasium.spaces import MultiDiscrete
    is_discrete = isinstance(env.action_space, MultiDiscrete)

    # Create policy model
    if is_discrete:
        print("[INFO] Using DiscretePolicy for MultiDiscrete action space")
        policy = DiscretePolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            clip_actions=policy_cfg.get("clip_actions", False),
            unnormalized_log_prob=policy_cfg.get("unnormalized_log_prob", True),
            reduction=policy_cfg.get("reduction", "sum"),
            hidden_sizes=policy_hidden_sizes,
            activation=policy_activation,
            num_actions=policy_cfg.get("num_actions", 6),
            num_categories=policy_cfg.get("num_categories", 3)
        )
    else:
        print("[INFO] Using CustomPolicy for continuous action space")
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
    if experiment_cfg["models"].get("separate", False):
        models = {"policy": policy, "value": value}
    else:
        models = {"policy": policy, "value": value}

    # Configure PPO agent
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg.update(experiment_cfg["agent"])

    # Convert learning rate scheduler from string to class
    if "learning_rate_scheduler" in ppo_cfg and isinstance(ppo_cfg["learning_rate_scheduler"], str):
        scheduler_name = ppo_cfg["learning_rate_scheduler"]
        if scheduler_name == "KLAdaptiveLR":
            ppo_cfg["learning_rate_scheduler"] = KLAdaptiveLR
        else:
            ppo_cfg["learning_rate_scheduler"] = None

    # Disable logging and checkpointing for evaluation
    ppo_cfg["experiment"]["write_interval"] = 0
    ppo_cfg["experiment"]["checkpoint_interval"] = 0

    # Create memory
    memory_size = experiment_cfg["agent"]["rollouts"]
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

    # Instantiate the PPO agent
    # Use custom PPO for MultiDiscrete action spaces (fixes stddev tracking bug)
    print("[INFO] Creating PPO agent...")
    is_discrete = hasattr(env.action_space, 'nvec')  # MultiDiscrete has 'nvec' attribute
    PPO_class = PPO_MultiCategorical if is_discrete else PPO

    if is_discrete:
        print("[INFO] Using PPO_MultiCategorical for discrete action space")

    agent = PPO_class(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    agent.load(resume_path)
    # set agent to evaluation mode
    agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {
                    a: outputs[-1][a].get("mean_actions", outputs[0][a])
                    for a in env.possible_agents
                }
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            print(f"Actions:{actions}")
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
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
