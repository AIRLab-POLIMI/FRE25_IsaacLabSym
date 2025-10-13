# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from .RockerBot import ROCKERBOT_CFG, wheelsJoints, steeringJoints


@configclass
class Fre25IsaaclabsymEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 300.0

    # - spaces definition
    action_space = 3  # Base control actions: steering, throttle, step_command
    # steering: 3 categories {-1, 0, 1}
    # throttle: 3 categories {-1, 0, 1}
    # step_command: 2 categories {0, 1} - binary action to advance command buffer

    # Hidden states support (configurable via agent config)
    # If num_hidden_states > 0, add extra dummy actions for memory
    # ALL past actions (control + hidden) are fed back as observations
    num_hidden_states = 0  # Set by agent config if using hidden states
    hidden_state_scale = 0.005  # Scale for differential hidden state updates (integrator step size)

    # Base observation space (without past actions):
    # 1 current steering + 40 lidar + 3 command buffer = 44 total
    # If hidden states enabled: 44 + (3 + num_hidden_states) for past actions
    observation_space = 1 + 40 * 1 + 3

    state_space = 0

    # simulation with increased GPU memory for many environments
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_max_rigid_contact_count=2**23,  # Increased for many environments
            gpu_max_rigid_patch_count=2**21,
            gpu_found_lost_pairs_capacity=2**25,  # Fix for found/lost pairs error
            gpu_found_lost_aggregate_pairs_capacity=2**20,
            gpu_total_aggregate_pairs_capacity=2**20,
            gpu_collision_stack_size=2**28,  # Fix for collision stack overflow (1GB)
        )
    )

    # robot(s)
    robot_cfg: ArticulationCfg = ROCKERBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=20.0, replicate_physics=True
    )

    # Whether to randomize robot yaw on reset
    randomize_yaw: bool = False

    # custom parameters/scales
    # - controllable joint
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    wheels_dofs_names = wheelsJoints
    steering_dofs_names = steeringJoints
    # - action scale
    action_scale = 50000.0  # [N]
    wheels_effort_scale = 15
    # The range of the steering action is [-1, 1], which corresponds to a steering angle of [-steering_scale, steering_scale] degrees
    steering_scale = 2  # degs/step
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]
