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
    num_hidden_states = 4  # Set by agent config if using hidden states
    hidden_state_scale = 0.005  # Scale for differential hidden state updates (integrator step size)

    # LiDAR / RayMarcher Parameters (declared early as they affect observation_space)
    lidar_rays_per_robot = 40  # Number of LiDAR rays per robot (observation space dimension)
    lidar_max_distance = 1.0  # Maximum sensing distance for LiDAR [m]
    lidar_tolerance = 0.01  # Tolerance for raymarching convergence [m]
    lidar_max_steps = 100  # Maximum number of raymarching steps per ray

    # Base observation space (without past actions):
    # 1 current steering + lidar_rays_per_robot + 3 command buffer
    # If hidden states enabled: base + (3 + num_hidden_states) for past actions
    observation_space = 1 + lidar_rays_per_robot + 3

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

    # ========================================================================================
    # Robot Control Parameters
    # ========================================================================================
    # - Controllable joints
    cart_dof_name = "slider_to_cart"  # Legacy parameter (not used in current env)
    pole_dof_name = "cart_to_pole"  # Legacy parameter (not used in current env)
    wheels_dofs_names = wheelsJoints
    steering_dofs_names = steeringJoints

    # - Action scales
    action_scale = 50000.0  # [N] - Legacy parameter (not used in current env)
    wheels_effort_scale = 15  # Velocity scale for wheel motors
    steering_scale = 2  # Steering angle change per step [degrees/step]

    # - Steering limits
    steering_buffer_min = -3.14 / 2  # Minimum steering angle [rad] (~-90 degrees)
    steering_buffer_max = 3.14 / 2  # Maximum steering angle [rad] (~90 degrees)

    # ========================================================================================
    # Scene Parameters
    # ========================================================================================
    # - Ground plane
    ground_plane_size = (10000000.0, 10000000.0)  # Ground plane dimensions [m]

    # - Lighting
    dome_light_intensity = 2000.0  # Scene lighting intensity
    dome_light_color = (0.75, 0.75, 0.75)  # Scene lighting RGB color

    # ========================================================================================
    # Command Buffer Parameters
    # ========================================================================================
    commands_length = 8  # Number of commands in sequence
    max_rows = 1  # Maximum number of rows per command (1 to max_rows)

    # ========================================================================================
    # Path Handler Parameters
    # ========================================================================================
    # Number of paths = 2 * (commands_length + 1) - two paths per command (left/right)
    paths_spacing = 1.2  # Spacing between parallel paths [m]
    n_control_points = 10  # Number of control points for path generation
    path_length = 3.0  # Length of each path segment [m]
    path_width = 0.15  # Width of each path [m]
    point_noise_std = 0.03  # Standard deviation of noise added to path points [m]

    # ========================================================================================
    # Waypoint Handler Parameters
    # ========================================================================================
    waypoint_reached_epsilon = 0.35  # Distance threshold to consider waypoint reached [m]
    max_distance_to_waypoint = 1.8  # Maximum allowed distance from waypoint before out-of-bounds [m]
    end_of_row_padding = 0.4  # Additional distance padding at end of each row [m]
    extra_waypoint_padding = 0.8  # Additional spacing for extra waypoints between rows [m]
    waypoints_per_row = 3  # Number of waypoints per row

    # ========================================================================================
    # Plant Handler Parameters
    # ========================================================================================
    n_plants_per_path = 10  # Number of plants per path (total = n_plants_per_path * n_paths)
    plant_radius = 0.22  # Radius of each plant for collision detection [m]

    # ========================================================================================
    # Reward Function Parameters
    # ========================================================================================
    # - Waypoint rewards
    waypoint_reward_base = 100.0  # Base reward for reaching a waypoint (The actual ammount reppresnt the reward for an entire row)

    # - Velocity rewards
    velocity_towards_waypoint_scale = 0.5  # Scale for velocity projection reward

    # - Penalties
    timeout_penalty = -100.0  # Penalty for reaching max episode length
    plant_collision_penalty = -100.0  # Penalty for colliding with a plant
    out_of_bounds_penalty = -100.0  # Penalty for exceeding max distance from waypoint
    command_step_penalty = 0.0  # Penalty for taking a command step (currently disabled)
    command_index_penalty_scale = -1.0  # Scale for command buffer misalignment penalty
    distance_penalty_scale = 1.0  # Scale for distance to waypoint penalty
    distance_penalty_threshold = 0.1  # Threshold for distance penalty [m]

    # - Final reward scaling
    total_reward_scale = 0.1  # Final scaling factor applied to total reward (division by 10)
