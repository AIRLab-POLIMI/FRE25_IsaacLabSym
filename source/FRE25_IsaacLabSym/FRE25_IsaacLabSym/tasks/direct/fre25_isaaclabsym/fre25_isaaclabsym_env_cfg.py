# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from .RockerBot import ROCKERBOT_CFG, wheelsJoints, steeringJoints


@configclass
class Fre25IsaaclabsymEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0
    # - spaces definition
    action_space = 4  # (2, {2})

    # 4 current steering + 4 past actions + 40 lidar  + 3 command buffer (maxRows=1 + isItRight? 1 + 1 * parity)
    observation_space = 4 + 4 + 40 * 1 + 3

    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = ROCKERBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=20.0, replicate_physics=True
    )

    # custom parameters/scales
    # - controllable joint
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    wheels_dofs_names = wheelsJoints
    steering_dofs_names = steeringJoints
    # - action scale
    action_scale = 50000.0  # [N]
    wheels_effort_scale = 20
    # The range of the steering action is [-1, 1], which corresponds to a steering angle of [-steering_scale, steering_scale] degrees
    steering_scale = 1.2  # degs/step
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]
