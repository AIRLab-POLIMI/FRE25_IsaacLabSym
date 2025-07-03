# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the leatherback robot."""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Get absolute path to workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# USD path with proper resolution for cross-platform compatibility
USD_PATH = os.path.join(WORKSPACE_ROOT, "Assets", "Collected_RockerBot", "RockerBot.usd")

wheelsJoints = [
    "front_left_wheel_joint",
    "front_right_wheel_joint",
    "rear_left_wheel_joint",
    "rear_right_wheel_joint"
]

steeringJoints = [
    "front_left_steer_joint",
    "front_right_steer_joint",
    "rear_left_steer_joint",
    "rear_right_steer_joint"
]


ROCKERBOT_CFG: ArticulationCfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.05),
        joint_pos={
            jointName: 0.0 for jointName in steeringJoints
        },
        joint_vel={
            jointName: 0.0 for jointName in wheelsJoints
        },
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel_joint"],
            effort_limit=40000.0,
            velocity_limit=500.0,
            stiffness=0.0,
            damping=100000.0,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=[".*steer_joint"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=10000.0,
            damping=10.0,
        ),
    },
)
