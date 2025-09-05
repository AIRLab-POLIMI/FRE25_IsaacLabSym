# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.markers import VisualizationMarkers
from isaaclab.assets import (
    Articulation,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)

import isaacsim.core.utils.prims as prim_utils

from .fre25_isaaclabsym_env_cfg import Fre25IsaaclabsymEnvCfg

from .WaypointRelated.Waypoint import WAYPOINT_CFG
from .WaypointRelated.WaypointHandler import WaypointHandler
from .PlantRelated.PlantHandler import PlantHandler
from .PathHandler import PathHandler
from .CommandBuffer import CommandBuffer
# import torch.autograd.profiler as profiler
import isaaclab.sim.schemas as schemas
from isaaclab.utils.math import axis_angle_from_quat as quat2axis


class Fre25IsaaclabsymEnv(DirectRLEnv):
    cfg: Fre25IsaaclabsymEnvCfg

    def __init__(
        self, cfg: Fre25IsaaclabsymEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.wheels_dof_idx, _ = self.robots.find_joints(self.cfg.wheels_dofs_names)
        self.steering_dof_idx, _ = self.robots.find_joints(self.cfg.steering_dofs_names)

        self.joint_pos = self.robots.data.joint_pos
        self.joint_vel = self.robots.data.joint_vel

    def _setup_scene(self):
        self.robots: Articulation = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add articulation to scene
        self.scene.articulations["robot"] = self.robots

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Add waypoint markers to the scene
        self.waypoint_markers = VisualizationMarkers(WAYPOINT_CFG)

        # Initialize command buffer
        self.commandBuffer = CommandBuffer(
            nEnvs=self.scene.num_envs,
            commandsLength=3,
            maxRows=2,
            device=self.device,
        )
        self.commandBuffer.randomizeCommands()

        # Build the paths
        self.paths = PathHandler(
            device=self.device,
            nEnvs=self.scene.num_envs,
            nPaths=6,
            pathsSpacing=2.0,
            nControlPoints=10,
            pathLength=2.5,
            pathWidth=.5,
            pointNoiseStd=0.2,
        )

        self.paths.generatePath()

        # Initialize waypoints
        self.waypoints: WaypointHandler = WaypointHandler(
            nEnvs=self.scene.num_envs,
            envsOrigins=self.scene.env_origins,
            commandBuffer=self.commandBuffer,
            pathHandler=self.paths,
            maxDistanceToWaypoint=self.paths.pathLength * 1.5,
        )
        self.waypoints.initializeWaypoints()

        # Add Plants to the scene
        self.plants = PlantHandler(
            nPlants=100,
            envsOrigins=self.scene.env_origins,
            plantRadius=0.3,
        )

        self.plants.spawnPlants()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.waypoints.visualizeWaypoints()
        self.waypoints.updateCurrentMarker()

    def _apply_action(self) -> None:
        # Use the first half of the dofs for the wheels and the second half for the steering
        steering_actions = self.actions[:, [0]]
        steering_actions = (
            steering_actions.repeat(1, len(self.steering_dof_idx))
            * self.cfg.steering_scale
            * 3.14
            / 180
        )
        # steering_actions = self.actions[:, len(self.wheels_dof_idx) :]
        # steering_actions = torch.zeros_like(steering_actions)
        self.robots.set_joint_position_target(
            steering_actions, joint_ids=self.steering_dof_idx
        )

        # Normalize wheel actions to be 1
        wheel_actions = self.actions[:, [1]] * self.cfg.wheels_effort_scale
        self.robots.set_joint_velocity_target(
            wheel_actions, joint_ids=self.wheels_dof_idx
        )

        # Update the command buffer
        command_step_actions = self.actions[:, -1]
        command_step_actions = torch.clamp(command_step_actions, 0, 1)

        # If the action is > 0.5, advance the command buffer by one step
        advance_command = command_step_actions > 0.5
        self.commandBuffer.stepCommands(advance_command)

        pass

    def _get_observations(self) -> dict:
        robot_pose = self.robots.data.root_state_w[:, :2]
        robot_quat = self.robots.data.root_state_w[:, 3:7]
        robotEuler = quat2axis(robot_quat)
        robotZs = robotEuler[:, [2]]  # Extract the Z component of the Euler angles
        self.waypoints.updateCurrentDiffs(robot_pose)
        lidar = self.plants.computeDistancesToPlants(
            robot_pose,
            robotZs,
        )

        # get the current commands
        currentCommands = self.commandBuffer.getCurrentCommands()

        obs = torch.cat(
            (
                self.joint_pos[:, self.steering_dof_idx],
                lidar,
                currentCommands
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        # print(f"Robot pose: {robot_pose}")
        # print(f"waypoint: {self.waypoints.currentWaypointPositions}")
        # with profiler.profile(record_shapes=True) as prof:
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        # print(f"Robot diffs: {torch.norm(self.waypoints.robotsdiffs, dim=1)}")
        # print(f"Robot pose: {robot_pose}")

        return observations

    def _get_rewards(self) -> torch.Tensor:
        stayAliveReward = (1.0 - self.reset_terminated.float())
        waypointReward = self.waypoints.currentWaypointIndices[:, 1].float()
        totalReward = stayAliveReward + waypointReward
        return totalReward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        DEBUG = False
        # The episode has reached the maximum length
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if DEBUG and any(time_out):
            print(f"Episode length reached: {self.max_episode_length} steps")

        # the robot has reached all the waypoints
        reached_all_waypoints = self.waypoints.taskCompleted
        if DEBUG and any(reached_all_waypoints):
            print(f"Task completed: reached all waypoints")

        # The robot is too far from the current waypoint
        out_of_bounds = self.waypoints.robotTooFarFromWaypoint
        if DEBUG and any(out_of_bounds):
            print(f"Episode terminated: robot out of bounds")

        # Check for plant collisions
        robot_pose = self.robots.data.root_state_w[:, :2]
        plant_collisions = self.plants.detectPlantCollision()
        if DEBUG and any(plant_collisions):
            print(f"Episode terminated: robot collided with a plant")

        # Completed command buffer
        completed_command_buffer = self.commandBuffer.dones()
        #######################
        #########################
        ########################
        completed_command_buffer = torch.zeros_like(completed_command_buffer)
        if DEBUG and any(completed_command_buffer):
            print(f"Task Failed: completed command buffer")

        taskCompleted = reached_all_waypoints | time_out
        taskFailed = out_of_bounds | plant_collisions | completed_command_buffer

        return taskFailed, taskCompleted

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robots._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robots.data.default_joint_pos[env_ids]
        # joint_pos[:, self._pole_dof_idx] += sample_uniform(
        #     self.cfg.initial_pole_angle_range[0] * math.pi,
        #     self.cfg.initial_pole_angle_range[1] * math.pi,
        #     joint_pos[:, self._pole_dof_idx].shape,
        #     joint_pos.device,
        # )
        joint_vel = self.robots.data.default_joint_vel[env_ids]

        default_root_state = self.robots.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robots.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robots.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robots.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.waypoints.resetWaypoints(env_ids)

        # Generate new path for the environment
        self.paths.generatePath()

        # Randomize plant positions
        self.plants.randomizePlantsPositions(env_ids, self.paths)

        # Reset command buffer
        self.commandBuffer.randomizeCommands(env_ids)
