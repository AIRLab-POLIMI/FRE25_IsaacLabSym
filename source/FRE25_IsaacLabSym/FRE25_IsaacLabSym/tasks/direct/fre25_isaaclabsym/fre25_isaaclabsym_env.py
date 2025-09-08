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

        # steering buffer to allow differential control
        self.steering_buffer = torch.zeros(
            (self.num_envs, len(self.steering_dof_idx)), device=self.device
        )

        # buffer for past lidar readings
        self.pastLidar = None

        # Initialize the robot pose
        self.robot_pose = None
        self.past_robot_pose = None

        # Initialize past command actions
        self.past_command_actions = None

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
            maxRows=1,
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
            pathLength=1.5,
            pathWidth=0.2,
            pointNoiseStd=0.2,
        )

        self.paths.generatePath()

        # Initialize waypoints
        self.waypoints: WaypointHandler = WaypointHandler(
            nEnvs=self.scene.num_envs,
            envsOrigins=self.scene.env_origins,
            commandBuffer=self.commandBuffer,
            pathHandler=self.paths,
            waipointReachedEpsilon=0.5,
            maxDistanceToWaypoint=max(
                self.paths.pathLength * 1.5,
                1.5 * self.paths.pathsSpacing * (self.commandBuffer.commandsLength + 1),
            ),
        )
        self.waypoints.initializeWaypoints()

        # Add Plants to the scene
        self.plants = PlantHandler(
            nPlants=50,
            envsOrigins=self.scene.env_origins,
            plantRadius=0.2,
        )

        self.plants.spawnPlants()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.waypoints.visualizeWaypoints()
        self.waypoints.updateCurrentMarker()

    def _apply_action(self) -> None:
        # Use the first half of the dofs for the wheels and the second half for the steering
        steering_actions = self.actions[:, [0]]
        steering_actions = torch.clamp(steering_actions, -1, 1)
        steering_actions = (
            steering_actions.repeat(1, len(self.steering_dof_idx))
            * self.cfg.steering_scale
            * 3.14
            / 180
        )
        self.steering_buffer += steering_actions
        self.steering_buffer = torch.clamp(
            self.steering_buffer,
            -3.14 / 2,
            3.14 / 2,
        )
        # steering_actions = self.actions[:, len(self.wheels_dof_idx) :]
        # steering_actions = torch.zeros_like(steering_actions)
        self.robots.set_joint_position_target(
            self.steering_buffer, joint_ids=self.steering_dof_idx
        )

        # Normalize wheel actions to be 1
        wheel_actions = self.actions[:, [1]]
        wheel_actions = torch.clamp(wheel_actions, -1, 1)
        wheel_actions *= self.cfg.wheels_effort_scale

        self.robots.set_joint_velocity_target(
            wheel_actions, joint_ids=self.wheels_dof_idx
        )

        # Update the command buffer
        command_step_actions = self.actions[:, -1]
        command_step_actions = torch.clamp(command_step_actions, 0, 1)

        # If the action is > 0.5, advance the command buffer by one step
        advance_command = command_step_actions > 0.5
        if self.past_command_actions is None:
            self.past_command_actions = advance_command

        step = advance_command & (~self.past_command_actions)

        self.commandBuffer.stepCommands(step)
        self.past_command_actions = advance_command

        pass

    def _get_observations(self) -> dict:
        self.past_robot_pose = self.robot_pose
        self.robot_pose = self.robots.data.root_state_w[:, :2]

        robot_quat = self.robots.data.root_state_w[:, 3:7]
        robotEuler = quat2axis(robot_quat)
        robotZs = robotEuler[:, [2]]  # Extract the Z component of the Euler angles
        self.waypoints.updateCurrentDiffs(self.robot_pose)
        lidar = self.plants.computeDistancesToPlants(
            self.robot_pose,
            robotZs,
        )

        if self.pastLidar is None:
            self.pastLidar = lidar

        # get the current commands
        currentCommands = self.commandBuffer.getCurrentCommands()

        obs = torch.cat(
            (
                self.joint_pos[:, self.steering_dof_idx],
                self.pastLidar,
                lidar,
                currentCommands,
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

        self.pastLidar = lidar

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Reward for staying alive
        stayAliveReward = (1.0 - self.reset_terminated.float()) / 100000  # (n_envs)

        # Reward for reaching waypoints
        waypointReward = self.waypoints.getReward().float()  # (n_envs,)

        # Reward for moving towards the waypoint computed as the dot product between the robot velocity and the normalized vector from the robot to the waypoint
        toWaypoint = self.waypoints.robotsdiffs
        toWaypointNorm = torch.norm(toWaypoint, dim=1, keepdim=True) + 1e-6
        toWaypointDir = toWaypoint / toWaypointNorm

        # Scalar projection of the robot velocity onto the toWaypointDir
        if self.robot_pose is None:
            self.robot_pose = self.robots.data.root_state_w[:, :2]

        if self.past_robot_pose is None:
            self.past_robot_pose = self.robot_pose

        velocityTowardsWaypoint = (
            self.robot_pose - self.past_robot_pose
        ) * toWaypointDir
        velocityTowardsWaypoint = torch.sum(velocityTowardsWaypoint, dim=1)

        # Time out penalty
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeOutPenalty = time_out.float() * -1000

        totalReward = (
            stayAliveReward + waypointReward + velocityTowardsWaypoint + timeOutPenalty
        )
        # print(f"totalReward: {totalReward}")
        return totalReward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        DEBUG = True
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

        # Generate new path for the environment
        self.paths.generatePath()

        # Randomize plant positions
        self.plants.randomizePlantsPositions(env_ids, self.paths)

        # Reset command buffer
        self.commandBuffer.randomizeCommands(env_ids)

        # Reset waypoints
        self.waypoints.resetWaypoints(env_ids)

        # Reset steering buffer
        self.steering_buffer[env_ids] = 0.0
