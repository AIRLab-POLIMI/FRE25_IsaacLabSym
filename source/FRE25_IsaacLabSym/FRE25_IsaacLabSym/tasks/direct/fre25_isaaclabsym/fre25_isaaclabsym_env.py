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

import matplotlib.pyplot as plt
from gymnasium.spaces import Box


class Fre25IsaaclabsymEnv(DirectRLEnv):
    cfg: Fre25IsaaclabsymEnvCfg

    def __init__(
        self, cfg: Fre25IsaaclabsymEnvCfg, render_mode: str | None = None, **kwargs
    ):
        cfg.action_space = Box(low=-1, high=1, shape=(cfg.action_space,), dtype=float)  # type: ignore
        self.cfg = cfg
        super().__init__(cfg, render_mode, **kwargs)

        self.wheels_dof_idx, _ = self.robots.find_joints(self.cfg.wheels_dofs_names)
        self.steering_dof_idx, _ = self.robots.find_joints(self.cfg.steering_dofs_names)

        self.joint_pos = self.robots.data.joint_pos
        self.joint_vel = self.robots.data.joint_vel

        # steering buffer to allow differential control
        self.steering_buffer = torch.zeros(
            (self.num_envs, len(self.steering_dof_idx)), device=self.device
        )

        # # Initialize plots
        # plt.ion()
        # self.fig, self.ax = plt.subplots()

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
            pathsSpacing=1.5,
            nControlPoints=10,
            pathLength=3,
            pathWidth=0.1,
            pointNoiseStd=0,
        )

        self.paths.generatePath()

        # Initialize waypoints
        self.waypoints: WaypointHandler = WaypointHandler(
            nEnvs=self.scene.num_envs,
            envsOrigins=self.scene.env_origins,
            commandBuffer=self.commandBuffer,
            pathHandler=self.paths,
            waipointReachedEpsilon=0.6,
            maxDistanceToWaypoint=max(
                self.paths.pathLength,
                1.5 * self.paths.pathsSpacing * (self.commandBuffer.maxRows),
            ),
            endOfRowPadding=0.7,
            waypointsPerRow=6,
        )
        self.waypoints.initializeWaypoints()

        # Add Plants to the scene
        self.plants = PlantHandler(
            nPlants=80,
            envsOrigins=self.scene.env_origins,
            plantRadius=0.22,
        )

        self.plants.spawnPlants()

        # buffer for past lidar readings
        self.pastLidar = torch.zeros(
            (self.num_envs, self.plants.raymarcher.raysPerRobot), device=self.device
        )

        # Initialize the robot pose
        self.robot_pose = self.scene.env_origins[:, :2].clone()
        self.past_robot_pose = self.scene.env_origins[:, :2].clone()

        # Initialize past command actions
        self.past_command_actions = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Initialize plant collision buffer
        self.plant_collision_buffer = torch.zeros(self.num_envs, device=self.device)

        # Initialize out of bound buffer
        self.out_of_bound_buffer = torch.zeros(self.num_envs, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Compute bound violations
        absoluteActions = torch.abs(actions)
        boundViolations = absoluteActions - 2
        boundViolations = torch.clamp(boundViolations, min=0.0)
        self.actionsBoundViolations = torch.sum(boundViolations, dim=1)

        actions = torch.clamp(actions, -1, 1)

        if self.actions is None:
            self.actions = torch.zeros_like(actions)
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

        # Update the command buffer when an even numbered waypoint is reached
        command_step_actions = (
            self.waypoints.getReward()
            * (self.waypoints.currentWaypointIndices[:, 1] % 3 == 0)
        ).float()
        command_step_actions = torch.clamp(command_step_actions, 0, 1)

        # Draw a random number and compare to the action to decide whether to step the command buffer
        randomNumber = torch.rand_like(command_step_actions)
        advance_command = randomNumber < command_step_actions
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

        # # plot lidar
        # self.ax.clear()
        # angles = torch.linspace(0, 360, steps=lidar.shape[1], device=self.device)
        # distances = lidar[0] * self.plants.raymarcher.maxDistance
        # points_x = distances * torch.cos(angles * math.pi / 180)
        # points_y = distances * torch.sin(angles * math.pi / 180)
        # self.ax.scatter(points_x.cpu().numpy(), points_y.cpu().numpy())
        # self.ax.plot(points_x.cpu().numpy(), points_y.cpu().numpy(), alpha=0.5)
        # self.ax.set_title("Lidar Readings")
        # self.ax.set_xlabel("X")
        # self.ax.set_ylabel("Y")
        # plt.draw()
        # plt.pause(0.001)

        if self.pastLidar is None:
            self.pastLidar = lidar

        # get the current commands
        currentCommands = self.commandBuffer.getCurrentCommands()

        obs = torch.cat(
            (
                self.steering_buffer % (2 * math.pi),
                self.actions,
                lidar,
                currentCommands,
            ),
            dim=-1,
        )
        observations = {"policy": obs}

        self.pastLidar = lidar

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Reward for staying alive
        # stayAliveReward = (1.0 - self.reset_terminated.float()) / 100000  # (n_envs)

        # Reward for reaching waypoints
        waypointReward = self.waypoints.getReward().float() * 10  # (n_envs,)

        # Reward for moving towards the waypoint computed as the dot product between the robot velocity and the normalized vector from the robot to the waypoint
        toWaypoint = self.waypoints.robotsdiffs
        toWaypointNorm = torch.norm(toWaypoint, dim=1, keepdim=True) + 1e-6
        toWaypointDir = toWaypoint / toWaypointNorm

        # Scalar projection of the robot velocity onto the toWaypointDir
        if self.robot_pose is None:
            self.robot_pose = self.robots.data.root_state_w[:, :2]

        if self.past_robot_pose is None:
            self.past_robot_pose = self.robot_pose

        dt = self.cfg.sim.dt
        if dt <= 0:
            dt = 1e-6

        velocity = (self.robot_pose - self.past_robot_pose) / dt
        velocityTowardsWaypoint = velocity * toWaypointDir
        velocityTowardsWaypoint = torch.sum(velocityTowardsWaypoint, dim=1)

        # print(
        #     f"toWaypointDir: {toWaypointDir}, velocity:{velocity}, dt: {dt}, velocityTowardsWaypoint: {velocityTowardsWaypoint}"
        # )
        velocityTowardsWaypoint /= 10

        # Comunte velocity orthogonal to the waypoint direction and penalize it
        # velocityOrthogonalToWaypoint = (
        #     velocity - velocityTowardsWaypoint[:, None] * toWaypointDir
        # )
        # velocityOrthogonalToWaypoint = torch.norm(velocityOrthogonalToWaypoint, dim=1)
        # velocityOrthogonalToWaypoint = -velocityOrthogonalToWaypoint / 20

        # Waypoint distance based reward
        # toWaypoint = self.waypoints.robotsdiffs
        # distance = torch.norm(toWaypoint, dim=1)
        # waypointDistanceReward = 1 - torch.tanh(distance / self.paths.pathLength)

        # Time out penalty
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeOutPenalty = time_out.float() * -50

        # Penalty for plant collisions
        plantCollisionPenalty = self.plant_collision_buffer.float() * -50
        self.plant_collision_buffer = torch.zeros(self.num_envs, device=self.device)

        # Out of bounds penalty
        out_of_bounds = self.waypoints.robotTooFarFromWaypoint
        outOfBoundsPenalty = out_of_bounds.float() * -50

        # Penalty for action bound violations
        actionBoundViolationPenalty = (
            -0 * self.actionsBoundViolations.float() ** 2
        )  # -1

        totalReward = (
            waypointReward
            + velocityTowardsWaypoint
            # + velocityOrthogonalToWaypoint
            + timeOutPenalty
            + plantCollisionPenalty
            + outOfBoundsPenalty
            + actionBoundViolationPenalty
        )
        # print(f"totalReward: {totalReward}")
        return totalReward / 10

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        DEBUG = True
        # The episode has reached the maximum length
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if DEBUG and any(time_out):
            print(f"Episode length reached: {self.max_episode_length} steps")

        # the robot has reached all the waypoints
        reached_all_waypoints = self.waypoints.taskCompleted
        if DEBUG and any(reached_all_waypoints):
            print("Task completed: reached all waypoints")

        # The robot is too far from the current waypoint
        out_of_bounds = self.waypoints.robotTooFarFromWaypoint
        self.out_of_bound_buffer = out_of_bounds
        if DEBUG and any(out_of_bounds):
            print("Episode terminated: robot out of bounds")

        # Check for plant collisions
        plant_collisions = self.plants.detectPlantCollision()
        self.plant_collision_buffer = plant_collisions
        if DEBUG and any(plant_collisions):
            print("Episode terminated: robot collided with a plant")

        # Completed command buffer
        completed_command_buffer = self.commandBuffer.dones()
        #######################
        #########################
        ########################
        # completed_command_buffer = torch.zeros_like(completed_command_buffer)
        if DEBUG and any(completed_command_buffer):
            print("Task Completed: completed command buffer")

        taskCompleted = reached_all_waypoints | time_out | completed_command_buffer
        taskFailed = out_of_bounds | plant_collisions

        return taskFailed, taskCompleted

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robots._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robots.data.default_joint_pos[env_ids]
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

        # RESET BUFFERS

        # Reset steering buffer
        self.steering_buffer[env_ids] = 0.0

        # Reset robot pose
        self.robot_pose[env_ids] = self.scene.env_origins[env_ids, :2]

        # Reset past robot pose
        self.past_robot_pose[env_ids] = self.scene.env_origins[env_ids, :2]

        # Reset past command actions
        self.past_command_actions[env_ids] = 0

        # Reset plant collision buffer
        self.plant_collision_buffer[env_ids] = 0

        # Reset out of bound buffer
        self.out_of_bound_buffer[env_ids] = 0

        # Reset past lidar
        self.pastLidar[env_ids] = 0.0

        # Reset actions
        self.actions[env_ids] = 0.0

        # Reset action bound violations
        if hasattr(self, "actionsBoundViolations"):
            self.actionsBoundViolations[env_ids] = 0.0
        pass
