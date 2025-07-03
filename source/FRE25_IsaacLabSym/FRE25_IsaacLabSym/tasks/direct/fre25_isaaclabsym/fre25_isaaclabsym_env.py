# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.markers import VisualizationMarkers

from .fre25_isaaclabsym_env_cfg import Fre25IsaaclabsymEnvCfg
from .Waypoint import WAYPOINT_CFG

# import torch.autograd.profiler as profiler


class WaypointHandler:
    def __init__(
        self,
        nEnvs: int,
        envsOrigins: torch.Tensor,
        nWaypoints: int = 10,
        lineLength: float = 10.0,
        lineWidth: float = 1.0,
        lineZ: float = 0.0,
        waipointReachedEpsilon: float = 1,
        maxDistanceToWaypoint: float = 3,
    ):
        assert nEnvs > 0, "Number of environments must be greater than 0, got {}".format(nEnvs)
        self.nEnvs = nEnvs

        assert nWaypoints > 0, "Number of waypoints must be greater than 0, got {}".format(nWaypoints)
        self.nWaypoints = nWaypoints

        assert envsOrigins.shape == (
            nEnvs,
            3,
        ), "envsOrigins must be of shape (nEnvs, 3), but got {}".format(
            envsOrigins.shape
        )
        self.envsOrigins: torch.Tensor = envsOrigins.unsqueeze(1).repeat(
            1, self.nWaypoints, 1
        )

        assert lineLength >= 0, "Line length must be greater than 0, got {}".format(lineLength)
        self.lineLength = lineLength

        assert lineWidth >= 0, "Line width must be greater than 0, got {}".format(lineWidth)
        self.lineWidth = lineWidth

        self.lineZ = lineZ

        assert (
            waipointReachedEpsilon >= 0
        ), "Waypoint reached epsilon must be greater than 0, got {}".format(
            waipointReachedEpsilon
        )
        self.waipointReachedEpsilon = waipointReachedEpsilon

        assert (
            maxDistanceToWaypoint > waipointReachedEpsilon
        ), "Max distance to waypoint must be greater than waypoint reached epsilon, got {} and {}".format(
            maxDistanceToWaypoint, waipointReachedEpsilon
        )
        self.maxDistanceToWaypoint = maxDistanceToWaypoint

        self.waypointsPositions = torch.zeros(
            (nEnvs, nWaypoints, 3), dtype=torch.float32, device=envsOrigins.device
        )

        # The indices of the current waypoint for each environment
        self.currentWaypointIndices = torch.zeros(
            (nEnvs, 2), dtype=torch.int32, device=envsOrigins.device
        )
        self.currentWaypointIndices[:, 0] = torch.arange(
            nEnvs, device=envsOrigins.device
        )

        # The position of the current waypoint for each environment
        self.currentWaypointPositions = torch.zeros(
            (nEnvs, 3), dtype=torch.float32, device=envsOrigins.device
        )

        # The current diffs to the current waypoint for each environment
        self.robotsdiffs = torch.zeros(
            (nEnvs, 2), dtype=torch.float32, device=envsOrigins.device
        )

        # Whether the robot is too far from the current waypoint for each environment
        self.robotTooFarFromWaypoint = torch.zeros(
            (nEnvs), dtype=torch.bool, device=envsOrigins.device
        )
        pass

    def initializeWaypoints(self):
        # Initialize waypoints in a straight line
        waypointsX = torch.linspace(0, self.lineLength, self.nWaypoints).repeat(
            self.nEnvs, 1
        )
        waypointsY = (2 * torch.rand(self.nEnvs, self.nWaypoints) - 1) * self.lineWidth
        waypointsZ = torch.full((self.nEnvs, self.nWaypoints), self.lineZ)

        self.waypointsPositions[:, :, 0] = waypointsX
        self.waypointsPositions[:, :, 1] = waypointsY
        self.waypointsPositions[:, :, 2] = waypointsZ

        # Add the environment origins to the waypoints
        self.waypointsPositions += self.envsOrigins

        self.markersVisualizer = VisualizationMarkers(WAYPOINT_CFG)

        # Select the current waypoint for each environment
        self.currentWaypointPositions = self.waypointsPositions[
            self.currentWaypointIndices[:, 0], self.currentWaypointIndices[:, 1]
        ]

        # update current marker
        self.updateCurrentMarker()

    def visualizeWaypoints(self):
        # Visualize waypoints
        linearizedPositions = self.waypointsPositions.view(
            self.nEnvs * self.nWaypoints, 3
        )
        self.markersVisualizer.visualize(translations=linearizedPositions)

    def randomizeWaipoints(self, env_ids: Sequence[int]):
        # Randomize the y coordinates of the waypoints
        waypointsY = (
            2
            * torch.rand(
                len(env_ids), self.nWaypoints, device=self.waypointsPositions.device
            )
            - 1
        ) * self.lineWidth

        # add the environment origins to the waypoints
        waypointsY += self.envsOrigins[env_ids, :, 1]

        # Update the y coordinates of the waypoints
        self.waypointsPositions[env_ids, :, 1] = waypointsY

    def resetWaypoints(self, env_ids: Sequence[int]):
        # Randomize the waypoints for the given environment ids
        self.randomizeWaipoints(env_ids)

        # Reset the current waypoint indices for the given environment ids
        self.currentWaypointIndices[env_ids, 1] = 0

        # Reset the current waypoint positions for the given environment ids
        self.currentWaypointPositions[env_ids] = self.waypointsPositions[
            self.currentWaypointIndices[env_ids, 0],
            self.currentWaypointIndices[env_ids, 1],
        ]

    def updateCurrentMarker(self):
        indexes = torch.zeros(
            (self.nEnvs, self.nWaypoints),
            dtype=torch.int,
            device=self.waypointsPositions.device,
        )
        indexes[
            self.currentWaypointIndices[:, 0], self.currentWaypointIndices[:, 1]
        ] = 1
        indexes = indexes.view(self.nEnvs * self.nWaypoints)
        self.markersVisualizer.visualize(marker_indices=indexes)

    def diffToCurrentWaypoint(self, robot_pos_xy: torch.Tensor) -> torch.Tensor:
        """
        For each environment, compute the difference between the robot position and the current waypoint position.

        Args:
            robot_pos_xy (torch.Tensor): The robot position in the xy plane. Shape: (nEnvs, 2)
        Returns:
            torch.Tensor: The difference between the robot position and the current waypoint position. Shape: (nEnvs, 2)
        """
        # assert robot_pos_xy.shape == (self.nEnvs, 2), "robot_pos_xy must be of shape (nEnvs, 2), but got {}".format(robot_pos_xy.shape)
        # assert currentWaypointsPositions.shape == (self.nEnvs, 2), "currentWaypointsPositions must be of shape (nEnvs, 2) but got {}".format(currentWaypointsPositions.shape)
        diff = robot_pos_xy - self.currentWaypointPositions[:, :2]
        # assert diff.shape == (self.nEnvs, 2), "diff must be of shape (nEnvs, 2)"
        return diff

    def waypointReachedUpdates(self, waypointReached: torch.Tensor):
        # Update the current waypoint index for each environment
        self.currentWaypointIndices[waypointReached, 1] += 1

        # Update the current waypoint position for each environment
        self.currentWaypointPositions[waypointReached] = self.waypointsPositions[
            self.currentWaypointIndices[waypointReached, 0],
            self.currentWaypointIndices[waypointReached, 1],
        ]

    def updateTooFarFromWaypoint(self):
        # Check if the robot is too far from the current waypoint
        self.robotTooFarFromWaypoint = (
            torch.norm(self.robotsdiffs, dim=1) > self.maxDistanceToWaypoint
        )

    def updateCurrentDiffs(self, robot_pos_xy: torch.Tensor):
        # Update the diffs to the current waypoint for each environment
        self.robotsdiffs = self.diffToCurrentWaypoint(robot_pos_xy)

        # Check if the robot is close to the current waypoint
        close_to_waypoint = (
            torch.norm(self.robotsdiffs, dim=1) < self.waipointReachedEpsilon
        )

        # Check if the robot is close to the current waypoint and update the waypoint index
        self.waypointReachedUpdates(close_to_waypoint)

        # Check if the robot is too far from the current waypoint
        self.updateTooFarFromWaypoint()

        return close_to_waypoint


class Fre25IsaaclabsymEnv(DirectRLEnv):
    cfg: Fre25IsaaclabsymEnvCfg

    def __init__(
        self, cfg: Fre25IsaaclabsymEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        # self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)
        self.wheels_dof_idx, _ = self.robots.find_joints(self.cfg.wheels_dofs_names)
        self.steering_dof_idx, _ = self.robots.find_joints(self.cfg.steering_dofs_names)

        # initialize waypoints
        # Compute the origins of the environments
        envsOrigins = self.scene.env_origins
        self.waypoints: WaypointHandler = WaypointHandler(
            nEnvs=self.scene.num_envs,
            envsOrigins=envsOrigins,
        )
        self.waypoints.initializeWaypoints()

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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.waypoints.visualizeWaypoints()
        self.waypoints.updateCurrentMarker()

    def _apply_action(self) -> None:
        # Use the first half of the dofs for the wheels and the second half for the steering
        steering_actions = (
            self.actions.repeat(1, len(self.steering_dof_idx))
            * self.cfg.steering_scale
            * 3.14
            / 180
        )
        # steering_actions = self.actions[:, len(self.wheels_dof_idx) :]
        self.robots.set_joint_position_target(
            steering_actions, joint_ids=self.steering_dof_idx
        )

        # Normalize wheel actions to be 1
        wheel_actions = torch.ones_like(steering_actions) * self.cfg.wheels_effort_scale
        self.robots.set_joint_effort_target(
            wheel_actions, joint_ids=self.wheels_dof_idx
        )

        pass

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self.wheels_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self.wheels_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self.steering_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self.steering_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        robot_pose = self.robots.data.root_state_w[:, :2]
        # print(f"Robot pose: {robot_pose}")
        # print(f"waypoint: {self.waypoints.currentWaypointPositions}")
        # with profiler.profile(record_shapes=True) as prof:
        self.waypoints.updateCurrentDiffs(robot_pose)
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(f"Robot diffs: {torch.norm(self.waypoints.robotsdiffs, dim=1)}")
        # print(f"Robot pose: {robot_pose}")

        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self.wheels_dof_idx[0]],
            self.joint_vel[:, self.wheels_dof_idx[0]],
            self.joint_pos[:, self.steering_dof_idx[0]],
            self.joint_vel[:, self.steering_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robots.data.joint_pos
        self.joint_vel = self.robots.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # torch.any(
        #     torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos,
        #     dim=1,
        # )
        out_of_bounds = self.waypoints.robotTooFarFromWaypoint
        print(f"Out of bounds: {out_of_bounds}")
        return out_of_bounds, time_out

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


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(
        torch.square(pole_pos).unsqueeze(dim=1), dim=-1
    )
    rew_cart_vel = rew_scale_cart_vel * torch.sum(
        torch.abs(cart_vel).unsqueeze(dim=1), dim=-1
    )
    rew_pole_vel = rew_scale_pole_vel * torch.sum(
        torch.abs(pole_vel).unsqueeze(dim=1), dim=-1
    )
    total_reward = (
        rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    )
    return total_reward
