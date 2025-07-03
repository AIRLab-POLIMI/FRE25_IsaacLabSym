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

from .WaypointRelated.Waypoint import WAYPOINT_CFG
from .WaypointRelated.WaypointHandler import WaypointHandler

# import torch.autograd.profiler as profiler


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
