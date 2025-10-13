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
from isaaclab.markers import VisualizationMarkers
from isaaclab.assets import Articulation
from isaaclab.utils.math import axis_angle_from_quat as quat2axis

from .fre25_isaaclabsym_env_cfg import Fre25IsaaclabsymEnvCfg
from .WaypointRelated.Waypoint import WAYPOINT_CFG
from .WaypointRelated.WaypointHandler import WaypointHandler
from .PlantRelated.PlantHandler import PlantHandler
from .PathHandler import PathHandler
from .CommandBuffer import CommandBuffer
from .CommandBuffer.CommandMarkerVisualizer import CommandBufferVisualizer

from gymnasium.spaces import MultiDiscrete


class Fre25IsaaclabsymEnv(DirectRLEnv):
    cfg: Fre25IsaaclabsymEnvCfg

    def __init__(
        self, cfg: Fre25IsaaclabsymEnvCfg, render_mode: str | None = None, **kwargs
    ):
        """Initialize the FRE25 agricultural navigation environment.

        Constructs a MultiDiscrete action space consisting of:
        - Steering: 3 categories {-1, 0, +1} (left, neutral, right)
        - Throttle: 3 categories {-1, 0, +1} (backward, neutral, forward)
        - Command buffer step: 2 categories {0, 1} (hold, advance)
        - Hidden states (optional): N × 2 categories {-1, +1} (memory accumulators)

        Args:
            cfg: Environment configuration containing all simulation parameters
            render_mode: Rendering mode for visualization
            **kwargs: Additional arguments passed to parent DirectRLEnv
        """
        cfg.nActions = cfg.action_space

        # Configure discrete action space
        num_hidden = getattr(cfg, 'num_hidden_states', 0)
        action_categories = [3, 3, 2] + [2] * num_hidden
        cfg.action_space = MultiDiscrete(action_categories)  # type: ignore

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

    def _setup_scene(self):
        """Set up the simulation scene with robot, ground, lighting, and task components.

        Initializes:
        - Robot articulation with Rockerbot model
        - Ground plane and lighting
        - Waypoint visualization markers
        - Command buffer for task instructions
        - Path handler for procedural row generation
        - Waypoint handler for navigation targets
        - Plant handler with ray marching LIDAR
        - Episode tracking buffers and statistics
        """
        self.robots: Articulation = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(size=self.cfg.ground_plane_size)
        )

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add articulation to scene
        self.scene.articulations["robot"] = self.robots

        # add lights
        light_cfg = sim_utils.DomeLightCfg(
            intensity=self.cfg.dome_light_intensity,
            color=self.cfg.dome_light_color
        )
        light_cfg.func("/World/Light", light_cfg)

        # Add waypoint markers to the scene
        self.waypoint_markers = VisualizationMarkers(WAYPOINT_CFG)

        # Initialize Command Visualizer
        self.command_visualizer = CommandBufferVisualizer()

        # Initialize command buffer
        self.commandBuffer = CommandBuffer(
            nEnvs=self.scene.num_envs,
            commandsLength=self.cfg.commands_length,
            maxRows=self.cfg.max_rows,
            device=self.device,
        )
        self.commandBuffer.randomizeCommands()

        # Build the paths
        self.paths = PathHandler(
            device=self.device,
            nEnvs=self.scene.num_envs,
            nPaths=2 * (self.cfg.commands_length + 1),  # Two paths per command (left/right)
            pathsSpacing=self.cfg.paths_spacing,
            nControlPoints=self.cfg.n_control_points,
            pathLength=self.cfg.path_length,
            pathWidth=self.cfg.path_width,
            pointNoiseStd=self.cfg.point_noise_std,
        )

        self.paths.generatePath()

        # Initialize waypoints
        self.waypoints: WaypointHandler = WaypointHandler(
            nEnvs=self.scene.num_envs,
            envsOrigins=self.scene.env_origins,
            commandBuffer=self.commandBuffer,
            pathHandler=self.paths,
            waipointReachedEpsilon=self.cfg.waypoint_reached_epsilon,
            maxDistanceToWaypoint=self.cfg.max_distance_to_waypoint,
            endOfRowPadding=self.cfg.end_of_row_padding,
            extraWaypointPadding=self.cfg.extra_waypoint_padding,
            waypointsPerRow=self.cfg.waypoints_per_row,
        )
        self.waypoints.initializeWaypoints()

        # Add Plants to the scene
        self.plants = PlantHandler(
            nPlants=self.cfg.n_plants_per_path * self.paths.nPaths,
            envsOrigins=self.scene.env_origins,
            plantRadius=self.cfg.plant_radius,
            raysPerRobot=self.cfg.lidar_rays_per_robot,
            maxDistance=self.cfg.lidar_max_distance,
            tol=self.cfg.lidar_tolerance,
            maxSteps=self.cfg.lidar_max_steps,
        )

        self.plants.spawnPlants()

        # buffer for past lidar readings
        self.pastLidar = torch.zeros(
            (self.num_envs, self.plants.raymarcher.raysPerRobot), device=self.device
        )

        # Action buffer configuration: stores ALL past actions (control + hidden states)
        # for edge detection and observation feedback
        self.num_hidden_states = getattr(self.cfg, "num_hidden_states", 0)
        self.num_control_actions = 3  # steering, throttle, step_command
        self.num_total_actions = self.num_control_actions + self.num_hidden_states

        # Buffer stores ALL past actions to be fed back as observations
        self.past_actions = torch.zeros(
            (self.num_envs, self.num_total_actions), device=self.device
        )

        # Initialize hidden state accumulators for differential control
        # Hidden actions {-1, +1} are integrated into bounded accumulators [-1, 1]
        if self.num_hidden_states > 0:
            self.hidden_state_accumulators = torch.zeros(
                (self.num_envs, self.num_hidden_states), device=self.device
            )

        # Initialize the robot pose
        self.robot_pose = self.scene.env_origins[:, :2].clone()
        self.past_robot_pose = self.scene.env_origins[:, :2].clone()

        # Initialize plant collision buffer
        self.plant_collision_buffer = torch.zeros(self.num_envs, device=self.device)

        # Initialize out of bound buffer
        self.out_of_bound_buffer = torch.zeros(self.num_envs, device=self.device)

        # Episode statistics tracking (for TensorBoard logging)
        self.episode_waypoints_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_plant_collisions = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_out_of_bounds = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_timeouts = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Buffer to track command steps taken this timestep (for penalty)
        self.command_step_buffer = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Tracking for "average steps between command steps" metric
        self.steps_since_last_command_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_command_steps_taken = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_total_steps_with_commands = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Convert discrete actions to continuous values and prepare for physics simulation.

        Action conversion:
        - Steering/throttle: {0, 1, 2} → {-1, 0, +1}
        - Step command: {0, 1} → {0, 1} (boolean)
        - Hidden states: {0, 1} → {-1, +1}

        Args:
            actions: Discrete action indices from the policy [batch_size, num_actions]
        """
        self.past_robot_pose = self.robot_pose

        # Convert discrete action indices to continuous values
        converted_actions = actions.clone().float()
        converted_actions[:, 0] = actions[:, 0] - 1  # steering: {0,1,2} → {-1,0,+1}
        converted_actions[:, 1] = actions[:, 1] - 1  # throttle: {0,1,2} → {-1,0,+1}
        converted_actions[:, 2] = actions[:, 2]  # step_command: keep as {0,1}
        if self.num_hidden_states > 0:
            converted_actions[:, 3:] = actions[:, 3:] * 2 - 1  # hidden: {0,1} → {-1,+1}

        self.actions = converted_actions
        self.waypoints.visualizeWaypoints()
        self.waypoints.updateCurrentMarker()
        self.command_visualizer.visualizeCommands(self.robots.data.root_state_w[:, :3], self.commandBuffer)

    def _apply_action(self) -> None:
        """Apply converted actions to robot joints and command buffer.

        Implements:
        - Differential steering control with clamped buffer
        - Direct velocity control for wheels
        - Rising edge detection for command buffer stepping
        - Differential integration for hidden state accumulators
        """
        # Update robot position and waypoint detection
        self.robot_pose = self.robots.data.root_state_w[:, :2]
        self.waypoints.updateCurrentDiffs(self.robot_pose)

        # Apply differential steering control (clamped to limits)
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
            self.cfg.steering_buffer_min,
            self.cfg.steering_buffer_max,
        )
        self.robots.set_joint_position_target(
            self.steering_buffer, joint_ids=self.steering_dof_idx
        )

        # Apply wheel velocity commands
        wheel_actions = self.actions[:, [1]]
        wheel_actions = torch.clamp(wheel_actions, -1, 1)
        wheel_actions *= self.cfg.wheels_effort_scale
        self.robots.set_joint_velocity_target(
            wheel_actions, joint_ids=self.wheels_dof_idx
        )

        # Command buffer stepping with rising edge detection
        step_command_action = self.actions[:, 2]
        step_command = (step_command_action > 0.5).bool()
        past_step_command = (self.past_actions[:, 2] > 0.5).bool()
        rising_edge = step_command & (~past_step_command)

        # Step command buffer on rising edge (0→1 transition only)
        # Note: Buffer is reset in _get_rewards() after penalty computation
        # due to decimation (4 calls per step)
        if rising_edge.any():
            env_ids_to_step = torch.where(rising_edge)[0].tolist()
            self.commandBuffer.stepCommands(env_ids_to_step)
            self.command_step_buffer[env_ids_to_step] = True

            # Track steps between command steps for metric logging
            self.episode_total_steps_with_commands[env_ids_to_step] += self.steps_since_last_command_step[env_ids_to_step]
            self.episode_command_steps_taken[env_ids_to_step] += 1
            self.steps_since_last_command_step[env_ids_to_step] = 0

        # Increment step counter for all environments
        self.steps_since_last_command_step += 1

        # Store ALL actions in internal buffer for edge detection and observations
        # Control actions: steering, throttle, step_command (normalized to proper ranges)
        all_actions = self.actions[:, :self.num_total_actions].clone()
        all_actions[:, 0:2] = torch.clamp(all_actions[:, 0:2], -1, 1)  # steering, throttle
        all_actions[:, 2] = torch.clamp(all_actions[:, 2], 0, 1)  # step_command (binary)

        # Apply differential control to hidden state accumulators
        # Actions {-1, +1} are velocity commands, accumulators are integrated positions
        if self.num_hidden_states > 0:
            hidden_deltas = all_actions[:, 3:] * self.cfg.hidden_state_scale
            self.hidden_state_accumulators += hidden_deltas
            self.hidden_state_accumulators = torch.clamp(
                self.hidden_state_accumulators, -1.0, 1.0
            )
            all_actions[:, 3:] = self.hidden_state_accumulators.clone()

        self.past_actions = all_actions

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine episode termination conditions.

        Termination occurs when:
        - Time out: Episode reaches maximum length
        - Success: All waypoints reached
        - Failure: Robot out of bounds or plant collision
        - Buffer complete: Command buffer exhausted (currently disabled)

        Returns:
            Tuple of (task_failed, task_completed) boolean tensors
        """
        # Track timeout occurrences
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.episode_timeouts += time_out.long()

        # Task completion: reached all waypoints
        reached_all_waypoints = self.waypoints.taskCompleted

        # Out of bounds detection and tracking
        out_of_bounds = self.waypoints.robotTooFarFromWaypoint
        self.out_of_bound_buffer = out_of_bounds
        self.episode_out_of_bounds += out_of_bounds.long()

        # Plant collision detection and tracking
        plant_collisions = self.plants.detectPlantCollision()
        self.plant_collision_buffer = plant_collisions
        self.episode_plant_collisions += plant_collisions.long()

        # Command buffer completion (currently disabled)
        completed_command_buffer = self.commandBuffer.dones()
        completed_command_buffer = torch.zeros_like(completed_command_buffer)

        taskCompleted = reached_all_waypoints | time_out | completed_command_buffer
        taskFailed = out_of_bounds | plant_collisions
        dones = taskCompleted | taskFailed

        # Log episode statistics for TensorBoard (sum and count only for efficiency)
        if dones.any():
            reset_mask = dones
            num_resets = reset_mask.sum().item()

            self.extras["log"] = {
                "episode_count": float(num_resets),
                "waypoints_reached_sum": float(self.episode_waypoints_reached[reset_mask].float().sum()),
                "plant_collisions_sum": float(self.episode_plant_collisions[reset_mask].float().sum()),
                "out_of_bounds_sum": float(self.episode_out_of_bounds[reset_mask].float().sum()),
                "timeouts_sum": float(self.episode_timeouts[reset_mask].float().sum()),
                "command_steps_sum": float(self.episode_command_steps_taken[reset_mask].float().sum()),
                "steps_between_commands_sum": float(self.episode_total_steps_with_commands[reset_mask].float().sum()),
            }
        else:
            self.extras["log"] = {}

        return taskFailed, taskCompleted

    def _get_rewards(self) -> torch.Tensor:
        """Compute multi-component reward signal.

        Reward components:
        - Waypoint reaching: Large positive reward for progress
        - Velocity alignment: Reward for moving toward waypoint
        - Distance penalty: Encourage approaching waypoint
        - Command index alignment: Penalize misaligned buffer state
        - Termination penalties: Out of bounds, collisions, timeout
        - Command step penalty: Optional cost for buffer advancement

        Returns:
            Total reward tensor scaled by cfg.total_reward_scale
        """

        waypoints_reached_this_step = self.waypoints.getReward().float()
        waypointReward = (
            waypoints_reached_this_step * self.cfg.waypoint_reward_base / self.waypoints.waypointsPerRow
        )
        self.episode_waypoints_reached += waypoints_reached_this_step.long()
        self.waypoints.resetRewardBuffer()

        # Compute velocity reward (scalar projection toward waypoint)
        toWaypoint = self.waypoints.robotsdiffs
        toWaypointNorm = torch.norm(toWaypoint, dim=1, keepdim=True) + 1e-6
        toWaypointDir = toWaypoint / toWaypointNorm

        if self.robot_pose is None:
            self.robot_pose = self.robots.data.root_state_w[:, :2]
        if self.past_robot_pose is None:
            self.past_robot_pose = self.robot_pose

        dt = self.cfg.sim.dt if self.cfg.sim.dt > 0 else 1e-6
        velocity = (self.robot_pose - self.past_robot_pose) / dt
        velocityTowardsWaypoint = torch.sum(velocity * toWaypointDir, dim=1)
        velocityTowardsWaypoint *= self.cfg.velocity_towards_waypoint_scale

        # Termination penalties
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeOutPenalty = time_out.float() * self.cfg.timeout_penalty
        plantCollisionPenalty = self.plant_collision_buffer.float() * self.cfg.plant_collision_penalty
        self.plant_collision_buffer = torch.zeros(self.num_envs, device=self.device)

        outOfBoundsPenalty = self.waypoints.robotTooFarFromWaypoint.float() * self.cfg.out_of_bounds_penalty
        commandStepPenalty = self.cfg.command_step_penalty * self.command_step_buffer.float()

        # Command buffer alignment penalty
        rightIndex = torch.clip(
            torch.clip(self.waypoints.currentWaypointIndices[:, 1] - 0, min=0) // (self.waypoints.waypointsPerRow + 1),
            max=self.cfg.commands_length - 1
        )
        commandIndices = self.commandBuffer.indexBuffer
        commandIndexDiff = torch.abs(rightIndex - commandIndices)
        commandIndexPenalty = self.cfg.command_index_penalty_scale * commandIndexDiff.float()

        # Reset command step buffer after reward computation (due to decimation)
        self.command_step_buffer[:] = False

        # Distance-based penalty (encourage approaching waypoint)
        distancePenalty = -torch.clamp(
            toWaypointNorm.squeeze() - self.cfg.distance_penalty_threshold, min=0
        ) * self.cfg.distance_penalty_scale

        totalReward = (
            waypointReward
            + velocityTowardsWaypoint
            + timeOutPenalty
            + plantCollisionPenalty
            + outOfBoundsPenalty
            + commandStepPenalty
            + distancePenalty
            + commandIndexPenalty
        )

        return totalReward * self.cfg.total_reward_scale

    def _get_observations(self) -> dict:
        """Construct observation dictionary for policy.

        Observations include:
        - Current steering angle (1D)
        - LIDAR readings (40D or configured number)
        - Command buffer state (3D: current command, parity, buffer state)
        - Past actions (3+ D: steering, throttle, step_command, hidden states)

        Returns:
            Dictionary with 'policy' key containing concatenated observations
        """

        robot_quat = self.robots.data.root_state_w[:, 3:7]
        robotEuler = quat2axis(robot_quat)
        robotZs = robotEuler[:, [2]]
        lidar = self.plants.computeDistancesToPlants(self.robot_pose, robotZs)

        if self.pastLidar is None:
            self.pastLidar = lidar

        currentCommands = self.commandBuffer.getCurrentCommands()

        # Concatenate all observation components
        obs_components = [
            self.steering_buffer[:, [0]],
            lidar,
            currentCommands,
            self.past_actions,
        ]

        obs = torch.cat(obs_components, dim=-1)
        observations = {"policy": obs}
        self.pastLidar = lidar

        return observations

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments to initial state.

        Resets:
        - Robot joint positions and velocities
        - Robot root state (position, orientation, velocities)
        - Path generation (procedural crop rows)
        - Plant positions
        - Command buffer
        - Waypoints
        - All episode tracking buffers

        Args:
            env_ids: Indices of environments to reset, or None for all environments
        """
        if env_ids is None:
            env_ids = self.robots._ALL_INDICES
        super()._reset_idx(env_ids)

        # Convert to tensor if needed for indexing
        env_ids_tensor = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids

        self.reset_buffers(env_ids_tensor)

        joint_pos = self.robots.data.default_joint_pos[env_ids_tensor]
        joint_vel = self.robots.data.default_joint_vel[env_ids_tensor]
        default_root_state = self.robots.data.default_root_state[env_ids_tensor]
        default_root_state[:, :3] += self.scene.env_origins[env_ids_tensor]

        # Optional yaw randomization
        if self.cfg.randomize_yaw:
            yaw_rand = torch.rand((env_ids_tensor.shape[0], 1), device=self.device) > 0.5
            yaw = yaw_rand.float() * math.pi
            quat = torch.zeros((env_ids_tensor.shape[0], 4), device=self.device)
            quat[:, 0] = torch.cos(yaw[:, 0] / 2)
            quat[:, 3] = torch.sin(yaw[:, 0] / 2)
            default_root_state[:, 3:7] = quat
            self.past_actions[env_ids_tensor, -1] = yaw_rand.squeeze(-1).float()

        self.joint_pos[env_ids_tensor] = joint_pos
        self.joint_vel[env_ids_tensor] = joint_vel

        self.robots.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robots.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robots.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Regenerate scene elements
        self.paths.generatePath()
        self.plants.randomizePlantsPositions(env_ids_tensor, self.paths)
        self.commandBuffer.randomizeCommands(env_ids)
        self.waypoints.resetWaypoints(env_ids)
        self.waypoints.resetRewardBuffer()

    def reset_buffers(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset all episode tracking buffers for specified environments.

        Resets:
        - Past actions buffer (control + hidden states)
        - Hidden state accumulators
        - Steering buffer
        - Robot pose tracking
        - Collision and bounds buffers
        - Episode statistics
        - Past LIDAR readings

        Args:
            env_ids: Indices of environments to reset (Sequence or Tensor)
        """
        # Reset past actions buffer (all actions: 3 control + hidden states)
        self.past_actions[env_ids, :3] = 0.0
        self.past_actions[env_ids, 3:] = 0.0  # Hidden state accumulators start at 0 (neutral)

        # Reset hidden state accumulators (differential integrators)
        if self.num_hidden_states > 0:
            self.hidden_state_accumulators[env_ids] = 0.0

        # Reset steering buffer
        self.steering_buffer[env_ids] = 0.0

        # Reset robot pose
        self.robot_pose[env_ids] = self.scene.env_origins[env_ids, :2]
        self.past_robot_pose[env_ids] = self.scene.env_origins[env_ids, :2]

        # Reset collision and bounds buffers
        self.plant_collision_buffer[env_ids] = 0
        self.out_of_bound_buffer[env_ids] = 0

        # Reset command step tracking
        self.command_step_buffer[env_ids] = False
        self.steps_since_last_command_step[env_ids] = 0
        self.episode_command_steps_taken[env_ids] = 0
        self.episode_total_steps_with_commands[env_ids] = 0

        # Reset sensor and action buffers
        self.pastLidar[env_ids] = 0.0
        self.actions[env_ids] = 0.0

        # Reset episode statistics
        self.episode_waypoints_reached[env_ids] = 0
        self.episode_plant_collisions[env_ids] = 0
        self.episode_out_of_bounds[env_ids] = 0
        self.episode_timeouts[env_ids] = 0
