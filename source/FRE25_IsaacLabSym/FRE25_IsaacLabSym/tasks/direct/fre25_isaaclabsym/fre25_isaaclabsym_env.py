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
from .CommandBuffer.CommandMarkerVisualizer import CommandBufferVisualizer

# import torch.autograd.profiler as profiler
import isaaclab.sim.schemas as schemas
from isaaclab.utils.math import axis_angle_from_quat as quat2axis

import matplotlib.pyplot as plt
from gymnasium.spaces import Box, MultiDiscrete


class Fre25IsaaclabsymEnv(DirectRLEnv):
    cfg: Fre25IsaaclabsymEnvCfg

    def __init__(
        self, cfg: Fre25IsaaclabsymEnvCfg, render_mode: str | None = None, **kwargs
    ):
        # Action space includes control actions + hidden state actions
        # Control actions: steering (3 cats), throttle (3 cats), step_command (2 cats)
        print(f"[ENV __init__] cfg.action_space = {cfg.action_space}")
        print(f"[ENV __init__] cfg.num_hidden_states = {getattr(cfg, 'num_hidden_states', 'NOT SET')}")

        cfg.nActions = cfg.action_space  # Total actions (base control + hidden states)

        # Discrete action space with mixed categories:
        # - steering: 3 categories {-1, 0, 1}
        # - throttle: 3 categories {-1, 0, 1}
        # - step_command: 2 categories {0, 1}
        # - hidden states: 2 categories {-1, 1} each (binary actions)
        num_hidden = getattr(cfg, 'num_hidden_states', 0)
        action_categories = [3, 3, 2] + [2] * num_hidden  # steering, throttle, step_cmd, hidden...
        cfg.action_space = MultiDiscrete(action_categories)  # type: ignore

        print(f"[ENV __init__] Created MultiDiscrete with {cfg.nActions} actions:")
        print(f"[ENV __init__]   - Steering: 3 categories {{-1, 0, 1}}")
        print(f"[ENV __init__]   - Throttle: 3 categories {{-1, 0, 1}}")
        print(f"[ENV __init__]   - Step Command: 2 categories {{0, 1}}")
        if num_hidden > 0:
            print(f"[ENV __init__]   - Hidden States: {num_hidden} × 2 categories {{-1, 1}} (binary)")

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
        print("SETUP SCENE")
        self.robots: Articulation = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(size=(10000000, 10000000))
        )

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add articulation to scene
        self.scene.articulations["robot"] = self.robots

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Add waypoint markers to the scene
        self.waypoint_markers = VisualizationMarkers(WAYPOINT_CFG)

        # Initialize Command Visualizer
        self.command_visualizer = CommandBufferVisualizer()

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
            pathsSpacing=1.2,
            nControlPoints=10,
            pathLength=4,
            pathWidth=0.15,
            pointNoiseStd=0.03,
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
            endOfRowPadding=0.8,
            waypointsPerRow=10,
        )
        self.waypoints.initializeWaypoints()

        # Add Plants to the scene
        self.plants = PlantHandler(
            nPlants=60,
            envsOrigins=self.scene.env_origins,
            plantRadius=0.22,
        )

        self.plants.spawnPlants()

        # buffer for past lidar readings
        self.pastLidar = torch.zeros(
            (self.num_envs, self.plants.raymarcher.raysPerRobot), device=self.device
        )

        # Initialize buffer for ALL past actions (control + hidden states)
        # Full buffer stored internally for edge detection and tracking
        # BUT only step_command + hidden states passed to observations (steering/throttle excluded)
        self.num_hidden_states = getattr(self.cfg, "num_hidden_states", 0)
        self.num_control_actions = 3  # steering, throttle, step_command
        self.num_total_actions = self.num_control_actions + self.num_hidden_states

        print(f"[ENV] 🧠 Initializing buffer for {self.num_total_actions} past actions (internal)")
        print(f"[ENV]    - {self.num_control_actions} control actions (steering, throttle, step_command)")
        print(f"[ENV]    - {self.num_hidden_states} hidden state actions (memory)")
        print(f"[ENV]    - Policy observes: step_command + {self.num_hidden_states} hidden (steering/throttle excluded)")

        # Buffer stores ALL past actions to be fed back as observations
        self.past_actions = torch.zeros(
            (self.num_envs, self.num_total_actions), device=self.device
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

        # LSTM policy handles temporal information - no manual hidden state needed

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # print("PRE PHYSICS STEP")
        # Update robot position and waypoint detection
        self.past_robot_pose = self.robot_pose

        # Convert discrete action indices to continuous values
        # Action 0 (steering): {0, 1, 2} → {-1, 0, +1}
        # Action 1 (throttle): {0, 1, 2} → {-1, 0, +1}
        # Action 2 (step_command): {0, 1} → {0, 1} (keep as is, then convert to bool)
        # Actions 3+ (hidden): {0, 1} → {-1, +1} (binary)

        # Convert actions - handle step_command and hidden states differently
        converted_actions = actions.clone().float()
        # Steering and throttle: subtract 1 to get {-1, 0, 1}
        converted_actions[:, 0] = actions[:, 0] - 1  # steering
        converted_actions[:, 1] = actions[:, 1] - 1  # throttle
        # step_command stays as {0, 1}
        converted_actions[:, 2] = actions[:, 2]
        # Hidden states: convert {0, 1} → {-1, +1}
        if self.num_hidden_states > 0:
            converted_actions[:, 3:] = actions[:, 3:] * 2 - 1  # 0→-1, 1→+1

        self.actions = converted_actions
        self.waypoints.visualizeWaypoints()
        self.waypoints.updateCurrentMarker()

        self.command_visualizer.visualizeCommands(self.robots.data.root_state_w[:, :3], self.commandBuffer)

    def _apply_action(self) -> None:
        # Update robot position and waypoint detection at each action step
        self.robot_pose = self.robots.data.root_state_w[:, :2]
        self.waypoints.updateCurrentDiffs(self.robot_pose)

        # print("APPLY ACTION")
        if not hasattr(self, "actions"):
            return
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

        # Agent-controlled command buffer stepping via step_command action
        # Extract step_command action (binary: 0 or 1)
        step_command_action = self.actions[:, 2]
        step_command = (step_command_action > 0.5).bool()  # Convert to boolean

        # Rising edge detection: step only when transitioning from 0 → 1
        # Get previous step_command from past_actions buffer (index 2)
        past_step_command = (self.past_actions[:, 2] > 0.5).bool()
        rising_edge = step_command & (~past_step_command)

        # Step command buffer for environments with rising edge
        # Note: Don't reset the buffer here because _apply_action is called multiple times
        # due to decimation (4 times). The buffer will be reset in _get_rewards() after
        # the reward is computed.
        if rising_edge.any():
            env_ids_to_step = torch.where(rising_edge)[0]
            self.commandBuffer.stepCommands(env_ids_to_step)
            # Mark that these environments took a command step
            self.command_step_buffer[env_ids_to_step] = True

            # Track steps between command steps for metric logging
            # Add the accumulated steps to the total
            self.episode_total_steps_with_commands[env_ids_to_step] += self.steps_since_last_command_step[env_ids_to_step]
            # Increment count of command steps taken
            self.episode_command_steps_taken[env_ids_to_step] += 1
            # Reset the step counter for these environments
            self.steps_since_last_command_step[env_ids_to_step] = 0

        # Increment step counter for all environments (tracks steps since last command step)
        self.steps_since_last_command_step += 1

        # Store ALL actions (control + hidden) in internal buffer for edge detection and tracking
        # Note: Only step_command + hidden states are passed to policy observations
        # Control actions: steering, throttle, step_command (now 3 actions)
        # The step_command is stored as {0, 1} to match what the agent outputs
        # Hidden states: remain as {-1, 0, 1}
        all_actions = self.actions[:, :self.num_total_actions].clone()
        # Normalize control actions to [-1, 1] range for consistency
        all_actions[:, 0:2] = torch.clamp(all_actions[:, 0:2], -1, 1)  # steering, throttle
        all_actions[:, 2] = torch.clamp(all_actions[:, 2], 0, 1)  # step_command (binary)
        if self.num_hidden_states > 0:
            all_actions[:, 3:] = torch.clamp(all_actions[:, 3:], -1, 1)  # hidden states
        self.past_actions = all_actions

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # print("GET DONES")
        DEBUG = False
        # The episode has reached the maximum length
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Track timeout occurrences
        self.episode_timeouts += time_out.long()
        if DEBUG and any(time_out):
            print(f"Episode length reached: {self.max_episode_length} steps")

        # the robot has reached all the waypoints
        reached_all_waypoints = self.waypoints.taskCompleted
        if DEBUG and any(reached_all_waypoints):
            print("Task completed: reached all waypoints")

        # The robot is too far from the current waypoint
        out_of_bounds = self.waypoints.robotTooFarFromWaypoint
        self.out_of_bound_buffer = out_of_bounds
        # Track out of bounds occurrences
        self.episode_out_of_bounds += out_of_bounds.long()
        if DEBUG and any(out_of_bounds):
            print("Episode terminated: robot out of bounds")

        # Check for plant collisions
        plant_collisions = self.plants.detectPlantCollision()
        self.plant_collision_buffer = plant_collisions
        # Track plant collision occurrences
        self.episode_plant_collisions += plant_collisions.long()
        if DEBUG and any(plant_collisions):
            print("Episode terminated: robot collided with a plant")

        # Completed command buffer
        completed_command_buffer = self.commandBuffer.dones()
        #######################
        #########################
        ########################
        completed_command_buffer = torch.zeros_like(completed_command_buffer)
        if DEBUG and any(completed_command_buffer):
            print("Task Completed: completed command buffer")

        taskCompleted = reached_all_waypoints | time_out | completed_command_buffer
        taskFailed = out_of_bounds | plant_collisions

        # Determine which environments are resetting (completing or failing)
        dones = taskCompleted | taskFailed

        # Log episode statistics to extras for TensorBoard
        # Store ONLY sum and count for resetting environments (no detailed values for efficiency)
        # The callback will use these to compute proper per-episode statistics
        if dones.any():
            reset_mask = dones
            num_resets = reset_mask.sum().item()

            # Store sum and count (count is shared across all metrics)
            # No detailed per-episode value lists - too expensive for large batch sizes
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
            # No environments resetting this step
            self.extras["log"] = {}

        return taskFailed, taskCompleted

    def _get_rewards(self) -> torch.Tensor:
        # print("GET REWARDS")
        # Robot position and waypoint detection already updated in _pre_physics_step

        # Reward for staying alive
        # stayAliveReward = (1.0 - self.reset_terminated.float()) / 100000  # (n_envs)

        waypoints_reached_this_step = self.waypoints.getReward().float()

        # Reward for reaching waypoints
        waypointReward = (
            waypoints_reached_this_step * 100 / self.waypoints.waypointsPerRow
        )  # (n_envs,)

        # Track waypoints reached for episode statistics
        self.episode_waypoints_reached += waypoints_reached_this_step.long()

        self.waypoints.resetRewardBuffer()

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
        velocityTowardsWaypoint *= 0.1

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
        timeOutPenalty = time_out.float() * -1000  # -50

        # Penalty for plant collisions
        plantCollisionPenalty = self.plant_collision_buffer.float() * -0  # -50
        self.plant_collision_buffer = torch.zeros(self.num_envs, device=self.device)

        # Out of bounds penalty
        out_of_bounds = self.waypoints.robotTooFarFromWaypoint
        outOfBoundsPenalty = out_of_bounds.float() * -0  # -50

        # Penalty for performing a command step
        # This encourages the agent to be selective about when to advance the command buffer
        commandStepPenalty = -self.command_step_buffer.float()

        # Reset command step buffer after computing the reward
        # This is done here (not in _apply_action) because _apply_action is called
        # multiple times per step due to decimation (4 times), but _get_rewards is
        # called once per step, so the penalty is applied correctly.
        self.command_step_buffer[:] = False

        # Note: No action bound violation penalty for discrete actions

        totalReward = (
            waypointReward
            + velocityTowardsWaypoint
            # + velocityOrthogonalToWaypoint
            + timeOutPenalty
            + plantCollisionPenalty
            + outOfBoundsPenalty
            + commandStepPenalty
        )
        # print(f"totalReward: {totalReward}")

        return totalReward / 10

    def _get_observations(self) -> dict:
        # print("GET OBSERVATIONS")
        # Robot pose and waypoint updates already done in _pre_physics_step

        robot_quat = self.robots.data.root_state_w[:, 3:7]
        robotEuler = quat2axis(robot_quat)
        robotZs = robotEuler[:, [2]]  # Extract the Z component of the Euler angles
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

        # Observations include:
        # 1. Current steering angle (1 dim) - from steering_buffer
        # 2. Lidar readings (40 dims)
        # 3. Current commands (3 dims)
        # 4. Past actions (memory): ONLY step_command + hidden states (1 + N dims)
        #    Note: Steering/throttle excluded as steering_buffer provides state
        obs_components = [
            self.steering_buffer[:, [0]],  # The Current Steering Angle
            lidar,  # The Current Lidar Readings
            currentCommands,  # The Current Commands
            self.past_actions[:, 2:],  # Only step_command + hidden states (skip steering & throttle)
        ]

        obs = torch.cat(obs_components, dim=-1)
        observations = {"policy": obs}

        self.pastLidar = lidar

        return observations

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # print("RESET IDX")
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

        # Reset waypoint reward buffer
        self.waypoints.resetRewardBuffer()

        # Reset past actions buffer (all actions: 3 control + hidden states)
        self.past_actions[env_ids] = 0.0

        # RESET BUFFERS

        # Reset steering buffer
        self.steering_buffer[env_ids] = 0.0

        # Reset robot pose
        self.robot_pose[env_ids] = self.scene.env_origins[env_ids, :2]

        # Reset past robot pose
        self.past_robot_pose[env_ids] = self.scene.env_origins[env_ids, :2]

        # Remove obsolete past_command_actions (no longer used)

        # Reset plant collision buffer
        self.plant_collision_buffer[env_ids] = 0

        # Reset out of bound buffer
        self.out_of_bound_buffer[env_ids] = 0

        # Reset command step buffer
        self.command_step_buffer[env_ids] = False

        # Reset command step tracking buffers
        self.steps_since_last_command_step[env_ids] = 0
        self.episode_command_steps_taken[env_ids] = 0
        self.episode_total_steps_with_commands[env_ids] = 0

        # Reset past lidar
        self.pastLidar[env_ids] = 0.0

        # Reset actions to 0.0 (which corresponds to discrete index 1 after conversion)
        self.actions[env_ids] = 0.0

        # Reset episode statistics
        self.episode_waypoints_reached[env_ids] = 0
        self.episode_plant_collisions[env_ids] = 0
        self.episode_out_of_bounds[env_ids] = 0
        self.episode_timeouts[env_ids] = 0
