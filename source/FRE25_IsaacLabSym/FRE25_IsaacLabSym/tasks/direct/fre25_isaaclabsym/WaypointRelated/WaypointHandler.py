import torch
from collections.abc import Sequence
from .Waypoint import WAYPOINT_CFG
from isaaclab.markers import VisualizationMarkers


class WaypointHandler:
    def __init__(
        self,
        nEnvs: int,
        envsOrigins: torch.Tensor,
        nWaypoints: int = 10,
        lineLength: float = 10,
        lineWidth: float = 1.0,
        lineZ: float = 0.0,
        waipointReachedEpsilon: float = 0.5,
        maxDistanceToWaypoint: float = 1.5,
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
        # Whether the robot has completed all the waypoints for each environment
        self.taskCompleted = torch.zeros(
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

        # reset the task completed status for the given environment ids
        self.taskCompleted[env_ids] = False

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
        notAtLastWaypoint = self.currentWaypointIndices[:, 1] < self.nWaypoints - 1
        waypointReachedAndNotAtLast = waypointReached & notAtLastWaypoint
        self.currentWaypointIndices[waypointReachedAndNotAtLast, 1] += 1

        # Update the current waypoint position for each environment
        self.currentWaypointPositions[waypointReachedAndNotAtLast] = self.waypointsPositions[
            self.currentWaypointIndices[waypointReachedAndNotAtLast, 0],
            self.currentWaypointIndices[waypointReachedAndNotAtLast, 1],
        ]

        self.taskCompleted = waypointReached & (~notAtLastWaypoint)

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
