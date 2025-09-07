import torch
from collections.abc import Sequence
from .Waypoint import WAYPOINT_CFG
from isaaclab.markers import VisualizationMarkers
from ..CommandBuffer import CommandBuffer
from ..PathHandler import PathHandler


class WaypointHandler:
    def __init__(
        self,
        nEnvs: int,
        envsOrigins: torch.Tensor,
        lineLength: float = 10,
        lineWidth: float = 1.0,
        lineZ: float = 0.0,
        waipointReachedEpsilon: float = 0.5,
        maxDistanceToWaypoint: float = 1.5,
        commandBuffer: CommandBuffer = None,
        pathHandler: PathHandler = None
    ):
        assert commandBuffer is not None, "commandBuffer must be provided"
        self.commandBuffer = commandBuffer

        assert pathHandler is not None, "pathHandler must be provided"
        self.pathHandler = pathHandler

        assert nEnvs > 0, "Number of environments must be greater than 0, got {}".format(nEnvs)
        self.nEnvs = nEnvs

        # The first row of plants adds 2 waypoints (start and end), each command adds 2 waypoints (landing row start and end)
        self.nWaypoints = (self.commandBuffer.commandsLength + 1) * 2

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
            (nEnvs, self.nWaypoints, 3), dtype=torch.float32, device=envsOrigins.device
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

        # Buffer to know if an agent has reached a waypoint at the current step
        self.waypointReachedBuffer = torch.zeros(
            (nEnvs), dtype=torch.bool, device=envsOrigins.device
        )

    def initializeWaypoints(self):
        # Initialize waypoints in a straight line
        waypointsX = torch.zeros((self.nEnvs, self.nWaypoints), device=self.waypointsPositions.device)

        # Since the robot goes forward, backward, forward, backward... the X positions of the waypoints are always the same
        # and they alternate between 0 and lineLength as 0, l, l, 0, 0, l, l, 0...
        waypointsX[:, 0::4] = 0  # even waypoints are the start of each row
        waypointsX[:, 1::4] = self.pathHandler.pathLength  # odd waypoints are the end of each row
        waypointsX[:, 2::4] = self.pathHandler.pathLength  # even waypoints are the start of each row
        waypointsX[:, 3::4] = 0  # odd waypoints are the end of each row

        waypointsY = torch.zeros((self.nEnvs, self.nWaypoints), device=self.waypointsPositions.device)
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

    def setWaypointsFromPathAndCommands(self, env_ids: Sequence[int]):
        pathWidth = self.pathHandler.pathsSpacing

        turnsMagnitudes = self.commandBuffer.commandBuffer[env_ids]  # (len(env_ids), commandsLength)
        turnDirections = self.commandBuffer.turnRightBuffer[env_ids]  # (len(env_ids), commandsLength)

        # Regardless of the commands the waypoints X positions are always the same

        # turnDirections is a bool tensor relative to the robot's frame of reference (is it right?)
        # now I want to convert it in a global Y sign (+Y or -Y)
        # To do this, since the robot always goes forward, backward, forward, backward... I can just invert the oddly numbered commands
        globalTurnDirections = turnDirections.clone()
        globalTurnDirections[:, 1::2] = ~globalTurnDirections[:, 1::2]

        # Now I can convert it the bool (is it -Y?) to a float (-1 or 1)
        globalTurnDirectionsSign = -(globalTurnDirections.float() * 2 - 1)  # convert to -1 and 1

        # Now I can compute the delta Y for each command by multiplying the turnsMagnitudes by the globalTurnDirectionsSign
        deltaY = turnsMagnitudes * globalTurnDirectionsSign * pathWidth  # (len(env_ids), commandsLength)

        # Since the robot starts at (0,0), I can compute the Y positions of the rows by cumulatively summing the deltaY
        rowsY = torch.cumsum(deltaY, dim=1)  # (len(env_ids), commandsLength)

        # Now I can compute the waypoints Y positions
        waypointsY = torch.zeros((len(env_ids), self.nWaypoints), device=self.waypointsPositions.device)
        waypointsY[:, 0:2] = 0  # first two waypoints have Y=0 since the robot starts at (0,0)
        waypointsY[:, 2::2] = rowsY  # even waypoints are the start of each row
        waypointsY[:, 3::2] = rowsY  # odd waypoints are the end of each row

        # add the environment origins to the waypoints
        waypointsY += self.envsOrigins[env_ids, :, 1]

        # Update the y coordinates of the waypoints
        self.waypointsPositions[env_ids, :, 1] = waypointsY

    def resetWaypoints(self, env_ids: Sequence[int]):
        # Randomize the waypoints for the given environment ids
        self.setWaypointsFromPathAndCommands(env_ids)

        # Reset the current waypoint indices for the given environment ids
        self.currentWaypointIndices[env_ids, 1] = 0

        # Reset the current waypoint positions for the given environment ids
        self.currentWaypointPositions[env_ids] = self.waypointsPositions[
            self.currentWaypointIndices[env_ids, 0],
            self.currentWaypointIndices[env_ids, 1],
        ]

        # reset the task completed status for the given environment ids
        self.taskCompleted[env_ids] = False

        # reset the waypoint reached status for the given environment ids
        self.waypointReachedBuffer[env_ids] = False

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
        For each environment, compute the difference between the current waypoint position and the robot position.

        Args:
            robot_pos_xy (torch.Tensor): The robot position in the xy plane. Shape: (nEnvs, 2)
        Returns:
            torch.Tensor: The difference between the robot position and the current waypoint position. Shape: (nEnvs, 2)
        """
        # assert robot_pos_xy.shape == (self.nEnvs, 2), "robot_pos_xy must be of shape (nEnvs, 2), but got {}".format(robot_pos_xy.shape)
        # assert currentWaypointsPositions.shape == (self.nEnvs, 2), "currentWaypointsPositions must be of shape (nEnvs, 2) but got {}".format(currentWaypointsPositions.shape)
        diff = self.currentWaypointPositions[:, :2] - robot_pos_xy
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

        # Update the waypoint reached buffer
        self.waypointReachedBuffer = waypointReached

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

    def getReward(self) -> torch.Tensor:
        tmp = self.waypointReachedBuffer
        self.waypointReachedBuffer = torch.zeros_like(self.waypointReachedBuffer)
        return tmp
