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
        pathHandler: PathHandler = None,
        endOfRowPadding: float = 1.0,
        extraWaypointPadding: float = 0.5,
        waypointsPerRow: int = 8,
    ):
        assert commandBuffer is not None, "commandBuffer must be provided"
        self.commandBuffer = commandBuffer

        assert pathHandler is not None, "pathHandler must be provided"
        self.pathHandler = pathHandler

        assert endOfRowPadding >= 0, "endOfRowPadding must be greater than 0"
        self.endOfRowPadding = endOfRowPadding

        assert extraWaypointPadding >= 0, "extraWaypointPadding must be greater than 0"
        self.extraWaypointPadding = extraWaypointPadding

        assert (
            nEnvs > 0
        ), "Number of environments must be greater than 0, got {}".format(nEnvs)
        self.nEnvs = nEnvs

        assert (
            waypointsPerRow >= 2
        ), "waypointsPerRow must be greater than 2, got {}".format(waypointsPerRow)
        self.waypointsPerRow = waypointsPerRow

        # Number of rows = number of commands (stops after last turn, no extra row)
        self.nRows = self.commandBuffer.commandsLength

        # Waypoints calculation:
        # First row: waypointsPerRow-1 (skip start), remaining rows: (nRows-1)*waypointsPerRow
        # Plus one extra waypoint after EACH row EXCEPT the last for turn guidance
        self.nWaypoints = self.nRows * self.waypointsPerRow - 1 + (self.nRows - 1)

        assert envsOrigins.shape == (
            nEnvs,
            3,
        ), "envsOrigins must be of shape (nEnvs, 3), but got {}".format(
            envsOrigins.shape
        )
        self.envsOrigins: torch.Tensor = envsOrigins.unsqueeze(1).repeat(
            1, self.nWaypoints, 1
        )

        assert lineLength >= 0, "Line length must be greater than 0, got {}".format(
            lineLength
        )
        self.lineLength = lineLength

        assert lineWidth >= 0, "Line width must be greater than 0, got {}".format(
            lineWidth
        )
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

        # Since the robot goes forward, backward, forward, backward... the X positions of the waypoints are always the same
        # In-row waypoints go from 0 to pathLength (no padding)
        # Extra waypoints at row ends have padding to give space for turning
        device = self.waypointsPositions.device

        # Create in-row waypoints (no padding)
        rowXs = torch.linspace(
            -self.endOfRowPadding,
            self.pathHandler.pathLength + self.endOfRowPadding,
            self.waypointsPerRow,
            device=device,
        )
        doubleRowXs = torch.cat((rowXs, torch.flip(rowXs, [0])), dim=0)

        envWaypointsX = torch.tile(doubleRowXs, (self.nRows // 2,))
        if self.nRows % 2 == 1:
            envWaypointsX = torch.cat((envWaypointsX, rowXs))
        envWaypointsX = envWaypointsX[1:]  # remove the first waypoint (it is at 0)

        # Insert extra waypoints at the end of each row EXCEPT the last for turn guidance
        # Extra waypoints have X with padding: either -padding or pathLength+padding
        n_original = len(envWaypointsX)
        n_extras = self.nRows - 1  # Extra waypoint after each row except the last

        # Create tensor with space for extras
        envWaypointsX_with_extras = torch.zeros(n_original + n_extras, device=device)

        # Calculate where each extra should be inserted
        # After removing first waypoint with envWaypointsX[1:], the structure is:
        # Row 0: indices [0 to waypointsPerRow-3] → (waypointsPerRow-2) waypoints? NO!
        # Wait, let me recalculate...
        #
        # Original before [1:]: waypointsPerRow waypoints per "period", nRows periods, minus 1
        # After [1:]: one less waypoint overall
        #
        # Actually the logic is:
        # - doubleRowXs has 2*waypointsPerRow elements
        # - We tile it nRows//2 times and possibly add one more rowXs
        # - Then remove the first element
        #
        # So for waypointsPerRow=10, nRows=4:
        # doubleRowXs has 20 elements
        # tile 2 times = 40 elements
        # remove first = 39 elements
        # Row 0: 0-8 (9 elements, ending at 8 = waypointsPerRow-2)
        # Row 1: 9-18 (10 elements, ending at 18 = 2*waypointsPerRow-2)
        # Row 2: 19-28 (10 elements, ending at 28 = 3*waypointsPerRow-2)
        # Row 3: 29-38 (10 elements, ending at 38 = 4*waypointsPerRow-2)
        #
        # So the last waypoint of row i is at: (i+1)*waypointsPerRow - 2
        # Now we insert extras after rows 0,1,2,...,nRows-2 (all except the last)

        # In the original array (before insertions), the last waypoint of row i is at:
        original_row_end_indices = (
            torch.arange(n_extras, device=device) + 1
        ) * self.waypointsPerRow - 2

        # In the new array (with insertions), each extra shifts subsequent indices by 1
        # Extra after row 0 goes at position: row_0_end + 1
        # Extra after row 1 goes at position: row_1_end + 1 + 1 (shifted by previous extra)
        # Extra after row i goes at position: row_i_end + 1 + i
        extra_positions_in_new = original_row_end_indices + torch.arange(
            1, n_extras + 1, device=device
        )

        # Create a mask for extra positions
        is_extra = torch.zeros(n_original + n_extras, dtype=torch.bool, device=device)
        is_extra[extra_positions_in_new] = True

        # Place original waypoints in non-extra positions
        envWaypointsX_with_extras[~is_extra] = envWaypointsX

        # Place extra waypoints with padding (vectorized)
        # Determine which direction each row goes: even rows go forward (end at pathLength+padding), odd rows go backward (end at -padding)
        # Row 0 ends at pathLength, so extra after row 0 should be at pathLength + padding
        # Row 1 ends at 0, so extra after row 1 should be at -padding
        row_indices = torch.arange(n_extras, device=device)
        is_forward_row = row_indices % 2 == 0  # Even rows go forward

        extra_X_values = torch.where(
            is_forward_row,
            self.pathHandler.pathLength + self.extraWaypointPadding,
            -self.extraWaypointPadding,
        )

        envWaypointsX_with_extras[is_extra] = extra_X_values

        # Use the new waypoints array
        envWaypointsX = envWaypointsX_with_extras

        waypointsX = envWaypointsX.unsqueeze(0).repeat(self.nEnvs, 1)

        waypointsY = torch.zeros(
            (self.nEnvs, self.nWaypoints), device=self.waypointsPositions.device
        )
        waypointsZ = torch.full((self.nEnvs, self.nWaypoints), self.lineZ)

        self.waypointsPositions[:, :, 0] = waypointsX
        self.waypointsPositions[:, :, 1] = waypointsY
        self.waypointsPositions[:, :, 2] = waypointsZ

        # Add the environment origins to the waypoints
        self.waypointsPositions += self.envsOrigins

        self.markersVisualizer = VisualizationMarkers(WAYPOINT_CFG)

        # Select the current waypoint for each environment
        env_indices = torch.arange(
            self.nEnvs, device=self.currentWaypointIndices.device
        )
        self.currentWaypointPositions = self.waypointsPositions[
            env_indices, self.currentWaypointIndices[:, 1]
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

        turnsMagnitudes = self.commandBuffer.commandBuffer[
            env_ids
        ]  # (len(env_ids), commandsLength)
        turnDirections = self.commandBuffer.turnRightBuffer[
            env_ids
        ]  # (len(env_ids), commandsLength)

        # Regardless of the commands the waypoints X positions are always the same

        # turnDirections is a bool tensor relative to the robot's frame of reference (is it right?)
        # now I want to convert it in a global Y sign (+Y or -Y)
        # To do this, since the robot always goes forward, backward, forward, backward... I can just invert the oddly numbered commands
        globalTurnDirections = turnDirections.clone()
        globalTurnDirections[:, 1::2] = ~globalTurnDirections[:, 1::2]

        # Now I can convert it the bool (is it -Y?) to a float (-1 or 1)
        globalTurnDirectionsSign = -(
            globalTurnDirections.float() * 2 - 1
        )  # convert to -1 and 1

        # Now I can compute the delta Y for each command by multiplying the turnsMagnitudes by the globalTurnDirectionsSign
        deltaY = (
            turnsMagnitudes * globalTurnDirectionsSign * pathWidth
        )  # (len(env_ids), commandsLength)

        # Since the robot starts at (0,0), I can compute the Y positions of the rows by cumulatively summing the deltaY
        rowsY = torch.cumsum(deltaY, dim=1)  # (len(env_ids), commandsLength)

        # Now I can compute the waypoints Y positions (fully vectorized)
        waypointsY = torch.zeros(
            (len(env_ids), self.nWaypoints), device=self.waypointsPositions.device
        )

        # Create a tensor with Y values for each row: first row is 0, rest from rowsY
        # Shape: (len(env_ids), nRows)
        # Note: nRows = commandsLength, so rowsY already has all rows except the first (which is at Y=0)
        allRowsY = torch.cat(
            [torch.zeros(len(env_ids), 1, device=rowsY.device), rowsY], dim=1
        )  # (len(env_ids), nRows+1) = (len(env_ids), commandsLength+1)

        # But we only want nRows rows, not nRows+1, so we need to slice
        # Since nRows = commandsLength, we have one too many rows
        # Actually, wait - let me reconsider the structure:
        # - First row (row 0): starts at Y=0
        # - After command 0, we're at Y=rowsY[:, 0]
        # - After command 1, we're at Y=rowsY[:, 1]
        # ...
        # - After command i, we're at Y=rowsY[:, i]
        #
        # But if nRows = commandsLength, then:
        # - Row 0 is at Y=0
        # - Row 1 is at Y=rowsY[:, 0] (after first command/turn)
        # - Row 2 is at Y=rowsY[:, 1] (after second command/turn)
        # ...
        # - Row i is at Y=rowsY[:, i-1] (after (i-1)-th command/turn)
        #
        # So allRowsY should be: [0, rowsY], which has shape (commandsLength+1)
        # But nRows = commandsLength, so we have one extra!
        #
        # The issue is that nRows = commandsLength means we have commandsLength rows,
        # but the Y position after commandsLength commands gives us commandsLength+1 distinct Y values
        # (initial + commandsLength transitions)
        #
        # Actually, I think the logic should be:
        # If we have commandsLength commands, and we stop after the last turn:
        # - Row 0: Y=0
        # - Turn 1 → Row 1: Y=rowsY[0]
        # - Turn 2 → Row 2: Y=rowsY[1]
        # - ...
        # - Turn commandsLength → Stop at Y=rowsY[commandsLength-1]
        #
        # So we have commandsLength rows (0 through commandsLength-1), not commandsLength+1
        # Therefore allRowsY should be [0, rowsY], but then we only use first nRows elements
        allRowsY = allRowsY[:, :self.nRows]  # (len(env_ids), nRows)

        # Strategy: Create Y values for "logical" structure without extras first,
        # then expand to include extras

        # Create mask for extra positions first (needed for X coordinate extraction)
        device = waypointsY.device
        n_extras = self.nRows - 1  # Extra waypoint after each row except the last
        original_row_end_indices = (
            torch.arange(n_extras, device=device) + 1
        ) * self.waypointsPerRow - 2
        extra_positions_in_new = original_row_end_indices + torch.arange(
            1, n_extras + 1, device=device
        )
        is_extra = torch.zeros(self.nWaypoints, dtype=torch.bool, device=device)
        is_extra[extra_positions_in_new] = True

        # Logical waypoints (without extras): waypointsPerRow-1 per row
        n_logical = self.nRows * self.waypointsPerRow - 1
        logical_waypointsY = torch.zeros(
            (len(env_ids), n_logical), device=waypointsY.device
        )

        # Create row assignment for each logical waypoint (which row does waypoint i belong to?)
        # Row 0: waypoints 0 to waypointsPerRow-2 (waypointsPerRow-1 total)
        # Row i (i>0): waypoints (i*waypointsPerRow-1) to ((i+1)*waypointsPerRow-2) (waypointsPerRow total)
        waypoint_indices = torch.arange(n_logical, device=waypointsY.device)
        # For waypoint index j, its row is: ceil((j+1)/waypointsPerRow)
        # But first row only has waypointsPerRow-1 waypoints, so:
        # j < waypointsPerRow-1: row 0
        # j >= waypointsPerRow-1: row = floor((j+1)/waypointsPerRow)
        row_assignment = torch.zeros(
            n_logical, dtype=torch.long, device=waypointsY.device
        )
        row_assignment[waypoint_indices >= self.waypointsPerRow - 1] = (
            waypoint_indices[waypoint_indices >= self.waypointsPerRow - 1] + 1
        ) // self.waypointsPerRow

        # Assign Y values using advanced indexing
        # For each waypoint, get the Y value of its row
        # allRowsY has shape (len(env_ids), nRows)
        # row_assignment has shape (n_logical,)
        # We want logical_waypointsY[env, waypoint] = allRowsY[env, row_assignment[waypoint]]
        logical_waypointsY = allRowsY[
            :, row_assignment
        ]  # Broadcasting: (len(env_ids), n_logical)

        # Sample Y offsets from spline for logical waypoints to follow path shape
        # Get X coordinates for logical waypoints (non-extra waypoints)
        # X coordinates are already stored in self.waypointsPositions - extract non-extra ones
        all_X = self.waypointsPositions[env_ids, :, 0]  # (len(env_ids), nWaypoints)
        # Remove environment origins to get relative X
        all_X = all_X - self.envsOrigins[env_ids, :, 0]
        logical_X = all_X[:, ~is_extra]  # (len(env_ids), n_logical)

        # Determine which path each waypoint belongs to based on row and commands
        # For each row, we need to figure out which path index to use
        # Row 0 starts at path 0, then each command changes path by turnsMagnitudes
        # Build path indices for each row: shape (len(env_ids), nRows)
        row_path_indices = torch.zeros(
            (len(env_ids), self.nRows), dtype=torch.long, device=logical_X.device
        )
        # First row always starts at middle path (index 0 relative to spline center)
        # Subsequent rows: cumsum of (turnsMagnitudes * globalTurnDirectionsSign)
        # But turnsMagnitudes is in units of "number of paths", and we're working with the spline Y offset
        # The spline gives us base Y, and we add path spacing later, so path_indices here are not needed
        # Actually, we just need the spline Y offset which is the same for all paths at a given X

        # Sample Y offsets from spline (all environments can use the spline for their X positions)
        # The spline provides the Y offset at each X position
        # We need to handle forward/backward rows: backward rows need reversed X sampling

        # Determine forward/backward for each row
        rows_forward = (
            torch.arange(self.nRows, device=logical_X.device) % 2 == 0
        )  # (nRows,)

        # For each logical waypoint, determine if its row is forward or backward
        waypoint_forward = rows_forward[row_assignment]  # (n_logical,)

        # For backward rows, we need to sample the spline in reverse X order
        # Spline t-values for forward: t = X / pathLength
        # Spline t-values for backward: t = (pathLength - X) / pathLength
        logical_X_for_sampling = torch.where(
            waypoint_forward[None, :],  # Broadcast to (len(env_ids), n_logical)
            logical_X,
            self.pathHandler.pathLength - logical_X,
        )

        # Sample Y offsets from spline (vectorized for all environments and waypoints)
        # Convert X to t values [0, 1]
        t_values = logical_X_for_sampling / self.pathHandler.pathLength
        t_values = torch.clamp(t_values, 0.0, 1.0)

        # We need to evaluate the spline for each environment's unique set of t values
        # The spline stores per-environment splines, and evaluate expects a 1D tensor of t values
        # Since each environment may have different t values, we need to evaluate per-environment

        # For now, let's use a vectorized approach by evaluating all unique t values
        # and then indexing. But actually, t_values shape is (len(env_ids), n_logical)
        # For full vectorization, we can flatten, get unique t values, evaluate, then map back

        # Simpler approach: evaluate per environment in a loop (still faster than Python loop due to GPU)
        # But let's try to vectorize: we can evaluate the spline at all t values for all envs
        # by creating a common grid and then selecting the right indices

        # Most vectorized approach: evaluate at specific t for each waypoint across all envs
        # Spline.evaluate(t) where t is (n_points,) returns (nEnvs, n_points, 2)
        # We can evaluate each unique t value, but different waypoints have different t values

        # Practical vectorized solution: evaluate spline for each environment's t_values
        spline_y_offsets = torch.zeros_like(logical_X)
        for i, env_id in enumerate(env_ids):
            # Get t values for this environment
            t_env = t_values[i]  # (n_logical,)
            # Evaluate spline for all t values at once
            spline_eval = self.pathHandler.spline.evaluate(
                t_env
            )  # (nEnvs, n_logical, 2)
            # Extract this environment's values (use env_id as index into full env array)
            spline_y_offsets[i] = spline_eval[env_id, :, 1]  # Y component

        # For backward rows, flip the Y offset sign (robot traveling in opposite direction)
        # Apply sign flip based on whether waypoint is in forward or backward row
        y_offset_sign = torch.where(waypoint_forward, 1.0, -1.0)  # (n_logical,)
        spline_y_offsets = spline_y_offsets * y_offset_sign[None, :]

        # Apply spline offsets to logical waypoints, but skip first waypoint (index 0)
        # Create mask: skip waypoint 0
        offset_mask = (
            torch.arange(n_logical, device=logical_X.device) > 0
        )  # (n_logical,)

        # Apply offsets only where mask is True
        logical_waypointsY = logical_waypointsY + (
            spline_y_offsets * offset_mask[None, :]
        )

        # Now insert extra waypoints with Y values
        # For intermediate turns (after rows 0 to nRows-2): Y = average of current and next row
        # No extra waypoint after the last row
        # Calculate Y values for extra waypoints
        if self.nRows > 1:
            # allRowsY[:, :-1] = current row Y (rows 0 to nRows-2)
            # allRowsY[:, 1:] = next row Y (rows 1 to nRows-1)
            # We want extras after rows 0 to nRows-2, so average of these
            extra_Y = (allRowsY[:, :-1] + allRowsY[:, 1:]) / 2.0  # (len(env_ids), nRows-1)
        else:
            # Only one row, so no extra waypoints needed
            extra_Y = torch.zeros((len(env_ids), 0), device=waypointsY.device)  # Empty tensor

        # Place logical waypoints in non-extra positions
        waypointsY[:, ~is_extra] = logical_waypointsY

        # Place extra waypoints (vectorized)
        # extra_Y has shape (len(env_ids), nRows)
        # extra_positions_in_new has shape (nRows,)
        # We want waypointsY[env_i, extra_positions_in_new] = extra_Y[env_i, :]
        for i in range(len(env_ids)):
            waypointsY[i, extra_positions_in_new] = extra_Y[i]

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
            env_ids,
            self.currentWaypointIndices[env_ids, 1],
        ]

        # reset the task completed status for the given environment ids
        self.taskCompleted[env_ids] = False

        # reset the waypoint reached status for the given environment ids
        self.waypointReachedBuffer[env_ids] = False

        # reset the robot too far from waypoint status for the given environment ids
        self.robotTooFarFromWaypoint[env_ids] = False

    def updateCurrentMarker(self):
        indexes = torch.zeros(
            (self.nEnvs, self.nWaypoints),
            dtype=torch.int,
            device=self.waypointsPositions.device,
        )
        env_indices = torch.arange(
            self.nEnvs, device=self.currentWaypointIndices.device
        )
        pastMask = torch.arange(
            self.nWaypoints, device=self.waypointsPositions.device
        ).repeat(self.nEnvs, 1) < self.currentWaypointIndices[:, 1].unsqueeze(1)
        # Set the past waypoints (blue)
        indexes[pastMask] = 2

        # Set the current waypoint (green)
        indexes[env_indices, self.currentWaypointIndices[:, 1]] = 1
        indexes = indexes.view(self.nEnvs * self.nWaypoints)

        # Set the scales of the markers
        scales = torch.ones(
            (self.nEnvs, self.nWaypoints, 3),
            dtype=torch.float32,
            device=self.waypointsPositions.device,
        )

        # Set the scale to waypoints more than 3 in the future to 0.2
        futureIndex = torch.clamp(
            self.currentWaypointIndices[:, 1] + 3, max=self.nWaypoints - 1
        )
        futureMask = torch.arange(
            self.nWaypoints, device=self.waypointsPositions.device
        ).repeat(self.nEnvs, 1) > futureIndex.unsqueeze(1)
        scales[futureMask] = 0.2

        self.markersVisualizer.visualize(
            marker_indices=indexes, scales=scales.view(-1, 3)
        )

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
        env_indices_reached = torch.where(waypointReachedAndNotAtLast)[0]
        self.currentWaypointPositions[waypointReachedAndNotAtLast] = (
            self.waypointsPositions[
                env_indices_reached,
                self.currentWaypointIndices[waypointReachedAndNotAtLast, 1],
            ]
        )

        self.taskCompleted = waypointReached & (~notAtLastWaypoint)

        # Update the waypoint reached buffer
        self.waypointReachedBuffer = self.waypointReachedBuffer | waypointReached

    def updateTooFarFromWaypoint(self):
        # Check if the robot is too far from the current waypoint
        maxDistances = (
            torch.ones_like(self.robotTooFarFromWaypoint, dtype=torch.float32)
            * self.maxDistanceToWaypoint
        )
        notFirstMask = self.currentWaypointIndices[:, 1] > 0
        notFirstIndices = torch.where(notFirstMask)[0]

        if notFirstIndices.numel() > 0:
            currentWaypointIndex = self.currentWaypointIndices[notFirstIndices, 1]
            notFirstCurrentWaypointPositions = self.waypointsPositions[
                notFirstIndices, currentWaypointIndex
            ]

            lastWaypointIndex = currentWaypointIndex - 1
            notFirstLastWaypointPositions = self.waypointsPositions[
                notFirstIndices, lastWaypointIndex
            ]

            distanceFromLastToCurrent = torch.norm(
                notFirstCurrentWaypointPositions[:, :2]
                - notFirstLastWaypointPositions[:, :2],
                dim=1,
            )
            distanceFromLastToCurrent += self.waipointReachedEpsilon
            maxDistances[notFirstIndices] = distanceFromLastToCurrent

        self.robotTooFarFromWaypoint = (
            torch.norm(self.robotsdiffs, dim=1) > maxDistances
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
        return self.waypointReachedBuffer

    def resetRewardBuffer(self):
        """Reset the waypoint reached buffer after all reward calculations are done"""
        self.waypointReachedBuffer = torch.zeros_like(self.waypointReachedBuffer)
