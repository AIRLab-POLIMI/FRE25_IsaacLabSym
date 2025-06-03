import torch


@torch.jit.script
def sdf(plantCenter, plantRadius: float, point):
    return torch.linalg.norm(plantCenter - point, dim=-1) - plantRadius


class RayMarcher:
    def __init__(self, envsOrigins: torch.Tensor, device: torch.device, raysPerRobot: int, maxDistance: float = 10.0, tol: float = 0.01, maxSteps=100):
        assert envsOrigins is not None, "envsOrigins must be provided"
        self.envsOrigins = envsOrigins
        self.nEnvs = envsOrigins.shape[0]
        assert raysPerRobot > 0, "Number of rays per robot must be greater than 0, got {}".format(raysPerRobot)
        self.raysPerRobot = raysPerRobot
        assert maxDistance > 0, "Max distance must be greater than 0, got {}".format(maxDistance)
        self.maxDistance = maxDistance
        assert tol > 0, "Tolerance must be greater than 0, got {}".format(tol)
        self.tol = tol
        assert maxSteps > 0, "Max steps must be greater than 0, got {}".format(maxSteps)
        self.maxSteps = maxSteps
        self.device = device

        # Initialize rays directions
        self.directionsAngles = torch.linspace(0, 2 * torch.pi, self.raysPerRobot, device=self.device)
        self.directions = torch.stack([torch.cos(self.directionsAngles), torch.sin(self.directionsAngles)], dim=-1)

    # Raymarching for multiple environments
    @torch.jit.script
    def raymarch_parallel_multiple_envs(start, directions, plantsPositions, plantRadius: float, max_distance: float = 5.0, tolerance: float = 1e-2, max_steps: int = 100):
        # start: [nEnvs, 2], directions: [nEnvs, N, 2]
        nEnvs = start.shape[0]
        N = directions.shape[1]
        positions = start.unsqueeze(1).repeat(1, N, 1)  # [nEnvs, N, 2]
        finished = torch.zeros(nEnvs, N, dtype=torch.bool, device=start.device)  # [nEnvs, N]
        distances = torch.zeros(nEnvs, N, dtype=torch.float32, device=start.device)  # [nEnvs, N]
        nSteps = torch.zeros(nEnvs, N, dtype=torch.float32, device=start.device)  # For debugging purposes
        for _ in range(max_steps):
            # Compute SDF for all positions
            # plantsPositions: [nEnvs, nPlants, 2], positions: [nEnvs, N, 2]
            # We want to compute SDF for each environment, for each ray, for each plant
            # Expand plantsPositions to [nEnvs, 1, nPlants, 2], positions to [nEnvs, N, 1, 2]
            sdf_vals = sdf(plantsPositions.unsqueeze(1), plantRadius, positions.unsqueeze(2))  # [nEnvs, N, nPlants]
            # Find the minimum SDF for each ray
            sdf_min, _ = torch.min(sdf_vals, dim=2)  # [nEnvs, N]
            # Find which rays are done
            hit = (sdf_min < tolerance)
            finished = finished | hit
            # Compute step size for unfinished rays
            step = directions * sdf_min.unsqueeze(-1)
            # Only update unfinished rays
            # positions = torch.where(finished.unsqueeze(-1), positions, positions + step)
            positions = (~finished)[..., None] * step + positions
            distances = distances + (~finished) * sdf_min  # Update distances only for unfinished rays
            nSteps = nSteps + (~finished)  # Count steps for debugging
            # Stop if all finished or max_distance reached
            if finished.all() or (distances > max_distance).all():
                break
            # Clamp positions that exceed max_distance
            over_max = distances > max_distance
            distances = torch.where(over_max, max_distance, distances)
            positions = torch.where(over_max.unsqueeze(-1), start.unsqueeze(1).repeat(1, N, 1) + directions * max_distance, positions)
            finished = finished | over_max

        return positions, distances, nSteps

    def sense(self, robot_pos_xy: torch.Tensor, angles: torch.Tensor, plantsPositions: torch.Tensor, plantRadius: float):
        """
        Perform raymarching to sense the environment.
        :param robot_pos_xy: Robot position in world coordinates (nEnvs, 2)
        :param plantsPositions: Plant positions in world coordinates (nEnvs, nPlants, 2)
        :param plantRadius: Radius of the plants
        :return: Positions and distances of the rays
        """
        assert robot_pos_xy.dim() == 2, "robot_pos_xy must be a 2D tensor, got {}D".format(robot_pos_xy.dim())
        assert plantsPositions.dim() == 3, "plantsPositions must be a 3D tensor, got {}D".format(plantsPositions.dim())
        assert plantsPositions.shape[0] == self.nEnvs, "plantsPositions must have the same number of environments as envsOrigins"

        assert angles.shape[0] == self.nEnvs, "angles must have the same number of environments as envsOrigins"
        offsettedAngles = self.directionsAngles.unsqueeze(0) + angles  # [nEnvs, raysPerRobot]
        offsettedAngles %= (2 * torch.pi)  # Ensure angles are in [0, 2*pi]

        self.directions = torch.stack(
            [torch.cos(offsettedAngles), torch.sin(offsettedAngles)], dim=-1
        )
        print("Ray directions shape:", self.directions.shape)

        positions, distances, nSteps = RayMarcher.raymarch_parallel_multiple_envs(
            robot_pos_xy, self.directions, plantsPositions, plantRadius, self.maxDistance, self.tol, self.maxSteps
        )

        return positions, distances, nSteps
