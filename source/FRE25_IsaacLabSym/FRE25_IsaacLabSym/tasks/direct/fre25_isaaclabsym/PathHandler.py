from torchcubicspline import (natural_cubic_spline_coeffs,
                              NaturalCubicSpline)
import torch
import matplotlib.pyplot as plt


class PathHandler:
    def __init__(self, device: str, nEnvs: int = 2, nPaths: int = 1, pathsSpacing: float = 2.0, nControlPoints: int = 10, pathLength: float = 10.0, pathWidth: float = 1.0, pointNoiseStd: float = 0.1):
        self.device = device

        assert nEnvs > 0, "Number of environments must be positive, got {}".format(nEnvs)
        self.nEnvs = nEnvs

        assert nPaths > 0, "Number of paths must be positive, got {}".format(nPaths)
        self.nPaths = nPaths

        assert pathsSpacing > 0, "Paths spacing must be positive, got {}".format(pathsSpacing)
        self.pathsSpacing = pathsSpacing

        assert nControlPoints > 0, "Number of control points must be positive, got {}".format(nControlPoints)
        self.nControlPoints = nControlPoints

        assert pathLength > 0, "Path length must be positive, got {}".format(pathLength)
        self.pathLength = pathLength

        assert pathWidth > 0, "Path width must be positive, got {}".format(pathWidth)
        self.pathWidth = pathWidth

        assert pointNoiseStd >= 0, "Point noise standard deviation must be non-negative, got {}".format(pointNoiseStd)
        self.pointNoiseStd = pointNoiseStd

    def generatePath(self):
        # TODO: find a way to generate the path just for some environments
        t = torch.linspace(0, 1, self.nControlPoints, device=self.device)
        controlsX = torch.linspace(0, self.pathLength, self.nControlPoints, device=self.device)
        controlsX = controlsX.repeat(self.nEnvs, 1)
        controlsY = (2 * torch.rand((self.nEnvs, self.nControlPoints), device=self.device) - 1) * self.pathWidth
        self.controls = torch.stack((controlsX, controlsY), dim=2)
        coeffs = natural_cubic_spline_coeffs(t, self.controls)
        self.spline = NaturalCubicSpline(coeffs)

    def evaluateSpline(self, ts: torch.Tensor):
        assert hasattr(self, 'spline'), "Spline not generated. Call generatePath() first."
        assert ts.dim() == 1, "Input tensor must be 1D, got {}D".format(ts.dim())
        assert ts.size(0) > 0, "Input tensor must not be empty"
        assert ts.min() >= 0 and ts.max() <= 1, "Input tensor values must be in the range [0, 1], got min: {}, max: {}".format(ts.min(), ts.max())
        return self.spline.evaluate(ts)

    def samplePoints(self, envIds: torch.Tensor, nPoints: int):
        assert hasattr(self, 'spline'), "Spline not generated. Call generatePath() first."
        assert nPoints > 0, "Number of points must be positive, got {}".format(nPoints)
        assert envIds.dim() == 1, "envIds must be a 1D tensor, got {}D".format(envIds.dim())
        assert envIds.size(0) > 0, "envIds must not be empty"
        assert envIds.max() < self.nEnvs, "envIds must be less than nEnvs, got max: {}".format(envIds.max())
        assert envIds.min() >= 0, "envIds must be non-negative, got min: {}".format(envIds.min())
        t = torch.linspace(0, 1, nPoints, device=self.device)
        points = self.spline.evaluate(t)[envIds]  # n_envs x nPoints x 2

        # Sample the path number
        pathNumber = torch.randint(0, self.nPaths, (envIds.shape[0], nPoints,), device=self.device)
        points[:, :, 1] += (pathNumber - 0.5 - (self.nPaths - 1) // 2) * self.pathsSpacing

        # Gaussian noise
        noise = torch.randn_like(points, device=self.device) * self.pointNoiseStd
        points += noise

        return points

    def gridPoints(self, envIds: torch.Tensor, nPoints: int):
        '''
        Sample points on a grid along the path for the given environments.
        The points are sampled along the path and then shifted in the y direction
        based on the path index to create multiple parallel paths.'''
        assert hasattr(self, 'spline'), "Spline not generated. Call generatePath() first."
        assert nPoints > 0, "Number of points must be positive, got {}".format(nPoints)
        assert envIds.dim() == 1, "envIds must be a 1D tensor, got {}D".format(envIds.dim())
        assert envIds.size(0) > 0, "envIds must not be empty"
        assert envIds.max() < self.nEnvs, "envIds must be less than nEnvs, got max: {}".format(envIds.max())
        assert envIds.min() >= 0, "envIds must be non-negative, got min: {}".format(envIds.min())
        t = torch.linspace(0, 1, nPoints // self.nPaths, device=self.device)
        points = self.spline.evaluate(t)[envIds]  # n_envs x nPoints/nPaths x 2
        points = points.repeat(1, self.nPaths, 1)  # n_envs x nPoints x 2

        # Sample the path number
        pathNumber = torch.arange(0, self.nPaths, device=self.device)  # nPaths
        yOffset = (pathNumber - 0.5 - (self.nPaths - 1) // 2) * self.pathsSpacing  # nPaths
        # Repeat elementwise
        yOffset = yOffset.repeat_interleave(nPoints // self.nPaths)  # nPoints
        points[:, :, 1] += yOffset

        # Gaussian noise
        noise = torch.randn_like(points, device=self.device) * self.pointNoiseStd
        points += noise

        return points

    def sampleYOffsetFromX(self, x_positions: torch.Tensor, path_indices: torch.Tensor):
        """
        Sample Y offsets from the spline for given X positions and path indices.

        Args:
            x_positions: X coordinates to sample at, shape (n_envs, n_waypoints)
            path_indices: Path indices for each waypoint, shape (n_envs, n_waypoints)

        Returns:
            Y offsets from the spline, shape (n_envs, n_waypoints)
        """
        assert hasattr(self, 'spline'), "Spline not generated. Call generatePath() first."
        assert x_positions.dim() == 2, f"x_positions must be 2D (n_envs, n_waypoints), got {x_positions.dim()}D"
        assert path_indices.dim() == 2, f"path_indices must be 2D (n_envs, n_waypoints), got {path_indices.dim()}D"
        assert x_positions.shape == path_indices.shape, f"x_positions and path_indices must have same shape"

        # Convert X positions to t values (normalized to [0, 1])
        t_values = x_positions / self.pathLength
        t_values = torch.clamp(t_values, 0.0, 1.0)

        # Flatten for batch evaluation
        n_envs, n_waypoints = x_positions.shape
        t_flat = t_values.flatten()

        # Evaluate spline at all t values
        # spline.evaluate expects 1D tensor, returns (n_envs, n_points, 2)
        spline_points = self.spline.evaluate(t_flat)  # (n_total_points, 2)

        # Reshape to (n_envs * n_waypoints, 2)
        spline_xy = spline_points.view(n_envs, n_waypoints, 2)

        # Extract Y offsets (index 1 is Y coordinate)
        y_offsets = spline_xy[:, :, 1]

        return y_offsets


if __name__ == "__main__":
    pathHandler = PathHandler(nPaths=5, nControlPoints=10, pathLength=10.0, pathWidth=0.5)
    pathHandler.generatePath()
    ts = torch.linspace(0, 1, 100)
    env = 1
    out = pathHandler.evaluateSpline(ts)[env]

    samples = pathHandler.samplePoints(torch.Tensor([0, 1]), 1000)
    print(out)

    for i in range(pathHandler.nPaths):
        plt.plot(out[:, 0], out[:, 1] + i * pathHandler.pathsSpacing, color='blue', label='spline')
        plt.scatter(pathHandler.controls[env, :, 0], pathHandler.controls[env, :, 1] + i * pathHandler.pathsSpacing, marker='x', color='red', label='data points')
    plt.scatter(samples[env, :, 0], samples[env, :, 1], marker='x', color='green', label='sampled points', alpha=0.5)
    plt.title('Natural Cubic Spline')
    plt.legend()
    plt.axis('equal')
    plt.show()
