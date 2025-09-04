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
        points = self.spline.evaluate(t)[envIds]

        # Sample the path number
        pathNumber = torch.randint(0, self.nPaths, (envIds.shape[0], nPoints,), device=self.device)
        points[:, :, 1] += (pathNumber - 0.5 - (self.nPaths - 1) // 2) * self.pathsSpacing

        # Gaussian noise
        noise = torch.randn_like(points, device=self.device) * self.pointNoiseStd
        points += noise

        return points


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
