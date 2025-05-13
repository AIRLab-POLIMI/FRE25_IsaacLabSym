from torchcubicspline import (natural_cubic_spline_coeffs,
                              NaturalCubicSpline)
import torch
import matplotlib.pyplot as plt


class PathHandler:
    def __init__(self, nPaths: int = 1, pathsSpacing: float = 2.0, nControlPoints: int = 10, pathLength: float = 10.0, pathWidth: float = 1.0, pointNoiseStd: float = 0.1):
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
        t = torch.linspace(0, 1, self.nControlPoints)
        controlsX = torch.linspace(0, self.pathLength, self.nControlPoints)
        controlsY = (2 * torch.rand(self.nControlPoints) - 1) * self.pathWidth
        self.controls = torch.stack((controlsX, controlsY), dim=1)
        coeffs = natural_cubic_spline_coeffs(t, self.controls)
        self.spline = NaturalCubicSpline(coeffs)

    def evaluateSpline(self, ts: torch.Tensor):
        assert hasattr(self, 'spline'), "Spline not generated. Call generatePath() first."
        assert ts.dim() == 1, "Input tensor must be 1D, got {}D".format(ts.dim())
        assert ts.size(0) > 0, "Input tensor must not be empty"
        assert ts.min() >= 0 and ts.max() <= 1, "Input tensor values must be in the range [0, 1], got min: {}, max: {}".format(ts.min(), ts.max())
        return self.spline.evaluate(ts)

    def samplePoints(self, nPoints: int):
        assert hasattr(self, 'spline'), "Spline not generated. Call generatePath() first."
        assert nPoints > 0, "Number of points must be positive, got {}".format(nPoints)
        t = torch.linspace(0, 1, nPoints)
        points = self.spline.evaluate(t)

        # Sample the path number
        pathNumber = torch.randint(0, self.nPaths, (nPoints,))
        points[:, 1] += pathNumber * self.pathsSpacing

        # Gaussian noise
        noise = torch.randn_like(points) * self.pointNoiseStd
        points += noise

        return points


if __name__ == "__main__":
    pathHandler = PathHandler(nPaths=5, nControlPoints=10, pathLength=10.0, pathWidth=0.5)
    pathHandler.generatePath()
    ts = torch.linspace(0, 1, 100)
    out = pathHandler.evaluateSpline(ts)

    samples = pathHandler.samplePoints(1000)
    print(out)

    for i in range(pathHandler.nPaths):
        plt.plot(out[:, 0], out[:, 1] + i * pathHandler.pathsSpacing, color='blue', label='spline')
        plt.scatter(pathHandler.controls[:, 0], pathHandler.controls[:, 1] + i * pathHandler.pathsSpacing, marker='x', color='red', label='data points')
    plt.scatter(samples[:, 0], samples[:, 1], marker='x', color='green', label='sampled points', alpha=0.5)
    plt.title('Natural Cubic Spline')
    plt.legend()
    plt.axis('equal')
    plt.show()
