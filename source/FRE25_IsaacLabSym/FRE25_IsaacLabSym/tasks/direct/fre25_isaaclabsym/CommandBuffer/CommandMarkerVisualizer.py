from .CommandMarkers import COMMAND_MARKER_CFG
from .CommandBuffer import CommandBuffer
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_from_angle_axis
import torch


class CommandBufferVisualizer:
    def __init__(self, z_offset: float = 0.3):
        self.markers = VisualizationMarkers(COMMAND_MARKER_CFG)
        self.z_offset = z_offset

    def visualizeCommands(self, robotPoses: torch.Tensor, commandBuffer: CommandBuffer):
        nCommands = commandBuffer.commandsLength

        translations = robotPoses  # n_env x 3
        translations = translations.unsqueeze(1).repeat(1, nCommands, 1)  # n_env x nCommands x 3
        translations[:, :, 2] += self.z_offset * torch.arange(1, nCommands + 1, device=robotPoses.device)  # offset in z for each command

        translations = translations.reshape(-1, 3)

        # Rotate arrows based on command
        rotations = torch.zeros((translations.shape[0], 4), device=robotPoses.device, dtype=torch.float32)  # n_env * nCommands x 4
        turnRight = commandBuffer.turnRightBuffer.bool()  # n_env * nCommands
        angles = torch.where(turnRight, torch.tensor(-90.0, device=robotPoses.device, dtype=torch.float32), torch.tensor(90.0, device=robotPoses.device, dtype=torch.float32))
        angles[:, 1::2] *= -1  # alternate left/right for each command in the buffer
        angles = angles.reshape(-1)  # n_env * nCommands
        angles_rad = angles * (3.14159265 / 180.0)
        rotations = quat_from_angle_axis(angles_rad, torch.tensor([0.0, 0.0, 1.0], device=robotPoses.device, dtype=torch.float32).unsqueeze(0).repeat(rotations.shape[0], 1))

        # Scales
        scales = 0.6 * torch.ones(commandBuffer.turnRightBuffer.shape + (3,), device=robotPoses.device, dtype=torch.float32)  # n_env * nCommands x 3
        currentIndices = commandBuffer.indexBuffer  # n_env
        scales[torch.arange(commandBuffer.nEnvs, device=robotPoses.device), currentIndices, :] = 1.0  # highlight current command
        scales = scales.reshape(-1, 3)

        self.markers.visualize(translations, orientations=rotations, scales=scales)
