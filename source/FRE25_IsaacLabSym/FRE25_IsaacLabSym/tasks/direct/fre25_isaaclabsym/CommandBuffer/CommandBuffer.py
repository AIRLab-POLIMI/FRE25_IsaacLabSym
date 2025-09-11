from typing import Sequence
import torch


class CommandBuffer:
    """A PyTorch-based command buffer for managing sequences of movement commands across multiple parallel environments.

    This class is designed for reinforcement learning scenarios where multiple environments need to execute
    sequences of movement commands with associated turn directions. Each command represents a number of steps
    to move forward, and each command has an associated turn direction (left/right).
    """

    def __init__(
        self,
        nEnvs: int = 10,
        commandsLength: int = 10,
        maxRows: int = 3,
        device: str = "cuda:0",
    ):
        """Initialize the CommandBuffer with specified parameters.

        Args:
            nEnvs (int, optional): Number of parallel environments to manage. Defaults to 10.
            commandsLength (int, optional): Length of command sequence per environment. Defaults to 10.
            maxRows (int, optional): Maximum number of steps per command (1 to maxRows). Defaults to 3.
            device (str, optional): PyTorch device for tensor operations. Defaults to 'cuda:0'.
        """
        self.nEnvs = nEnvs
        self.commandsLength = commandsLength
        self.maxRows = maxRows
        self.device = device

        # Initialize the command buffer with 0s
        self.commandBuffer = torch.zeros(
            (nEnvs, commandsLength), device=device, dtype=torch.int32
        )

        # Initialize the turn right buffer with False (meaning turn left by default)
        self.turnRightBuffer = torch.zeros(
            (nEnvs, commandsLength), device=device, dtype=torch.bool
        )

        # Initialize index buffer to keep track of the current command index for each environment
        self.indexBuffer = torch.zeros((nEnvs,), dtype=torch.int32, device=device)

        # Initialize done buffer to keep track of completed commands
        self.doneBuffer = torch.zeros((nEnvs,), dtype=torch.bool, device=device)

    def randomizeCommands(self, env_ids: Sequence[int] | None = None):
        """Randomly generate new command sequences for specified environments.

        Generates random movement commands (1 to maxRows steps) and random turn directions
        for each command in the sequence. Resets the index and done buffers for the specified environments.

        Args:
            env_ids (Sequence[int] | None, optional): List of environment IDs to randomize.
                                                    If None, randomizes all environments. Defaults to None.
        """
        if env_ids is None:
            env_ids = list(range(self.nEnvs))

        self.commandBuffer[env_ids] = torch.randint(
            low=1,
            high=self.maxRows + 1,
            size=(len(env_ids), self.commandsLength),
            device=self.device,
            dtype=torch.int32,
        )
        self.turnRightBuffer[env_ids] = torch.randint(
            low=0,
            high=2,
            size=(len(env_ids), self.commandsLength),
            device=self.device,
            dtype=torch.bool,
        )

        self.indexBuffer[env_ids] = torch.zeros(
            (len(env_ids),), dtype=torch.int32, device=self.device
        )

        self.doneBuffer[env_ids] = torch.zeros(
            (len(env_ids),), dtype=torch.bool, device=self.device
        )

    def getCurrentCommands(self) -> torch.Tensor:
        """Get the current commands for all environments in one-hot encoded format.

        Returns the current command for each environment as a one-hot encoded tensor
        where the first maxRows columns represent the movement steps (1 to maxRows)
        and the last column represents the turn direction (0=left, 1=right).

        Returns:
            torch.Tensor: Tensor of shape (nEnvs, maxRows + 1) containing one-hot encoded
                         current commands with turn direction as the last column.
        """
        commands = self.commandBuffer[
            torch.arange(self.nEnvs, device=self.device), self.indexBuffer
        ]
        turnRight = self.turnRightBuffer[
            torch.arange(self.nEnvs, device=self.device), self.indexBuffer
        ]

        # One hot encoding for commands
        oneHotCommands = torch.zeros(
            (self.nEnvs, self.maxRows), device=self.device, dtype=torch.int32
        )
        oneHotCommands[torch.arange(self.nEnvs, device=self.device), commands - 1] = 1

        currentCommands = torch.cat(
            (
                oneHotCommands,
                turnRight.unsqueeze(1).int(),
                (self.indexBuffer % 2).unsqueeze(1).int(),
            ),
            dim=1,
        )

        return currentCommands

    def stepCommands(self, env_ids: Sequence[int]):
        """Execute one step of the current commands for specified environments.

        Decrements the current command value by 1 for each specified environment.
        When a command reaches 0, automatically advances to the next command in the sequence.
        Marks environments as done when they complete all commands in their sequence.

        Args:
            env_ids (Sequence[int]): List of environment IDs to step through their commands.
        """
        self.commandBuffer[env_ids, self.indexBuffer[env_ids]] -= 1

        # clamp to avoid negative values
        self.commandBuffer = torch.clamp(self.commandBuffer, min=0)

        # If the current command reaches 0, move to the next command
        done_mask = self.commandBuffer[env_ids, self.indexBuffer[env_ids]] <= 0
        self.indexBuffer[env_ids] += done_mask.int()

        # If the index exceeds the command length, mark as done
        self.doneBuffer[env_ids] = self.indexBuffer[env_ids] >= self.commandsLength

        self.indexBuffer = torch.clamp(self.indexBuffer, max=self.commandsLength - 1)

    def dones(self) -> torch.Tensor:
        """Get the completion status of all environments.

        Returns a boolean tensor indicating which environments have completed
        all commands in their sequence.

        Returns:
            torch.Tensor: Boolean tensor of shape (nEnvs,) where True indicates
                         the environment has completed all its commands.
        """
        return self.doneBuffer


if __name__ == "__main__":
    commandBuffer = CommandBuffer(nEnvs=5, commandsLength=2, maxRows=3, device="cuda:0")
    commandBuffer.randomizeCommands()
    print("Command Buffer:")
    print(commandBuffer.commandBuffer)
    print("Turn Right Buffer:")
    print(commandBuffer.turnRightBuffer)
    print("Index Buffer:")
    print(commandBuffer.indexBuffer)
    print("Current Commands:")
    print(commandBuffer.getCurrentCommands())

    commandBuffer.stepCommands(env_ids=[0, 1, 2])
    print("Command Buffer after step:")
    print(commandBuffer.commandBuffer)
    print("Current Commands after step:")
    print(commandBuffer.getCurrentCommands())
    print("Index Buffer after step:")
    print(commandBuffer.indexBuffer)

    commandBuffer.randomizeCommands(env_ids=[0])
    print("Command Buffer after re-randomizing env 0:")
    print(commandBuffer.commandBuffer)
    print("Turn Right Buffer after re-randomizing env 0:")
    print(commandBuffer.turnRightBuffer)
    print("Index Buffer after re-randomizing env 0:")
    print(commandBuffer.indexBuffer)
    print("Current Commands after re-randomizing env 0:")
    print(commandBuffer.getCurrentCommands())

    while not commandBuffer.doneBuffer.all():
        commandBuffer.stepCommands(env_ids=list(range(commandBuffer.nEnvs)))
        print("Current Commands during stepping:")
        print(commandBuffer.getCurrentCommands())
        print("Index Buffer during stepping:")
        print(commandBuffer.indexBuffer)
        print("Done Buffer during stepping:")
        print(commandBuffer.doneBuffer)
        print("-----")
