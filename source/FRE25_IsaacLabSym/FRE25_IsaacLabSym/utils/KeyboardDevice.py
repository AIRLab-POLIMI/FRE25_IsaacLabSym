# Copyright (c) 2022-2025, Paolo Ginefra.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for FRE25 simulation."""

import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation

import carb
import omni

from isaaclab.devices.device_base import DeviceBase  # type: ignore


class KeyboardManager(DeviceBase):
    """A keyboard controller for sending commands as steering/throttle and binary command (next command).

    This class is designed to provide a keyboard controller for a 4wis mobile robot that has a command buffer.
    This has been developed mainly for debugging purposes.
    It uses the Omniverse keyboard interface to listen to keyboard events and map them to robot's
    task-space commands.

    For this first version, the kinematic is steel crab, i.e., the robot can move in any direction on the plane without changing its bearing.
    the robot expects:
        - a steering interpreted as the variation of the crab direction (in radians) in time (i.e., how much the robot should turn left/right)
        - a throttle interpreted as the velocity of the robot in the crab direction
        - a binary command to indicate if the robot should make the buffer step (true) or not (false)

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Steering left/right            A (left)          D (right)
        Throttle forward/backward      W (forward)       S (backward)
        Step Command Buffer            E (hold down)
        Reset                          L
        ============================== ================= =================

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self, steeringSensitivity: float = 1, throttleSensitivity: float = 1):
        """Initialize the keyboard layer.

        Args:
            steeringSensitivity: Magnitude of input steering command scaling. Defaults to 0.4.
            throttleSensitivity: Magnitude of scale input throttle commands scaling. Defaults to 0.8.
        """
        # store inputs
        self.steeringSensitivity = steeringSensitivity
        self.throttleSensitivity = throttleSensitivity

        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(
                event, *args
            ),
        )

        # bindings for keyboard to command
        self._create_key_bindings()

        # internal state
        self._kinematic_command_buffer: np.ndarray = np.zeros(2, dtype=np.float32)
        self._step_command_buffer: bool = False

        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for FRE25: {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tSteering left: A\n"
        msg += "\tSteering right: D\n"
        msg += "\tThrottle forward: W\n"
        msg += "\tThrottle backward: S\n"
        msg += "\tStep Command Buffer: E (hold down to set to 1)\n"
        msg += "\tReset: L\n"
        msg += "\t----------------------------------------------\n"
        return msg

    """
    Operations
    """

    def reset(self):
        """Reset the internal state of the keyboard controller."""
        self._kinematic_command_buffer = np.zeros(2, dtype=np.float32)
        self._step_command_buffer = False

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> tuple[np.ndarray, bool]:
        """Provides the result from keyboard event state.
        The step command is reset after each call.

        Returns:
            A tuple containing the steering, throttle and gripper state.
        """
        # return the command and gripper state
        stepCommand = self._step_command_buffer
        return self._kinematic_command_buffer, stepCommand

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            if event.input.name == "E":
                self._step_command_buffer = True
            elif event.input.name in ["W", "S", "A", "D"]:
                self._kinematic_command_buffer += self._INPUT_KEY_MAPPING[
                    event.input.name
                ]

        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name == "E":
                self._step_command_buffer = False
            if event.input.name in ["W", "S", "A", "D"]:
                self._kinematic_command_buffer -= self._INPUT_KEY_MAPPING[
                    event.input.name
                ]
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # step command buffer (hold down 'E')
            "E": True,
            # x-axis (forward)
            "W": np.asarray([0.0, 1.0]) * self.throttleSensitivity,
            "S": np.asarray([0.0, -1.0]) * self.throttleSensitivity,
            # y-axis (left-right)
            "A": np.asarray([1.0, 0.0]) * self.steeringSensitivity,
            "D": np.asarray([-1.0, 0.0]) * self.steeringSensitivity,
        }
