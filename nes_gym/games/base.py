"""The interface every game implements."""

from abc import ABC, abstractmethod
from typing import ClassVar

import nes_py
import numpy as np


class GameSpec(ABC):
    """Everything NesEnv needs to know about one game.

    The methods read the game's state from its 2 KB of RAM, so a spec never touches the screen.

    Attributes:
        actions: The controller-1 button masks (nes_py.BUTTON_* or'd together) the agent can choose from. The action
            index is a position in this tuple.
    """
    actions: ClassVar[tuple[int, ...]]

    @abstractmethod
    def boot(self, core: nes_py.NesCore) -> None:
        """Runs the game from power-on to the first frame of gameplay.

        Episodes start from the state left here.

        Args:
            core: The emulator core to drive.
        """

    @abstractmethod
    def reward(self, prev_ram: np.ndarray, ram: np.ndarray) -> float:
        """Computes the reward for one step.

        Args:
            prev_ram: The RAM before the step.
            ram: The RAM after the step.

        Returns:
            The reward for the step.
        """

    @abstractmethod
    def terminated(self, ram: np.ndarray) -> bool:
        """Reports whether the episode is over (e.g. a life was lost or the level was finished).

        Args:
            ram: The current RAM.

        Returns:
            True if the episode is over.
        """

    def playing(self, ram: np.ndarray) -> bool:
        """Reports whether the game is in normal play, where a NesEnv episode could be running.

        Demo conversion starts a new segment only where this is true: at the start of a recording, and after the game
        reports `terminated`. Defaults to not terminated, which suits games with no screens between lives.

        Args:
            ram: The current RAM.

        Returns:
            True if the game is in normal play.
        """
        return not self.terminated(ram)

    def paused(self, ram: np.ndarray) -> bool:
        """Reports whether the game is paused (frozen until the player unpauses).

        Demo conversion skips paused windows without ending the segment, since the agent can't pause. Defaults to
        False.

        Args:
            ram: The current RAM.

        Returns:
            True if the game is paused.
        """
        return False

    def info(self, ram: np.ndarray) -> dict:
        """Returns extra values for logging (e.g. position, level).

        Not used for training.

        Args:
            ram: The current RAM.

        Returns:
            A dict of values to log.
        """
        return {}
