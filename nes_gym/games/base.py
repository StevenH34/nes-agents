"""The interface every game implements."""

from abc import ABC, abstractmethod
from typing import ClassVar

import nes_py
import numpy as np


class GameSpec(ABC):
    """Everything NesEnv needs to know about one game.

    `actions` lists the controller-1 button masks (nes_py.BUTTON_* or'd together) the agent can choose from. The
    action index is a position in this tuple. The other methods read the game's state from its 2 KB of RAM, so a
    spec never touches the screen.
    """

    actions: ClassVar[tuple[int, ...]]

    @abstractmethod
    def boot(self, core: nes_py.NesCore) -> None:
        """Run the game from power-on to the first frame of gameplay. Episodes start from the state left here."""

    @abstractmethod
    def reward(self, prev_ram: np.ndarray, ram: np.ndarray) -> float:
        """Reward for one step, given the RAM before and after it."""

    @abstractmethod
    def terminated(self, ram: np.ndarray) -> bool:
        """Whether the episode is over (e.g. a life was lost or the level was finished)."""

    def playing(self, ram: np.ndarray) -> bool:
        """Whether the game is in normal play, where a NesEnv episode could be running. Demo conversion starts a new
        segment only where this is true: at the start of a recording, and after the game reports `terminated`.
        Defaults to not terminated, which suits games with no screens between lives."""
        return not self.terminated(ram)

    def paused(self, ram: np.ndarray) -> bool:
        """Whether the game is paused (frozen until the player unpauses). Demo conversion skips paused windows
        without ending the segment, since the agent can't pause. Defaults to False."""
        return False

    def info(self, ram: np.ndarray) -> dict:
        """Extra values for logging (e.g. position, level). Not used for training."""
        return {}
