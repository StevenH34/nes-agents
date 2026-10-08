"""A placeholder game, so the environment can be tested on any ROM."""

import nes_py
import numpy as np

from nes_gym.games.base import GameSpec


class NullSpec(GameSpec):
    """A game with no goal, for testing NesEnv on any ROM.

    It has one action (no buttons), gives zero reward and never terminates. Episodes start at power-on.

    Attributes:
        actions: A single action that presses no buttons.
    """
    actions = (0,)

    def boot(self, core: nes_py.NesCore) -> None:
        """Does nothing, so episodes start at power-on.

        Args:
            core: The emulator core (unused).
        """

    def reward(self, prev_ram: np.ndarray, ram: np.ndarray) -> float:
        """Gives no reward.

        Args:
            prev_ram: The RAM before the step (unused).
            ram: The RAM after the step (unused).

        Returns:
            Always 0.0.
        """
        return 0.0

    def terminated(self, ram: np.ndarray) -> bool:
        """Never ends the episode.

        Args:
            ram: The current RAM (unused).

        Returns:
            Always False.
        """
        return False
