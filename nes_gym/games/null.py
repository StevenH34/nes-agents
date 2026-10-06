"""A placeholder game, so the environment can be tested on any ROM."""

import nes_py
import numpy as np

from nes_gym.games.base import GameSpec


class NullSpec(GameSpec):
    """One action (no buttons), zero reward, never terminates. Episodes start at power-on."""

    actions = (0,)

    def boot(self, core: nes_py.NesCore) -> None:
        pass

    def reward(self, prev_ram: np.ndarray, ram: np.ndarray) -> float:
        return 0.0

    def terminated(self, ram: np.ndarray) -> bool:
        return False
