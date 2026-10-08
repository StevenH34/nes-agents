"""Game specs, registered by the name NesEnv's `game` argument takes.

To add a game, subclass GameSpec and register the class in GAMES under the name NesEnv should accept.

Attributes:
    GAMES (dict[str, type[GameSpec]]): Maps each game name to its GameSpec subclass. NesEnv looks up its `game`
        argument here.
"""

from nes_gym.games.base import GameSpec
from nes_gym.games.null import NullSpec
from nes_gym.games.smb import SmbSpec

GAMES: dict[str, type[GameSpec]] = {
    "null": NullSpec,
    "smb": SmbSpec,
}

__all__ = ["GAMES", "GameSpec", "NullSpec", "SmbSpec"]
