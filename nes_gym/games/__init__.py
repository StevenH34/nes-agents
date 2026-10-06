"""Game specs, registered by the name NesEnv's `game` argument takes."""

from nes_gym.games.base import GameSpec
from nes_gym.games.null import NullSpec
from nes_gym.games.smb import SmbSpec

GAMES: dict[str, type[GameSpec]] = {
    "null": NullSpec,
    "smb": SmbSpec,
}

__all__ = ["GAMES", "GameSpec", "NullSpec", "SmbSpec"]
