"""Gymnasium environments for NES games running on nes-emulator-cpp.

The package exposes NesEnv, which wraps the emulator's nes_py module as a Gymnasium environment. The game it plays
is picked by name from the specs registered in `nes_gym.games.GAMES`.

Example:
    ```python
    from nes_gym import NesEnv

    env = NesEnv("path/to/smb.nes", game="smb")
    obs, info = env.reset()
    ```
"""

from nes_gym.env import NesEnv

__all__ = ["NesEnv"]
