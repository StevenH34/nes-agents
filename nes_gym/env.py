"""Gymnasium environment wrapping the emulator's nes_py module."""

import os

import gymnasium as gym
import nes_py
import numpy as np
from gymnasium import spaces

from nes_gym.games import GAMES

OBS_TYPES = ("pixels", "ram")
RAM_SIZE = 2048


class NesEnv(gym.Env):
    """One NES game as a Gymnasium environment.

    Game-specific behaviour comes from the GameSpec registered under `game`. Observations are an (84, 84, 1) 
    uint8 grayscale frame (`obs_type="pixels"`, made by the emulator's shared C++ preprocessing) or the 2 KB 
    of CPU RAM (`obs_type="ram"`). Each step holds the action's buttons for `frame_skip` frames. Frame stacking 
    is left to SB3's VecFrameStack.

    The first reset() boots the game and caches a save state at the start of gameplay; later resets reload it. 
    The emulator is deterministic, so every episode starts from the same frame and `seed` has no effect on it.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str | os.PathLike,
        game: str = "smb",
        obs_type: str = "pixels",
        frame_skip: int = 4,
        render_mode: str | None = None,
    ):
        if game not in GAMES:
            raise ValueError(f"unknown game {game!r}; expected one of {sorted(GAMES)}")
        if obs_type not in OBS_TYPES:
            raise ValueError(f"unknown obs_type {obs_type!r}; expected one of {OBS_TYPES}")
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be at least 1, got {frame_skip}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")

        self.core = nes_py.NesCore(os.fspath(rom_path))
        # Not `self.spec`: gym.Env uses that name for its registration EnvSpec.
        self.game = GAMES[game]()
        self.obs_type = obs_type
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(len(self.game.actions))
        if obs_type == "pixels":
            shape = (nes_py.OBS_SIZE, nes_py.OBS_SIZE, 1)
        else:
            shape = (RAM_SIZE,)
        self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        self._start_state: bytes | None = None
        # The emulator's save state doesn't include the drawn image (that keeps states small), so after load_state()
        # frame() and obs84() still show the previous episode until the next frame is drawn. The start images are
        # cached alongside the start state and returned until the first step.
        self._start_frame: np.ndarray | None = None
        self._start_obs84: np.ndarray | None = None
        self._at_start = False
        self._prev_ram: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self._start_state is None:
            self.game.boot(self.core)
            self._start_state = self.core.save_state()
            self._start_frame = self.core.frame()
            self._start_obs84 = self.core.obs84()
        else:
            self.core.load_state(self._start_state)
        self._at_start = True

        ram = self.core.ram()
        self._prev_ram = ram
        return self._observe(ram), self.game.info(ram)

    def step(self, action):
        if self._prev_ram is None:
            raise RuntimeError("call reset() before step()")

        self.core.step(self.game.actions[int(action)], frames=self.frame_skip)
        self._at_start = False
        ram = self.core.ram()
        reward = float(self.game.reward(self._prev_ram, ram))
        terminated = bool(self.game.terminated(ram))
        self._prev_ram = ram
        # truncated is always False; episode time limits come from gymnasium's TimeLimit wrapper.
        return self._observe(ram), reward, terminated, False, self.game.info(ram)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._start_frame.copy() if self._at_start else self.core.frame()
        return None

    def _observe(self, ram: np.ndarray) -> np.ndarray:
        if self.obs_type == "pixels":
            obs = self._start_obs84.copy() if self._at_start else self.core.obs84()
            return obs[..., np.newaxis]
        # A copy, so a caller modifying the observation can't change the RAM the next reward is computed from.
        return ram.copy()
