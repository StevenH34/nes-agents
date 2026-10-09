"""Vectorised, frame-stacked NesEnvs for SB3, shared by bc_train.py, train.py and play.py.

The stack and transpose are added only in wrap_vec_env(), so behaviour cloning, RL and playback all see the same
observations. For pixels that's (n_envs, 4, 84, 84) uint8, oldest frame first: the same shape as bc.Dataset.stack()
and the Phase 6 ONNX input. PPO sees the frames are already channels-first and doesn't transpose them again.

RAM observations aren't images, so they're stacked to (n_envs, 4 × 2048) and not transposed.
"""

import functools
import os

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from nes_gym.env import NesEnv


def wrap_vec_env(venv: VecEnv, n_stack: int = 4) -> VecEnv:
    """Stacks a VecEnv's last `n_stack` observations, channels-first for images.

    VecFrameStack stacks on the last axis, turning (84, 84, 1) frames into (84, 84, n_stack), and zeroes the stack
    when an episode ends. VecTransposeImage then makes that (n_stack, 84, 84).

    Args:
        venv: The VecEnv to wrap, e.g. of NesEnvs or a test's fake env.
        n_stack: The number of observations in each stack.

    Returns:
        The wrapped VecEnv.

    Raises:
        ValueError: `n_stack` is less than 1.
    """
    if n_stack < 1:
        raise ValueError(f"n_stack must be at least 1, got {n_stack}")
    venv = VecFrameStack(venv, n_stack)
    if is_image_space(venv.observation_space):
        venv = VecTransposeImage(venv)
    return venv


def _make_env(
    rom: str,
    game: str,
    obs_type: str,
    frame_skip: int,
    max_episode_steps: int | None,
    render_mode: str | None,
) -> Monitor:
    """Builds one NesEnv with its optional TimeLimit and a Monitor. Module-level so SubprocVecEnv can pickle it."""
    env = NesEnv(rom, game=game, obs_type=obs_type, frame_skip=frame_skip, render_mode=render_mode)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return Monitor(env)


def make_vec_env(
    rom: str | os.PathLike,
    game: str,
    obs_type: str = "pixels",
    frame_skip: int = 4,
    n_envs: int = 1,
    n_stack: int = 4,
    max_episode_steps: int | None = None,
    subproc: bool = False,
    render_mode: str | None = None,
) -> VecEnv:
    """Builds `n_envs` NesEnvs as one frame-stacked VecEnv.

    Each env is NesEnv, then TimeLimit (only if `max_episode_steps` is given), then Monitor, which records each
    episode's reward and length for SB3's logs. The envs are combined in a DummyVecEnv, or a SubprocVecEnv with one
    process each, then passed to wrap_vec_env(). With `subproc`, one env is built and closed in this process first,
    so a bad argument or ROM raises here instead of killing the workers.

    There's no seed: NesEnv is deterministic, so every episode starts from the same frame.

    Args:
        rom: Path to the game's .nes ROM file.
        game: The key of the game's GameSpec in nes_gym.games.GAMES.
        obs_type: "pixels" or "ram", as in NesEnv.
        frame_skip: The number of frames each action's buttons are held for.
        n_envs: The number of environments.
        n_stack: The number of observations in each stack.
        max_episode_steps: Truncates episodes after this many steps, or None for no limit.
        subproc: Runs each env in its own process (SubprocVecEnv) instead of in this one (DummyVecEnv).
        render_mode: "rgb_array" so the envs can render the full frame (play.py), or None.

    Returns:
        The VecEnv. For pixels its observations are (n_envs, n_stack, 84, 84) uint8.

    Raises:
        ValueError: `n_envs` or `n_stack` is less than 1. NesEnv raises ValueError for a bad `game`, `obs_type`,
            `frame_skip` or `render_mode`.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be at least 1, got {n_envs}")
    if n_stack < 1:
        raise ValueError(f"n_stack must be at least 1, got {n_stack}")
    factory = functools.partial(_make_env, str(rom), game, obs_type, frame_skip, max_episode_steps, render_mode)
    if subproc:
        # A worker that fails to build its env dies, and the parent only sees EOFError or BrokenPipeError. Building
        # one here first raises the real error. It's cheap: NesEnv doesn't boot the game until the first reset().
        factory().close()
    vec_cls = SubprocVecEnv if subproc else DummyVecEnv
    return wrap_vec_env(vec_cls([factory] * n_envs), n_stack)
