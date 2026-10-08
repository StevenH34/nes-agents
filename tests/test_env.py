"""Tests for NesEnv mechanics.

The tests use the null game on the committed nestest ROM, so no commercial ROM is needed.
"""

import warnings

import nes_py
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from nes_gym import NesEnv
from nes_gym.games import GAMES, NullSpec


@pytest.mark.parametrize("obs_type", ["pixels", "ram"])
def test_check_env(nestest_rom, obs_type):
    """Passes gymnasium's environment checker.

    Args:
        obs_type: The observation type to check.
    """
    with warnings.catch_warnings():
        # check_env warns that it can't test other render modes without gymnasium.make; irrelevant here.
        warnings.simplefilter("ignore", UserWarning)
        check_env(NesEnv(nestest_rom, game="null", obs_type=obs_type))


def test_pixel_observation(nestest_rom):
    """Gives a square grayscale uint8 screen that fits the observation space."""
    env = NesEnv(nestest_rom, game="null", obs_type="pixels")
    obs, _ = env.reset()
    assert obs.shape == (nes_py.OBS_SIZE, nes_py.OBS_SIZE, 1)
    assert obs.dtype == np.uint8
    assert env.observation_space.contains(obs)


def test_ram_observation(nestest_rom):
    """Gives the 2 KB of CPU RAM as a uint8 array that fits the observation space."""
    env = NesEnv(nestest_rom, game="null", obs_type="ram")
    obs, _ = env.reset()
    assert obs.shape == (2048,)
    assert obs.dtype == np.uint8
    assert env.observation_space.contains(obs)


def test_ram_observation_is_a_copy(nestest_rom):
    """Gives a copy of the RAM, so changing the observation leaves the emulator untouched."""
    env = NesEnv(nestest_rom, game="null", obs_type="ram")
    obs, _ = env.reset()
    obs[:] = 0xFF
    assert not np.array_equal(env.core.ram(), obs)


def test_action_space_matches_game(nestest_rom):
    """Has one discrete action per action of the game."""
    assert NesEnv(nestest_rom, game="null").action_space.n == len(NullSpec.actions)


@pytest.mark.parametrize("obs_type", ["pixels", "ram"])
def test_reset_returns_same_start(nestest_rom, obs_type):
    """Gives the same observation on every reset, even after steps have changed the screen.

    Pixels too: the save state doesn't include the drawn image, so a reloaded start must not show the old one.

    Args:
        obs_type: The observation type to compare.
    """
    env = NesEnv(nestest_rom, game="null", obs_type=obs_type)
    first, _ = env.reset()
    for _ in range(10):
        moved, *_ = env.step(0)
    assert not np.array_equal(first, moved), "the screen must change, or this test can't catch a stale image"
    second, _ = env.reset()
    np.testing.assert_array_equal(first, second)


def test_render_after_reset_returns_start_frame(nestest_rom):
    """Renders the start frame after a reset, not the last frame drawn before it."""
    env = NesEnv(nestest_rom, game="null", render_mode="rgb_array")
    env.reset()
    first = env.render()
    for _ in range(10):
        env.step(0)
    assert not np.array_equal(first, env.render())
    env.reset()
    np.testing.assert_array_equal(first, env.render())


def test_reset_observation_is_a_copy(nestest_rom):
    """Gives a copy of the start observation, so changing it doesn't affect later resets."""
    env = NesEnv(nestest_rom, game="null", obs_type="pixels")
    first, _ = env.reset()
    first[:] = 0xFF
    second, _ = env.reset()
    assert not np.array_equal(first, second)


def test_boot_runs_once(nestest_rom, monkeypatch):
    """Boots the game once, not on every reset."""
    boots = []

    class CountingSpec(NullSpec):
        def boot(self, core):
            boots.append(1)

    monkeypatch.setitem(GAMES, "counting", CountingSpec)
    env = NesEnv(nestest_rom, game="counting")
    for _ in range(3):
        env.reset()
    assert len(boots) == 1


def test_frame_skip(nestest_rom):
    """Repeats each action for `frame_skip` frames.

    One step with frame_skip=4 lands on the same state as four steps with frame_skip=1.
    """
    skip4 = NesEnv(nestest_rom, game="null", obs_type="ram", frame_skip=4)
    skip1 = NesEnv(nestest_rom, game="null", obs_type="ram", frame_skip=1)
    skip4.reset()
    skip1.reset()
    obs4 = skip4.step(0)[0]
    for _ in range(4):
        obs1 = skip1.step(0)[0]
    np.testing.assert_array_equal(obs4, obs1)
    np.testing.assert_array_equal(skip4.core.frame(), skip1.core.frame())


def test_step_returns_game_values(nestest_rom):
    """Takes the reward, termination and info from the game spec, and never truncates."""
    env = NesEnv(nestest_rom, game="null")
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info == {}


def test_step_before_reset_raises(nestest_rom):
    """Raises RuntimeError when stepped before the first reset."""
    with pytest.raises(RuntimeError, match="reset"):
        NesEnv(nestest_rom, game="null").step(0)


def test_render(nestest_rom):
    """Renders the full 256x240 RGB frame in `rgb_array` mode."""
    env = NesEnv(nestest_rom, game="null", render_mode="rgb_array")
    env.reset()
    frame = env.render()
    assert frame.shape == (240, 256, 3)
    assert frame.dtype == np.uint8


def test_render_without_mode_returns_none(nestest_rom):
    """Renders nothing when no render mode is set."""
    env = NesEnv(nestest_rom, game="null")
    env.reset()
    assert env.render() is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"game": "nope"},
        {"game": "null", "obs_type": "rgb"},
        {"game": "null", "frame_skip": 0},
        {"game": "null", "render_mode": "human"},
    ],
    ids=["game", "obs_type", "frame_skip", "render_mode"],
)
def test_bad_arguments(nestest_rom, kwargs):
    """Raises ValueError for an unknown game, observation type or render mode, or a frame skip below 1.

    Args:
        kwargs: The NesEnv arguments, one of them bad.
    """
    with pytest.raises(ValueError):
        NesEnv(nestest_rom, **kwargs)


def test_bad_rom(tmp_path):
    """Raises RuntimeError for a file that isn't a valid ROM."""
    bad = tmp_path / "bad.nes"
    bad.write_bytes(b"not a rom")
    with pytest.raises(RuntimeError):
        NesEnv(bad, game="null")
