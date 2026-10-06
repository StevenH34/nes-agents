"""NesEnv mechanics, using the null game on the committed nestest ROM (no commercial ROM needed)."""

import warnings

import nes_py
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from nes_gym import NesEnv
from nes_gym.games import GAMES, NullSpec


@pytest.mark.parametrize("obs_type", ["pixels", "ram"])
def test_check_env(nestest_rom, obs_type):
    with warnings.catch_warnings():
        # check_env warns that it can't test other render modes without gymnasium.make; irrelevant here.
        warnings.simplefilter("ignore", UserWarning)
        check_env(NesEnv(nestest_rom, game="null", obs_type=obs_type))


def test_pixel_observation(nestest_rom):
    env = NesEnv(nestest_rom, game="null", obs_type="pixels")
    obs, _ = env.reset()
    assert obs.shape == (nes_py.OBS_SIZE, nes_py.OBS_SIZE, 1)
    assert obs.dtype == np.uint8
    assert env.observation_space.contains(obs)


def test_ram_observation(nestest_rom):
    env = NesEnv(nestest_rom, game="null", obs_type="ram")
    obs, _ = env.reset()
    assert obs.shape == (2048,)
    assert obs.dtype == np.uint8
    assert env.observation_space.contains(obs)


def test_ram_observation_is_a_copy(nestest_rom):
    env = NesEnv(nestest_rom, game="null", obs_type="ram")
    obs, _ = env.reset()
    obs[:] = 0xFF
    assert not np.array_equal(env.core.ram(), obs)


def test_action_space_matches_game(nestest_rom):
    assert NesEnv(nestest_rom, game="null").action_space.n == len(NullSpec.actions)


def test_reset_returns_same_start(nestest_rom):
    env = NesEnv(nestest_rom, game="null", obs_type="ram")
    first, _ = env.reset()
    for _ in range(10):
        env.step(0)
    second, _ = env.reset()
    np.testing.assert_array_equal(first, second)


def test_boot_runs_once(nestest_rom, monkeypatch):
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
    """One step with frame_skip=4 lands on the same state as four steps with frame_skip=1."""
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
    env = NesEnv(nestest_rom, game="null")
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info == {}


def test_step_before_reset_raises(nestest_rom):
    with pytest.raises(RuntimeError, match="reset"):
        NesEnv(nestest_rom, game="null").step(0)


def test_render(nestest_rom):
    env = NesEnv(nestest_rom, game="null", render_mode="rgb_array")
    env.reset()
    frame = env.render()
    assert frame.shape == (240, 256, 3)
    assert frame.dtype == np.uint8


def test_render_without_mode_returns_none(nestest_rom):
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
    with pytest.raises(ValueError):
        NesEnv(nestest_rom, **kwargs)


def test_bad_rom(tmp_path):
    bad = tmp_path / "bad.nes"
    bad.write_bytes(b"not a rom")
    with pytest.raises(RuntimeError):
        NesEnv(bad, game="null")
