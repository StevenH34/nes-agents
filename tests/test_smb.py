"""Super Mario Bros. spec.

The reward/termination tests use synthetic RAM, so they always run. The rest play the real game and are skipped
unless $SMB_ROM is set.
"""

import numpy as np
import pytest

from nes_gym import NesEnv
from nes_gym.games import smb
from nes_gym.games.smb import SmbSpec

RUN_RIGHT = SmbSpec.actions.index(smb.RIGHT | smb.B)
NOOP = SmbSpec.actions.index(0)


def make_ram(x=40, time=400, player_state=smb.STATE_NORMAL, float_state=0, y_viewport=1):
    ram = np.zeros(2048, dtype=np.uint8)
    ram[smb.X_PAGE], ram[smb.X_IN_PAGE] = divmod(x, 256)
    for address, digit in zip(smb.TIMER_DIGITS, f"{time:03d}"):
        ram[address] = int(digit)
    ram[smb.PLAYER_STATE] = player_state
    ram[smb.FLOAT_STATE] = float_state
    ram[smb.Y_VIEWPORT] = y_viewport
    return ram


# --- Synthetic RAM ---


def test_ram_helpers():
    ram = make_ram(x=300, time=123)
    assert smb.x_position(ram) == 300
    assert smb.timer(ram) == 123


def test_reward_rightward_progress():
    assert SmbSpec().reward(make_ram(x=40), make_ram(x=47)) == 7.0


def test_reward_leftward_is_negative():
    assert SmbSpec().reward(make_ram(x=47), make_ram(x=40)) == -7.0


def test_reward_ignores_teleport():
    assert SmbSpec().reward(make_ram(x=40), make_ram(x=40 + smb.MAX_X_STEP + 1)) == 0.0


def test_reward_time_penalty():
    assert SmbSpec().reward(make_ram(time=400), make_ram(time=399)) == -1.0


def test_reward_ignores_timer_reset():
    assert SmbSpec().reward(make_ram(time=0), make_ram(time=400)) == 0.0


@pytest.mark.parametrize(
    "dying",
    [
        {"player_state": smb.STATE_DYING},
        {"player_state": smb.STATE_DEAD},
        {"y_viewport": 2},
    ],
    ids=["dying", "dead", "pit"],
)
def test_death(dying):
    spec = SmbSpec()
    ram = make_ram(**dying)
    assert spec.reward(make_ram(), ram) == -smb.REWARD_CLIP
    assert spec.terminated(ram)


def test_flag_bonus_once_and_terminates():
    spec = SmbSpec()
    on_pole = make_ram(float_state=smb.FLOAT_FLAGPOLE)
    assert spec.reward(make_ram(), on_pole) == smb.REWARD_CLIP
    assert spec.reward(on_pole, on_pole) == 0.0
    assert spec.terminated(on_pole)


def test_normal_play_does_not_terminate():
    assert not SmbSpec().terminated(make_ram())


def test_info():
    info = SmbSpec().info(make_ram(x=300, time=250))
    assert info == {"x": 300, "world": 1, "level": 1, "lives": 0, "time": 250, "flag_get": False}


# --- Real ROM ($SMB_ROM) ---


@pytest.fixture
def env(smb_rom):
    return NesEnv(smb_rom, game="smb", obs_type="ram")


@pytest.fixture
def pixel_env(smb_rom):
    return NesEnv(smb_rom, game="smb", obs_type="pixels", render_mode="rgb_array")


def test_boot_lands_in_1_1(env):
    _, info = env.reset()
    assert info["world"] == 1
    assert info["level"] == 1
    assert info["x"] == 40
    assert info["time"] == 400
    assert not info["flag_get"]


def test_reset_is_deterministic(env):
    first, _ = env.reset()
    for _ in range(20):
        env.step(RUN_RIGHT)
    second, _ = env.reset()
    np.testing.assert_array_equal(first, second)


def test_reset_after_death_shows_start_of_1_1(pixel_env):
    """After dying, the next episode's first observation and screen are the start of 1-1, not the death frame."""
    first_obs, _ = pixel_env.reset()
    first_frame = pixel_env.render()
    for _ in range(200):
        _, _, terminated, _, _ = pixel_env.step(RUN_RIGHT)
        if terminated:
            break
    assert terminated
    obs, _ = pixel_env.reset()
    np.testing.assert_array_equal(first_obs, obs)
    np.testing.assert_array_equal(first_frame, pixel_env.render())


def test_running_right_earns_reward(env):
    env.reset()
    total = sum(env.step(RUN_RIGHT)[1] for _ in range(10))
    assert total > 0
    assert env.game.info(env.core.ram())["x"] > 40


def test_standing_still_costs_time(env):
    env.reset()
    rewards = [env.step(NOOP)[1] for _ in range(20)]
    assert min(rewards) == -1.0
    assert max(rewards) == 0.0


def test_running_into_first_goomba_ends_episode(env):
    env.reset()
    for _ in range(200):
        _, reward, terminated, _, _ = env.step(RUN_RIGHT)
        if terminated:
            break
    assert terminated
    assert reward == -smb.REWARD_CLIP


def test_not_smb_rom_raises(nestest_rom):
    with pytest.raises(RuntimeError, match="did not reach gameplay"):
        NesEnv(nestest_rom, game="smb").reset()
