"""Tests for the vectorised, frame-stacked envs, and that bc.Dataset stacks frames exactly as they do.

Needs stable-baselines3 (the `train` extra); skipped without it. Real envs use the null game on the committed nestest
ROM. Everything runs on the CPU.
"""

import functools

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage  # noqa: E402

from nes_gym.bc import Dataset  # noqa: E402
from nes_gym.demos import Demo  # noqa: E402
from nes_gym.env import RAM_SIZE  # noqa: E402
from nes_gym.vec_env import make_vec_env, wrap_vec_env  # noqa: E402


class FakeEnv(gym.Env):
    """An env whose frames are each filled with a new value, with episodes of chosen lengths.

    Values run 1–255 and wrap, so 0 only ever appears as VecFrameStack's padding.

    Args:
        schedule: (length, how) per episode, cycled; `how` is "terminated" or "truncated".
    """

    observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
    action_space = spaces.Discrete(1)

    def __init__(self, schedule):
        self.schedule = schedule
        self.episode = -1
        self.t = 0
        self.value = 0

    def _observe(self):
        """Returns a frame filled with the next value."""
        self.value = self.value % 255 + 1
        return np.full((84, 84, 1), self.value, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        """Starts the next episode in the schedule."""
        super().reset(seed=seed)
        self.episode += 1
        self.t = 0
        return self._observe(), {}

    def step(self, action):
        """Ends the episode once it reaches its scheduled length."""
        self.t += 1
        length, how = self.schedule[self.episode % len(self.schedule)]
        done = self.t >= length
        return self._observe(), 0.0, done and how == "terminated", done and how == "truncated", {}


def test_dataset_stack_matches_vec_frame_stack():
    """Dataset.stack() rebuilds every stack VecFrameStack gave, including episodes shorter than the stack."""
    schedules = [
        [(1, "terminated"), (2, "truncated"), (5, "terminated"), (3, "truncated")],
        [(7, "truncated"), (1, "truncated"), (4, "terminated")],
    ]
    venv = wrap_vec_env(DummyVecEnv([functools.partial(FakeEnv, schedule) for schedule in schedules]))
    try:
        stacks = [venv.reset()]
        dones = []
        for _ in range(60):
            obs, _, done, _ = venv.step(np.zeros(len(schedules), dtype=np.int64))
            stacks.append(obs)
            dones.append(done)
    finally:
        venv.close()
    stacks = np.stack(stacks)  # (61, n_envs, 4, 84, 84)
    dones = np.array(dones)  # (60, n_envs)
    assert stacks.shape[1:] == (len(schedules), 4, 84, 84)

    for env in range(len(schedules)):
        env_stacks = stacks[:, env]
        # SB3 resets automatically: the observation returned with done=True is the next episode's first frame.
        starts = np.concatenate([[True], dones[:, env]])
        frames = env_stacks[:, -1, :, :, None].copy()
        dataset = Dataset([Demo("fake", frames, np.zeros(len(frames), dtype=np.int64), starts)])
        assert np.array_equal(dataset.stack(np.arange(len(dataset))), env_stacks)


def test_wrap_vec_env_rejects_bad_n_stack():
    """n_stack below 1 raises ValueError."""
    venv = DummyVecEnv([functools.partial(FakeEnv, [(3, "terminated")])])
    try:
        with pytest.raises(ValueError, match="n_stack"):
            wrap_vec_env(venv, n_stack=0)
    finally:
        venv.close()


def test_make_vec_env_pixels(nestest_rom):
    """Gives (1, 4, 84, 84) uint8 stacks that start zero-padded and restart when TimeLimit ends an episode."""
    venv = make_vec_env(nestest_rom, "null", max_episode_steps=5)
    try:
        assert isinstance(venv, VecTransposeImage)
        assert venv.observation_space.shape == (4, 84, 84)
        assert venv.observation_space.dtype == np.uint8

        obs = venv.reset()
        assert obs.shape == (1, 4, 84, 84)
        assert obs.dtype == np.uint8
        assert not obs[0, :3].any()

        for step in range(1, 6):
            prev = obs
            obs, _, dones, _ = venv.step(np.zeros(1, dtype=np.int64))
            assert dones[0] == (step == 5)
            if step < 5:
                # The stack shifts by one: the old newest frame is now second-newest.
                assert np.array_equal(obs[0, 2], prev[0, 3])
        # The new episode's stack is zero-padded again.
        assert not obs[0, :3].any()
    finally:
        venv.close()


def test_make_vec_env_ram(nestest_rom):
    """RAM observations are stacked flat and not transposed."""
    venv = make_vec_env(nestest_rom, "null", obs_type="ram")
    try:
        assert isinstance(venv, VecFrameStack)
        assert venv.reset().shape == (1, 4 * RAM_SIZE)
    finally:
        venv.close()


@pytest.mark.parametrize("kwargs", [{"n_envs": 0}, {"n_stack": 0}])
def test_make_vec_env_rejects_bad_counts(nestest_rom, kwargs):
    """n_envs or n_stack below 1 raises ValueError.

    Args:
        kwargs: The bad argument.
    """
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        make_vec_env(nestest_rom, "null", **kwargs)


def test_make_vec_env_subproc(nestest_rom):
    """Runs each env in its own process with the same observations."""
    venv = make_vec_env(nestest_rom, "null", n_envs=2, subproc=True)
    try:
        obs = venv.reset()
        assert obs.shape == (2, 4, 84, 84)
        obs, _, _, _ = venv.step(np.zeros(2, dtype=np.int64))
        # The emulator is deterministic, so both envs show the same frames.
        assert np.array_equal(obs[0], obs[1])
    finally:
        venv.close()


def test_make_vec_env_subproc_raises_real_error(nestest_rom):
    """A bad argument with subproc raises ValueError, not a pipe error from a dead worker."""
    with pytest.raises(ValueError, match="unknown game"):
        make_vec_env(nestest_rom, "no-such-game", n_envs=2, subproc=True)


def test_ppo_does_not_transpose_again(nestest_rom):
    """PPO sees the stacks are already channels-first and uses the VecEnv as it is."""
    venv = make_vec_env(nestest_rom, "null")
    try:
        model = PPO("CnnPolicy", venv, device="cpu")
        assert model.get_env() is venv
        assert model.policy.observation_space.shape == (4, 84, 84)
    finally:
        venv.close()
