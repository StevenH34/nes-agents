"""Tests for play.py's frame scaling, checkpoint following, episode loop and model loading.

The window itself (OpenCV) isn't tested. scale_frame and Follower need only numpy; the rest needs stable-baselines3
and is skipped without it. Episodes use the null game on the committed nestest ROM, on the CPU.
"""

import sys

import numpy as np
import pytest

import play
from nes_gym.checkpoints import checkpoint_path


def test_scale_frame_is_sharp_bgr_and_contiguous():
    """Each pixel becomes a scale × scale block, with red and blue swapped."""
    rgb = np.random.default_rng(0).integers(0, 256, size=(240, 256, 3), dtype=np.uint8)

    big = play.scale_frame(rgb, 3)

    assert big.shape == (720, 768, 3)
    assert big.dtype == np.uint8
    assert big.flags.c_contiguous
    assert np.array_equal(big[::3, ::3], rgb[..., ::-1])
    assert (big[3:6, 6:9] == rgb[1, 2, ::-1]).all()


class FakeLoader:
    """Loads a checkpoint as its file name, or raises FileNotFoundError for names in `missing`."""

    def __init__(self):
        self.missing = set()
        self.loaded = []

    def __call__(self, path):
        if path.name in self.missing:
            raise FileNotFoundError(path)
        self.loaded.append(path.name)
        return path.name


def touch(directory, step):
    """Creates an empty checkpoint file for a step."""
    checkpoint_path(directory, step).write_bytes(b"")


def test_follower_loads_only_newer_checkpoints(tmp_path):
    """Waits for a first checkpoint, then loads each newer one once and ignores older ones and temp files."""
    loader = FakeLoader()
    follower = play.Follower(tmp_path, loader)

    assert follower.poll() is None
    touch(tmp_path, 5)
    assert follower.poll() == ("step_5.zip", 5)
    assert follower.poll() is None

    (tmp_path / ".step_20.zip.0123abcd.tmp").write_bytes(b"")
    assert follower.poll() is None

    touch(tmp_path, 10)
    assert follower.poll() == ("step_10.zip", 10)
    touch(tmp_path, 7)
    assert follower.poll() is None
    assert follower.step == 10
    assert loader.loaded == ["step_5.zip", "step_10.zip"]


def test_follower_skips_checkpoint_deleted_before_loading(tmp_path):
    """A checkpoint that vanishes before it loads keeps the current model; the next poll tries again."""
    loader = FakeLoader()
    follower = play.Follower(tmp_path, loader)
    touch(tmp_path, 5)
    follower.poll()

    touch(tmp_path, 15)
    loader.missing.add("step_15.zip")
    assert follower.poll() is None
    assert follower.step == 5

    loader.missing.clear()
    assert follower.poll() == ("step_15.zip", 15)


def test_follower_missing_directory(tmp_path):
    """A folder training hasn't created yet just has no checkpoint."""
    assert play.Follower(tmp_path / "not-yet", FakeLoader()).poll() is None


@pytest.fixture
def pixel_venv(nestest_rom):
    """A rendering, 5-step-capped null-game env on nestest."""
    pytest.importorskip("stable_baselines3")
    from nes_gym.vec_env import make_vec_env

    venv = make_vec_env(nestest_rom, "null", max_episode_steps=5, render_mode="rgb_array")
    yield venv
    venv.close()


def test_play_episode_shows_each_frame(pixel_venv):
    """Shows the reset frame and every step but the last, and returns the episode's stats."""
    from stable_baselines3 import PPO

    model = PPO("CnnPolicy", pixel_venv, device="cpu", seed=0)
    frames = []

    def show(frame):
        frames.append(frame)
        return True

    result = play.play_episode(model, pixel_venv, deterministic=True, show=show)

    assert result == {"distance": 0, "flag": False, "length": 5}
    assert len(frames) == 5
    assert all(frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3 for frame in frames)


def test_play_episode_stops_when_show_returns_false(pixel_venv):
    """Returns None as soon as show() says the viewer quit."""
    from stable_baselines3 import PPO

    model = PPO("CnnPolicy", pixel_venv, device="cpu", seed=0)
    calls = []

    def show(frame):
        calls.append(frame)
        return len(calls) < 2

    assert play.play_episode(model, pixel_venv, deterministic=True, show=show) is None
    assert len(calls) == 2


def test_load_model_checks_spaces(pixel_venv, nestest_rom, tmp_path):
    """A pixel model loads; a RAM model is rejected with a clear error."""
    from stable_baselines3 import PPO

    from nes_gym.vec_env import make_vec_env

    pixel_path = tmp_path / "pixels.zip"
    PPO("CnnPolicy", pixel_venv, device="cpu", seed=0).save(pixel_path)
    assert play.load_model(pixel_path, pixel_venv, "cpu").observation_space == pixel_venv.observation_space

    ram_venv = make_vec_env(nestest_rom, "null", obs_type="ram")
    try:
        ram_path = tmp_path / "ram.zip"
        PPO("MlpPolicy", ram_venv, device="cpu", seed=0).save(ram_path)
    finally:
        ram_venv.close()
    with pytest.raises(ValueError, match="ram.zip: the model expects observations"):
        play.load_model(ram_path, pixel_venv, "cpu")


@pytest.mark.parametrize(
    "extra, message",
    [
        ([], "no such file"),
        (["--fps", "0"], "--fps: must be at least 0.1"),
        (["--fps", "1e-310"], "--fps: must be at least 0.1"),
        (["--fps", "inf"], "--fps: must be a finite number"),
        (["--episodes", "-1"], "--episodes: must be at least 0"),
    ],
)
def test_main_rejects_bad_arguments(tmp_path, capsys, extra, message):
    """Bad arguments give a usage error before any env or window opens.

    Args:
        extra: The bad arguments.
        message: Text the usage error must contain.
    """
    with pytest.raises(SystemExit) as exc:
        play.main([str(tmp_path / "missing.zip"), "--rom", "x.nes", *extra])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "episodes, follow, stochastic, expected",
    [
        (None, False, False, 1),
        (None, True, False, 0),
        (None, False, True, 0),
        (None, True, True, 0),
        (3, False, False, 3),
        (0, False, False, 0),
        (2, True, True, 2),
    ],
)
def test_default_episodes(episodes, follow, stochastic, expected):
    """Plain argmax playback plays once; following or sampling plays forever; an explicit value wins.

    Args:
        episodes: The --episodes value, or None.
        follow: Whether --follow was given.
        stochastic: Whether --stochastic was given.
        expected: The resolved episode count (0 = forever).
    """
    assert play.default_episodes(episodes, follow, stochastic) == expected


def test_main_reports_a_model_that_does_not_fit(nestest_rom, tmp_path, capsys):
    """A RAM model on the pixel env is a usage error, reported before OpenCV is needed."""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from nes_gym.vec_env import make_vec_env

    ram_venv = make_vec_env(nestest_rom, "null", obs_type="ram")
    try:
        ram_path = tmp_path / "ram.zip"
        PPO("MlpPolicy", ram_venv, device="cpu", seed=0).save(ram_path)
    finally:
        ram_venv.close()

    with pytest.raises(SystemExit) as exc:
        play.main([str(ram_path), "--rom", str(nestest_rom), "--game", "null"])
    assert exc.value.code == 2
    assert "the model expects observations" in capsys.readouterr().err


def test_main_follow_rejects_a_file(tmp_path, capsys):
    """--follow needs a folder, not a checkpoint file."""
    path = tmp_path / "bc_policy.zip"
    path.write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        play.main([str(path), "--rom", "x.nes", "--follow"])
    assert exc.value.code == 2
    assert "--follow takes a checkpoint folder" in capsys.readouterr().err


# main's loop, with OpenCV, the model and the Follower faked. The env is the real nestest null game.


class FakeCv2:
    """Stands in for the cv2 module: records window calls, and plays back scripted key presses.

    Args:
        log: A shared list that window opening is appended to, to check its order against polls.
        keys: waitKey() results, one per call; after they run out, no key is pressed.
        close_after: Reports the window closed once this many frames have been shown, or None to keep it open.
    """

    WINDOW_AUTOSIZE = 1
    WND_PROP_VISIBLE = 4

    def __init__(self, log, keys=(), close_after=None):
        self.log = log
        self.keys = list(keys)
        self.close_after = close_after
        self.frames = 0
        self.titles = []
        self.destroyed = False

    def namedWindow(self, name, flags):  # noqa: N802 (cv2's name)
        self.log.append("namedWindow")

    def imshow(self, name, image):
        assert image.shape == (720, 768, 3)
        self.frames += 1

    def waitKey(self, ms):  # noqa: N802
        return self.keys.pop(0) if self.keys else -1

    def getWindowProperty(self, name, prop):  # noqa: N802
        return 0 if self.close_after is not None and self.frames >= self.close_after else 1

    def setWindowTitle(self, name, title):  # noqa: N802
        self.titles.append(title)

    def destroyAllWindows(self):  # noqa: N802
        self.destroyed = True


class FakeModel:
    """A model that always presses nothing, and records each prediction under its name in `played`."""

    def __init__(self, name, played):
        self.name = name
        self.played = played

    def predict(self, obs, deterministic):
        """Returns action 0 for the one env."""
        self.played.append(self.name)
        return np.zeros(1, dtype=np.int64), None


@pytest.fixture
def loop(monkeypatch, nestest_rom, tmp_path):
    """Fakes cv2, the model loader and the wait between polls; returns a helper that runs main.

    The helper takes the checkpoint argument, extra arguments and FakeCv2 options, and returns (FakeCv2, the
    episode lines main printed).
    """
    pytest.importorskip("stable_baselines3")
    log, played = [], []
    monkeypatch.setattr(play, "WAIT_POLL", 0)
    monkeypatch.setattr(play, "load_model", lambda path, venv, device: FakeModel(path.name, played))

    def run(checkpoint, extra=(), capsys=None, **cv2_options):
        fake = FakeCv2(log, **cv2_options)
        monkeypatch.setitem(sys.modules, "cv2", fake)
        play.main([str(checkpoint), "--rom", str(nestest_rom), "--game", "null", "--max-steps", "3", *extra])
        lines = capsys.readouterr().out.splitlines() if capsys else []
        return fake, [line for line in lines if line.startswith("episode ")]

    run.log, run.played = log, played
    return run


@pytest.fixture
def policy_file(tmp_path):
    """An existing file to pass as the checkpoint (load_model is faked, so its contents don't matter)."""
    path = tmp_path / "bc_policy.zip"
    path.write_bytes(b"")
    return path


def test_main_plays_one_episode_by_default(loop, policy_file, capsys):
    """Plain argmax playback plays once, showing 3 frames for a 3-step episode, then closes the window."""
    fake, episodes = loop(policy_file, capsys=capsys)

    assert episodes == ["episode 1: distance 0, flag False, length 3"]
    assert fake.frames == 3
    assert fake.titles == ["bc_policy"]
    assert fake.destroyed


def test_main_plays_episodes_count(loop, policy_file, capsys):
    """--episodes N plays exactly N."""
    _, episodes = loop(policy_file, ["--episodes", "2"], capsys=capsys)

    assert [line.split(":")[0] for line in episodes] == ["episode 1", "episode 2"]


@pytest.mark.parametrize("key", [ord("q"), 27])
def test_main_quits_on_key(loop, policy_file, capsys, key):
    """q or Esc stops mid-episode, before any episode finishes.

    Args:
        key: The key pressed on the second frame.
    """
    fake, episodes = loop(policy_file, ["--episodes", "0"], capsys=capsys, keys=[-1, key])

    assert episodes == []
    assert fake.frames == 2
    assert fake.destroyed


def test_main_quits_when_window_closed(loop, policy_file, capsys):
    """Closing the window stops like a quit key."""
    fake, episodes = loop(policy_file, ["--episodes", "0"], capsys=capsys, close_after=2)

    assert episodes == []
    assert fake.frames == 2


def test_main_follow_opens_window_after_first_checkpoint_and_switches(loop, monkeypatch, tmp_path, capsys):
    """--follow waits without a window, then switches to a newer checkpoint between episodes."""
    first, second = FakeModel("first", loop.played), FakeModel("second", loop.played)
    script = [None, None, (first, 1), None, (second, 2)]

    class ScriptedFollower:
        """Returns the scripted poll results in order, then nothing newer."""

        def __init__(self, directory, loader):
            pass

        def poll(self):
            result = script.pop(0) if script else None
            loop.log.append("poll" if result is None else f"poll -> step {result[1]}")
            return result

    monkeypatch.setattr(play, "Follower", ScriptedFollower)
    run_dir = tmp_path / "run"

    fake, episodes = loop(run_dir, ["--follow", "--episodes", "2"], capsys=capsys)

    # The window opens only once the first checkpoint has loaded, never while waiting.
    assert loop.log == ["poll", "poll", "poll -> step 1", "namedWindow", "poll", "poll -> step 2"]
    assert fake.titles == ["run step 1", "run step 2"]
    assert len(episodes) == 2
    # Each 3-step episode makes 3 predictions: the first episode by the first model, the second by the second.
    assert loop.played == ["first"] * 3 + ["second"] * 3
