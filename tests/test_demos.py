"""Demo conversion, using recordings made on the committed nestest ROM (no commercial ROM needed)."""

import nes_py
import numpy as np
import pytest

from nes_gym import demos
from nes_gym.games import GAMES, GameSpec, NullSpec
from nes_gym.games.smb import SmbSpec

A = nes_py.BUTTON_A
B = nes_py.BUTTON_B
START = nes_py.BUTTON_START
DOWN = nes_py.BUTTON_DOWN
LEFT = nes_py.BUTTON_LEFT
RIGHT = nes_py.BUTTON_RIGHT

NOOP_I = SmbSpec.actions.index(0)
RIGHT_I = SmbSpec.actions.index(RIGHT)
A_I = SmbSpec.actions.index(A)
LEFT_I = SmbSpec.actions.index(LEFT)


class SmbActionsSpec(GameSpec):
    """SMB's actions on any ROM, with no boot. `terminated` is true right after the windows in `ends_after`, and
    `playing` is false right after the windows in `unplayable_after` (0-based, counting every window of the
    recording; -1 means before the first window).

    The RAM is ignored. This relies on load_demo asking exactly one of playing() or terminated() before the first
    window and after each window, so a shared call count gives the window.
    """

    actions = SmbSpec.actions

    def __init__(self, ends_after=(), unplayable_after=()):
        self.ends_after = set(ends_after)
        self.unplayable_after = set(unplayable_after)
        self.checks = 0

    def boot(self, core):
        pass

    def reward(self, prev_ram, ram):
        return 0.0

    def terminated(self, ram):
        return self._window() in self.ends_after

    def playing(self, ram):
        return self._window() not in self.unplayable_after

    def _window(self):
        window = self.checks - 1
        self.checks += 1
        return window


def start_state(rom):
    """A save state a little after power-on, so the screen has been drawn."""
    core = nes_py.NesCore(str(rom))
    core.step(0, frames=30)
    return core.save_state(), core.rom_checksum()


def record(path, rom, buttons, checksum_xor=0):
    """Write a recording of `buttons` from start_state() to `path`. A non-zero `checksum_xor` corrupts its ROM
    checksum."""
    state, checksum = start_state(rom)
    rec = nes_py.Recording(checksum ^ checksum_xor, state, np.asarray(buttons, dtype=np.uint8))
    nes_py.save_recording(rec, str(path))
    return path


def random_buttons(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([0, RIGHT, RIGHT | A, B, START], size=n).astype(np.uint8)


def convert(path, rom, game=None, **kwargs):
    return demos.load_demo(path, nes_py.NesCore(str(rom)), game or NullSpec(), **kwargs)


# --- Action mapping ---


@pytest.mark.parametrize(
    "mask, expected",
    [
        (0, NOOP_I),
        (RIGHT, RIGHT_I),
        (RIGHT | A | B, SmbSpec.actions.index(RIGHT | A | B)),
        (B, NOOP_I),  # B alone does nothing; NOOP and R+B are equally far, the earlier action wins
        (LEFT | B, LEFT_I),
        (LEFT | A, A_I),  # equally far from A and L
        (RIGHT | DOWN, RIGHT_I),  # buttons no action uses don't change the choice
        (START, NOOP_I),
    ],
    ids=["noop", "right", "right_a_b", "b", "left_b", "left_a", "right_down", "start"],
)
def test_nearest_action(mask, expected):
    assert demos.nearest_action(mask, SmbSpec.actions) == expected


def test_window_mask_most_common():
    assert demos.window_mask(np.array([RIGHT, A, A, A], dtype=np.uint8)) == A


def test_window_mask_tie_goes_to_last_held():
    assert demos.window_mask(np.array([RIGHT, RIGHT, A, A], dtype=np.uint8)) == A
    assert demos.window_mask(np.array([A, A, RIGHT, RIGHT], dtype=np.uint8)) == RIGHT


def test_mask_name():
    assert demos.mask_name(0) == "NOOP"
    assert demos.mask_name(RIGHT | A) == "R+A"


# --- Replaying recordings ---


def test_pair_count(tmp_path, nestest_rom):
    """50 full windows; the first (no valid image) and the 3 trailing frames are dropped."""
    demo = convert(record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(4 * 50 + 3)), nestest_rom)
    assert len(demo) == 49
    assert demo.observations.shape == (49, nes_py.OBS_SIZE, nes_py.OBS_SIZE, 1)
    assert demo.observations.dtype == np.uint8
    assert demo.actions.dtype == np.int64
    assert demo.segment_count == 1


def test_observations_match_replay(tmp_path, nestest_rom):
    """Pair i's observation is the screen after replaying windows 0..i, i.e. just before window i + 1."""
    buttons = random_buttons(4 * 20)
    demo = convert(record(tmp_path / "a.nesdemo", nestest_rom, buttons), nestest_rom)

    core = nes_py.NesCore(str(nestest_rom))
    core.load_state(start_state(nestest_rom)[0])
    for i in range(len(demo)):
        for mask in buttons[i * 4 : (i + 1) * 4]:
            core.step(int(mask))
        np.testing.assert_array_equal(demo.observations[i, ..., 0], core.obs84())
    assert not np.array_equal(demo.observations[0], demo.observations[-1]), "the screen must change"


def test_ram_observations(tmp_path, nestest_rom):
    demo = convert(record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(4 * 10)), nestest_rom, obs_type="ram")
    assert demo.observations.shape == (9, 2048)
    assert demo.observations.dtype == np.uint8


def test_actions_and_remaps(tmp_path, nestest_rom):
    windows = [0, RIGHT, B, LEFT | A, RIGHT | DOWN, RIGHT]
    buttons = np.repeat(np.array(windows, dtype=np.uint8), 4)
    demo = convert(record(tmp_path / "a.nesdemo", nestest_rom, buttons), nestest_rom, game=SmbActionsSpec())

    np.testing.assert_array_equal(demo.actions, [RIGHT_I, NOOP_I, A_I, RIGHT_I, RIGHT_I])
    assert demo.remapped == {(B, NOOP_I): 1, (LEFT | A, A_I): 1, (RIGHT | DOWN, RIGHT_I): 1}
    assert demo.remap_count == 3


def test_segments(tmp_path, nestest_rom):
    """Terminated after windows 9 and 29, and not yet playable after window 10: windows 10, 11 and 30 are skipped,
    and segments start at the first pair (window 1) and at windows 12 and 31."""
    path = record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(4 * 50))
    demo = convert(path, nestest_rom, game=SmbActionsSpec(ends_after=(9, 29), unplayable_after=(10,)))
    assert len(demo) == 46
    np.testing.assert_array_equal(np.flatnonzero(demo.episode_starts), [0, 9, 27])
    assert demo.segment_count == 3


def test_recording_starting_outside_play(tmp_path, nestest_rom):
    """Not playable at the start or after windows 0 and 1, so the first pair is window 3."""
    path = record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(4 * 50))
    demo = convert(path, nestest_rom, game=SmbActionsSpec(unplayable_after=(-1, 0, 1)))
    assert len(demo) == 47
    assert demo.segment_count == 1


def test_not_playing_inside_a_segment_is_ignored(tmp_path, nestest_rom):
    """Inside a segment only `terminated` ends it, so states like pipes or power-ups don't split it."""

    class PlayableOnlyAtStart(SmbActionsSpec):
        def __init__(self):
            super().__init__()
            self.asked = 0

        def playing(self, ram):
            self.asked += 1
            return self.asked == 1

    path = record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(4 * 50))
    demo = convert(path, nestest_rom, game=PlayableOnlyAtStart())
    assert len(demo) == 49
    assert demo.segment_count == 1


@pytest.mark.parametrize("frames", [0, 7])
def test_too_short_for_a_pair(tmp_path, nestest_rom, frames):
    demo = convert(record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(frames)), nestest_rom)
    assert len(demo) == 0
    assert demo.observations.shape == (0, nes_py.OBS_SIZE, nes_py.OBS_SIZE, 1)


def test_different_rom_rejected(tmp_path, nestest_rom):
    path = record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(40), checksum_xor=1)
    with pytest.raises(ValueError, match="different ROM"):
        convert(path, nestest_rom)


@pytest.mark.parametrize(
    "kwargs", [{"obs_type": "rgb"}, {"frame_skip": 0}], ids=["obs_type", "frame_skip"]
)
def test_bad_arguments(tmp_path, nestest_rom, kwargs):
    path = record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(40))
    with pytest.raises(ValueError):
        convert(path, nestest_rom, **kwargs)


# --- Loading several recordings ---


def test_load_demos_searches_folders(tmp_path, nestest_rom):
    (tmp_path / "sub").mkdir()
    record(tmp_path / "b.nesdemo", nestest_rom, random_buttons(40))
    record(tmp_path / "sub" / "a.nesdemo", nestest_rom, random_buttons(40))
    (tmp_path / "notes.txt").write_text("not a recording")

    loaded = demos.load_demos([tmp_path], nestest_rom, game="null")
    assert [demo.path for demo in loaded] == [str(tmp_path / "b.nesdemo"), str(tmp_path / "sub" / "a.nesdemo")]


def test_load_demos_no_recordings(tmp_path, nestest_rom):
    with pytest.raises(ValueError, match="No .nesdemo files"):
        demos.load_demos([tmp_path], nestest_rom, game="null")


def test_load_demos_unknown_game(tmp_path, nestest_rom):
    with pytest.raises(ValueError, match="Invalid game"):
        demos.load_demos([tmp_path], nestest_rom, game="nope")


def test_cli_report(tmp_path, nestest_rom, monkeypatch, capsys):
    monkeypatch.setitem(GAMES, "smb_actions", SmbActionsSpec)
    windows = [0, B, B, LEFT | A, RIGHT]
    record(tmp_path / "a.nesdemo", nestest_rom, np.repeat(np.array(windows, dtype=np.uint8), 4))

    demos.main(["--rom", str(nestest_rom), "--game", "smb_actions", str(tmp_path)])
    out = capsys.readouterr().out
    assert "4 pairs, 1 segments, 3 remapped (75.0%)" in out
    assert "Total: 4 pairs from 1 recordings, 3 remapped (75.0%)" in out
    assert out.index("B -> NOOP: 2") < out.index("L+A -> A: 1")
