"""Tests for demo conversion.

The recordings are made on the committed nestest ROM, so no commercial ROM is needed.
"""

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
    """A GameSpec with SMB's actions that runs on any ROM, with no boot.

    `terminated` is true right after the windows in `ends_after`, and `playing` is false right after the windows in
    `unplayable_after` (0-based, counting every window of the recording; -1 means before the first window). `paused`
    is true at the start of the windows in `paused_at`.

    The RAM is ignored. This relies on load_demo asking exactly one of playing() or terminated() before the first
    window and after each window, so a shared call count gives the window. paused() is asked once before the first
    window and after each window too, so it keeps its own count.

    Args:
        ends_after: The windows right after which `terminated` is true.
        unplayable_after: The windows right after which `playing` is false; -1 means before the first window.
        paused_at: The windows at whose start `paused` is true.

    Attributes:
        ends_after: The set of windows right after which `terminated` is true.
        unplayable_after: The set of windows right after which `playing` is false.
        paused_at: The set of windows at whose start `paused` is true.
        checks: The number of playing() and terminated() calls so far.
        pause_checks: The number of paused() calls so far.
    """

    actions = SmbSpec.actions

    def __init__(self, ends_after=(), unplayable_after=(), paused_at=()):
        self.ends_after = set(ends_after)
        self.unplayable_after = set(unplayable_after)
        self.paused_at = set(paused_at)
        self.checks = 0
        self.pause_checks = 0

    def boot(self, core):
        """Does nothing; the recording's save state already starts where it should.

        Args:
            core: The emulator core, which is left untouched.
        """

    def reward(self, prev_ram, ram):
        """Gives no reward.

        Args:
            prev_ram: The RAM before the step, ignored.
            ram: The RAM after the step, ignored.

        Returns:
            Always 0.0.
        """
        return 0.0

    def terminated(self, ram):
        """Reports whether the episode ended in the window just replayed.

        Args:
            ram: The CPU RAM, ignored.

        Returns:
            True if the window just replayed is in `ends_after`.
        """
        return self._window() in self.ends_after

    def playing(self, ram):
        """Reports whether the game is playable after the window just replayed.

        Args:
            ram: The CPU RAM, ignored.

        Returns:
            False if the window just replayed is in `unplayable_after`, otherwise True.
        """
        return self._window() not in self.unplayable_after

    def paused(self, ram):
        """Reports whether the game is paused at the start of the next window.

        Args:
            ram: The CPU RAM, ignored.

        Returns:
            True if the next window is in `paused_at`.
        """
        window = self.pause_checks
        self.pause_checks += 1
        return window in self.paused_at

    def _window(self):
        """Counts a playing() or terminated() call and gives the window it was asked after.

        Returns:
            The 0-based window just replayed, or -1 before the first window.
        """
        window = self.checks - 1
        self.checks += 1
        return window


def start_state(rom):
    """Makes a save state a little after power-on, so the screen has been drawn.

    Args:
        rom: Path to the ROM to power on.

    Returns:
        A tuple of the save state and the ROM's checksum.
    """
    core = nes_py.NesCore(str(rom))
    core.step(0, frames=30)
    return core.save_state(), core.rom_checksum()


def record(path, rom, buttons, checksum_xor=0):
    """Writes a recording of `buttons`, starting from start_state(), to `path`.

    Args:
        path: Where to save the recording.
        rom: Path to the ROM the recording is made on.
        buttons: The button mask held on each frame.
        checksum_xor: XORed into the recorded ROM checksum; non-zero corrupts it.

    Returns:
        `path`, for chaining.
    """
    state, checksum = start_state(rom)
    rec = nes_py.Recording(checksum ^ checksum_xor, state, np.asarray(buttons, dtype=np.uint8))
    nes_py.save_recording(rec, str(path))
    return path


def random_buttons(n, seed=0):
    """Makes random per-frame button masks from a small set of SMB-like inputs.

    Args:
        n: The number of frames.
        seed: The random seed, so the buttons are reproducible.

    Returns:
        A uint8 array of `n` button masks.
    """
    rng = np.random.default_rng(seed)
    return rng.choice([0, RIGHT, RIGHT | A, B, START], size=n).astype(np.uint8)


def convert(path, rom, game=None, **kwargs):
    """Converts a recording with demos.load_demo on a fresh core.

    Args:
        path: Path to the recording.
        rom: Path to the ROM to replay it on.
        game: The GameSpec instance to use; defaults to NullSpec().
        **kwargs: Passed on to demos.load_demo.

    Returns:
        The converted demo.
    """
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
    """Maps a button mask to the closest of SMB's actions.

    Args:
        mask: The button mask held.
        expected: The index of the action it should map to.
    """
    assert demos.nearest_action(mask, SmbSpec.actions) == expected


def test_window_mask_most_common():
    """Picks the mask held for the most frames of the window."""
    assert demos.window_mask(np.array([RIGHT, A, A, A], dtype=np.uint8)) == A


def test_window_mask_tie_goes_to_last_held():
    """Breaks a tie between masks in favour of the one held last."""
    assert demos.window_mask(np.array([RIGHT, RIGHT, A, A], dtype=np.uint8)) == A
    assert demos.window_mask(np.array([A, A, RIGHT, RIGHT], dtype=np.uint8)) == RIGHT


def test_mask_name():
    """Names an empty mask NOOP and joins button names with "+"."""
    assert demos.mask_name(0) == "NOOP"
    assert demos.mask_name(RIGHT | A) == "R+A"


# --- Replaying recordings ---


def test_pair_count(tmp_path, nestest_rom):
    """Makes one pair per full window except the first.

    The recording has 50 full windows; the first (no valid image) and the 3 trailing frames are dropped.
    """
    demo = convert(record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(4 * 50 + 3)), nestest_rom)
    assert len(demo) == 49
    assert demo.observations.shape == (49, nes_py.OBS_SIZE, nes_py.OBS_SIZE, 1)
    assert demo.observations.dtype == np.uint8
    assert demo.actions.dtype == np.int64
    assert demo.segment_count == 1


def test_observations_match_replay(tmp_path, nestest_rom):
    """Matches each observation to the screen from replaying the recording directly.

    Pair i's observation is the screen after replaying windows 0..i, i.e. just before window i + 1.
    """
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
    """Gives the 2 KB of CPU RAM as observations when `obs_type="ram"`."""
    demo = convert(record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(4 * 10)), nestest_rom, obs_type="ram")
    assert demo.observations.shape == (9, 2048)
    assert demo.observations.dtype == np.uint8


def test_actions_and_remaps(tmp_path, nestest_rom):
    """Maps each window to an action and counts masks that aren't one of the game's actions."""
    windows = [0, RIGHT, B, LEFT | A, RIGHT | DOWN, RIGHT]
    buttons = np.repeat(np.array(windows, dtype=np.uint8), 4)
    demo = convert(record(tmp_path / "a.nesdemo", nestest_rom, buttons), nestest_rom, game=SmbActionsSpec())

    np.testing.assert_array_equal(demo.actions, [RIGHT_I, NOOP_I, A_I, RIGHT_I, RIGHT_I])
    assert demo.remapped == {(B, NOOP_I): 1, (LEFT | A, A_I): 1, (RIGHT | DOWN, RIGHT_I): 1}
    assert demo.remap_count == 3


def test_segments(tmp_path, nestest_rom):
    """Splits the demo into segments at terminations and resumes once the game is playable.

    Terminated after windows 9 and 29, and not yet playable after window 10: windows 10, 11 and 30 are skipped, and
    segments start at the first pair (window 1) and at windows 12 and 31.
    """
    path = record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(4 * 50))
    demo = convert(path, nestest_rom, game=SmbActionsSpec(ends_after=(9, 29), unplayable_after=(10,)))
    assert len(demo) == 46
    np.testing.assert_array_equal(np.flatnonzero(demo.episode_starts), [0, 9, 27])
    assert demo.segment_count == 3


def test_recording_starting_outside_play(tmp_path, nestest_rom):
    """Skips windows until the game is playable.

    Not playable at the start or after windows 0 and 1, so the first pair is window 3.
    """
    path = record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(4 * 50))
    demo = convert(path, nestest_rom, game=SmbActionsSpec(unplayable_after=(-1, 0, 1)))
    assert len(demo) == 47
    assert demo.segment_count == 1


def test_not_playing_inside_a_segment_is_ignored(tmp_path, nestest_rom):
    """Keeps a segment going when `playing` turns false inside it.

    Inside a segment only `terminated` ends it, so states like pipes or power-ups don't split it.
    """

    class PlayableOnlyAtStart(SmbActionsSpec):
        """An SmbActionsSpec that is playable only the first time it's asked.

        Attributes:
            asked: The number of playing() calls so far.
        """

        def __init__(self):
            super().__init__()
            self.asked = 0

        def playing(self, ram):
            """Reports the game as playable only on the first call.

            Args:
                ram: The CPU RAM, ignored.

            Returns:
                True on the first call, False after.
            """
            self.asked += 1
            return self.asked == 1

    path = record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(4 * 50))
    demo = convert(path, nestest_rom, game=PlayableOnlyAtStart())
    assert len(demo) == 49
    assert demo.segment_count == 1


def test_paused_windows_skipped_without_ending_the_segment(tmp_path, nestest_rom):
    """Skips and counts paused windows without splitting the segment.

    Paused at the start of windows 5-7 and 20: those pairs are skipped and counted, in one segment. A paused window
    outside a segment (window 0 is never a pair) isn't counted.
    """
    path = record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(4 * 50))
    demo = convert(path, nestest_rom, game=SmbActionsSpec(paused_at=(0, 5, 6, 7, 20)))
    assert len(demo) == 45
    assert demo.paused == 4
    assert demo.segment_count == 1


@pytest.mark.parametrize("frames", [0, 7])
def test_too_short_for_a_pair(tmp_path, nestest_rom, frames):
    """Gives an empty demo, with correctly shaped observations, when there's no second full window.

    Args:
        frames: The length of the recording in frames.
    """
    demo = convert(record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(frames)), nestest_rom)
    assert len(demo) == 0
    assert demo.observations.shape == (0, nes_py.OBS_SIZE, nes_py.OBS_SIZE, 1)


def test_different_rom_rejected(tmp_path, nestest_rom):
    """Rejects a recording whose ROM checksum doesn't match the ROM it's replayed on."""
    path = record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(40), checksum_xor=1)
    with pytest.raises(ValueError, match="different ROM"):
        convert(path, nestest_rom)


@pytest.mark.parametrize(
    "kwargs", [{"obs_type": "rgb"}, {"frame_skip": 0}], ids=["obs_type", "frame_skip"]
)
def test_bad_arguments(tmp_path, nestest_rom, kwargs):
    """Rejects an unsupported `obs_type` or a `frame_skip` below 1.

    Args:
        kwargs: The bad argument passed to load_demo.
    """
    path = record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(40))
    with pytest.raises(ValueError):
        convert(path, nestest_rom, **kwargs)


# --- Loading several recordings ---


def test_load_demos_searches_folders(tmp_path, nestest_rom):
    """Finds .nesdemo files in subfolders, in sorted order, and ignores other files."""
    (tmp_path / "sub").mkdir()
    record(tmp_path / "b.nesdemo", nestest_rom, random_buttons(40))
    record(tmp_path / "sub" / "a.nesdemo", nestest_rom, random_buttons(40))
    (tmp_path / "notes.txt").write_text("not a recording")

    loaded = demos.load_demos([tmp_path], nestest_rom, game="null")
    assert [demo.path for demo in loaded] == [str(tmp_path / "b.nesdemo"), str(tmp_path / "sub" / "a.nesdemo")]


@pytest.mark.parametrize("as_type", [str, lambda p: p], ids=["str", "path"])
def test_load_demos_single_path(tmp_path, nestest_rom, as_type):
    """Accepts one folder or file not wrapped in a list.

    A str isn't split into characters, and a Path isn't iterated.

    Args:
        as_type: Turns the Path into the type passed to load_demos.
    """
    record(tmp_path / "a.nesdemo", nestest_rom, random_buttons(40))
    record(tmp_path / "b.nesdemo", nestest_rom, random_buttons(40))

    loaded = demos.load_demos(as_type(tmp_path), nestest_rom, game="null")
    assert [demo.path for demo in loaded] == [str(tmp_path / "a.nesdemo"), str(tmp_path / "b.nesdemo")]
    loaded = demos.load_demos(as_type(tmp_path / "a.nesdemo"), nestest_rom, game="null")
    assert [demo.path for demo in loaded] == [str(tmp_path / "a.nesdemo")]


def test_load_demos_generator_named_in_error(tmp_path, nestest_rom):
    """Names the searched folders in the "no recordings" error even when they come from a generator.

    A generator is read once, so the error can't re-read it to list the folders.
    """
    with pytest.raises(ValueError, match="No .nesdemo files") as error:
        demos.load_demos((p for p in [tmp_path]), nestest_rom, game="null")
    assert "in []" not in str(error.value)


def test_load_demos_no_recordings(tmp_path, nestest_rom):
    """Raises when the folders hold no .nesdemo files."""
    with pytest.raises(ValueError, match="No .nesdemo files"):
        demos.load_demos([tmp_path], nestest_rom, game="null")


def test_load_demos_unknown_game(tmp_path, nestest_rom):
    """Raises when `game` isn't a registered GameSpec."""
    with pytest.raises(ValueError, match="Invalid game"):
        demos.load_demos([tmp_path], nestest_rom, game="nope")


def test_cli_report(tmp_path, nestest_rom, monkeypatch, capsys):
    """Prints per-recording and total counts, with remaps listed most frequent first."""
    monkeypatch.setitem(GAMES, "smb_actions", SmbActionsSpec)
    windows = [0, B, B, LEFT | A, RIGHT]
    record(tmp_path / "a.nesdemo", nestest_rom, np.repeat(np.array(windows, dtype=np.uint8), 4))

    demos.main(["--rom", str(nestest_rom), "--game", "smb_actions", str(tmp_path)])
    out = capsys.readouterr().out
    assert "4 pairs, 1 segments, 3 remapped (75.0%), 0 paused windows skipped" in out
    assert "Total: 4 pairs from 1 recordings, 3 remapped (75.0%), 0 paused windows skipped" in out
    assert out.index("B -> NOOP: 2") < out.index("L+A -> A: 1")
