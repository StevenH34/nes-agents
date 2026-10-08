"""Turn .nesdemo recordings (F10 in the emulator) into (observation, action) pairs for behaviour cloning.

Each recording is replayed through NesCore from its start state using the human's real per-frame buttons, so the
emulator reproduces exactly what happened. The frames are cut into windows of `frame_skip`, matching one NesEnv
step. Each window yields one pair:

- observation: what NesEnv would return before that step, i.e. the screen (or RAM) after the previous window;
- action: the most common button mask in the window, mapped to the closest of the game's actions.

The first window of each recording is dropped: a save state doesn't include the drawn image, so right after
load_state() the screen still shows whatever was drawn before (the same reason NesEnv caches its start images).

Recordings run on through deaths and respawns, but NesEnv ends the episode there. Pairs are therefore split into
segments: a segment ends on the window where the game reports `terminated`. Windows are then skipped until the
game reports `playing` again (for SMB: past the lives screen, castle walk and a level's walk-in), and a new
segment starts there. A recording that starts outside play (e.g. on the title screen) is skipped the same way
until play begins. Inside a segment only `terminated` is checked, so states NesEnv episodes also pass through
(pipes, power-ups) stay in the segment. `episode_starts` marks the first pair of each segment, so frame stacking
for training can restart there, as SB3's VecFrameStack does after a reset.

Windows that start while the game is `paused` are skipped too, but don't end the segment: the agent can't pause,
and the game resumes the same episode on the same frozen screen. Their count is reported.
"""

import argparse
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import nes_py
import numpy as np

from nes_gym.env import OBS_TYPES, RAM_SIZE
from nes_gym.games import GAMES
from nes_gym.games.base import GameSpec

RECORDING_SUFFIX = ".nesdemo"

BUTTON_NAMES = (
    (nes_py.BUTTON_UP, "U"),
    (nes_py.BUTTON_DOWN, "D"),
    (nes_py.BUTTON_LEFT, "L"),
    (nes_py.BUTTON_RIGHT, "R"),
    (nes_py.BUTTON_A, "A"),
    (nes_py.BUTTON_B, "B"),
    (nes_py.BUTTON_SELECT, "Select"),
    (nes_py.BUTTON_START, "Start"),
)


def mask_name(mask: int) -> str:
    """Returns a readable name for a button mask.

    Args:
        mask: The button mask.

    Returns:
        The pressed buttons joined with "+", e.g. "R+A", or "NOOP" if no buttons are pressed.
    """
    return "+".join(name for bit, name in BUTTON_NAMES if mask & bit) or "NOOP"


def nearest_action(mask: int, actions: tuple[int, ...]) -> int:
    """Finds the action closest to a button mask.

    Distance is the number of buttons that differ. Ties go to the earlier action. Buttons no action uses (e.g. Start,
    Down) add the same distance to every action, so they never change the choice: R+Down maps to R. Examples for
    SMB: B alone maps to NOOP, L+B to L, and L+A (equally far from A and L) to A.

    Args:
        mask: The button mask to map.
        actions: The game's actions, as button masks.

    Returns:
        The index into `actions` of the closest action.
    """
    return min(range(len(actions)), key=lambda i: ((mask ^ actions[i]).bit_count(), i))


def window_mask(masks: np.ndarray) -> int:
    """Picks the button mask that represents one window.

    Ties go to the mask held last: that's the player's current intent, and it carries on into the next window.

    Args:
        masks: The per-frame button masks in the window.

    Returns:
        The most common mask in the window.
    """
    seq = masks.tolist()
    counts = Counter(seq)
    best = max(counts.values())
    return next(m for m in reversed(seq) if counts[m] == best)


def _observe(core: nes_py.NesCore, obs_type: str) -> np.ndarray:
    """Returns the current observation, in the same form as NesEnv's.

    Args:
        core: The emulator core to observe.
        obs_type: "pixels" for the (84, 84, 1) screen, otherwise the (2048,) RAM.

    Returns:
        The observation as a uint8 array.
    """
    if obs_type == "pixels":
        return core.obs84()[..., np.newaxis]
    return core.ram()


@dataclass
class Demo:
    """The (observation, action) pairs from one recording.

    Attributes:
        path: The recording's file path.
        observations: (T, 84, 84, 1) or (T, 2048) uint8, the same as NesEnv's observations.
        actions: (T,) int64 indices into the game's actions.
        episode_starts: (T,) bool, True on the first pair of each segment.
        remapped: Maps (window mask, action index) to the number of windows whose mask wasn't one of the actions and
            was remapped.
        paused: The number of windows skipped because the game was paused.
    """

    path: str
    observations: np.ndarray
    actions: np.ndarray
    episode_starts: np.ndarray
    remapped: Counter = field(default_factory=Counter)
    paused: int = 0

    def __len__(self) -> int:
        """Returns the number of pairs."""
        return len(self.actions)

    @property
    def remap_count(self) -> int:
        """The number of windows whose mask was remapped to a different action."""
        return sum(self.remapped.values())

    @property
    def segment_count(self) -> int:
        """The number of segments, i.e. episodes, in the recording."""
        return int(self.episode_starts.sum())


def load_demo(
    path: str | os.PathLike,
    core: nes_py.NesCore,
    game: GameSpec,
    obs_type: str = "pixels",
    frame_skip: int = 4,
) -> Demo:
    """Replays one recording and converts it to (observation, action) pairs.

    Args:
        path: The .nesdemo file.
        core: The emulator core to replay on. It must have the recording's ROM loaded.
        game: The game the recording is of.
        obs_type: The observation type, one of OBS_TYPES.
        frame_skip: The number of frames per window, matching one NesEnv step.

    Returns:
        The recording's pairs.

    Raises:
        ValueError: If `obs_type` or `frame_skip` is invalid, or the recording was made on a different ROM.
    """
    if obs_type not in OBS_TYPES:
        raise ValueError(f"Invalid observation type: {obs_type}. Expected one of {OBS_TYPES}.")
    if frame_skip < 1:
        raise ValueError(f"Invalid frame skip: {frame_skip}. Must be at least 1.")

    path = os.fspath(path)
    recording = nes_py.load_recording(path)
    if recording.rom_checksum != core.rom_checksum():
        raise ValueError(
            f"{path}: recording was made with a different ROM (recording checksum {recording.rom_checksum:#010x}, "
            f"loaded ROM {core.rom_checksum():#010x})"
        )

    core.load_state(recording.start_state)
    nearest = [nearest_action(mask, game.actions) for mask in range(256)]
    buttons = recording.buttons
    observations, actions, episode_starts = [], [], []
    remapped = Counter()
    # Inside a segment: from the first playable window until the game reports terminated.
    in_segment = game.playing(core.ram())
    new_segment = True
    # Whether the game is paused at the start of the current window, i.e. on its observation.
    paused = game.paused(core.ram())
    paused_windows = 0

    # A trailing window shorter than frame_skip will be ignored.
    for k in range(len(buttons) // frame_skip):
        window = buttons[k * frame_skip : (k + 1) * frame_skip]
        if k > 0 and in_segment and paused:
            paused_windows += 1
        # Window 0 has no valid observation.
        elif k > 0 and in_segment:
            mask = window_mask(window)
            action = nearest[mask]
            if game.actions[action] != mask:
                remapped[(mask, action)] += 1
            observations.append(_observe(core, obs_type))
            actions.append(action)
            episode_starts.append(new_segment)
            new_segment = False

        # Replay the player's real inputs.
        for mask in window:
            core.step(int(mask))

        ram = core.ram()
        paused = game.paused(ram)
        if in_segment:
            in_segment = not game.terminated(ram)
        elif game.playing(ram):
            in_segment = True
            new_segment = True

    if obs_type == "pixels":
        empty_shape = (0, nes_py.OBS_SIZE, nes_py.OBS_SIZE, 1)
    else:
        empty_shape = (0, RAM_SIZE)

    return Demo(
        path=path,
        observations=np.stack(observations) if observations else np.empty(empty_shape, dtype=np.uint8),
        actions=np.array(actions, dtype=np.int64),
        episode_starts=np.array(episode_starts, dtype=bool),
        remapped=remapped,
        paused=paused_windows,
    )


def find_recordings(paths: Iterable[str | os.PathLike]) -> list[Path]:
    """Finds the recordings in a list of files and folders.

    Args:
        paths: Files, kept as given, and folders, searched recursively for *.nesdemo.

    Returns:
        The files in the order given, with each folder's recordings sorted in its place.
    """
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(p.rglob(f"*{RECORDING_SUFFIX}")))
        else:
            files.append(p)
    return files


def load_demos(
    paths: str | os.PathLike | Iterable[str | os.PathLike],
    rom_path: str | os.PathLike,
    game: str = "smb",
    obs_type: str = "pixels",
    frame_skip: int = 4,
) -> list[Demo]:
    """Converts recordings made on one ROM to (observation, action) pairs.

    Args:
        paths: One path or several: files, or folders searched recursively for *.nesdemo.
        rom_path: The ROM the recordings were made on.
        game: The game's key in GAMES.
        obs_type: The observation type, one of OBS_TYPES.
        frame_skip: The number of frames per window, matching one NesEnv step.

    Returns:
        One Demo per recording, in the order find_recordings returns them.

    Raises:
        ValueError: If `game` is unknown, no recordings are found, `obs_type` or `frame_skip` is invalid, or a
            recording was made on a different ROM (raised on the first such recording).
    """
    if game not in GAMES:
        raise ValueError(f"Invalid game: {game!r}. Expected one of {sorted(GAMES)}.")

    # A single path, not its characters (a str is iterable) or a TypeError (a Path isn't).
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = list(paths)

    files = find_recordings(paths)
    if not files:
        raise ValueError(f"No {RECORDING_SUFFIX} files found in {[os.fspath(p) for p in paths]}")

    core = nes_py.NesCore(os.fspath(rom_path))
    spec = GAMES[game]()

    return [load_demo(f, core, spec, obs_type=obs_type, frame_skip=frame_skip) for f in files]


def remap_report(demos: list[Demo], actions: tuple[int, ...], top: int = 10) -> str:
    """Summarizes the pairs and action remapping in a set of recordings.

    Args:
        demos: The converted recordings.
        actions: The game's actions, as button masks.
        top: The number of most common remaps to list.

    Returns:
        One line per recording with its pair, segment, remap and paused-window counts, a total line, then the `top`
        most common remaps across all recordings.
    """
    lines = []
    total_pairs = total_remaps = total_paused = 0
    remapped = Counter()

    for demo in demos:
        rate = demo.remap_count / len(demo) if len(demo) else 0.0
        lines.append(
            f"{demo.path}: {len(demo)} pairs, {demo.segment_count} segments, "
            f"{demo.remap_count} remapped ({rate:.1%}), {demo.paused} paused windows skipped"
        )
        total_pairs += len(demo)
        total_remaps += demo.remap_count
        total_paused += demo.paused
        remapped.update(demo.remapped)

    rate = total_remaps / total_pairs if total_pairs else 0.0
    lines.append(
        f"Total: {total_pairs} pairs from {len(demos)} recordings, {total_remaps} remapped ({rate:.1%}), "
        f"{total_paused} paused windows skipped"
    )
    for (mask, action), count in remapped.most_common(top):
        lines.append(f"  {mask_name(mask)} -> {mask_name(actions[action])}: {count}")

    return "\n".join(lines)


def main(argv=None) -> None:
    """Converts recordings from the command line and prints the remap report.

    Args:
        argv: The command-line arguments, or None to use sys.argv.
    """
    parser = argparse.ArgumentParser(description="Convert .nesdemo recordings and report action remapping")
    parser.add_argument("paths", nargs="+", help=f"{RECORDING_SUFFIX} files or folders containing them")
    parser.add_argument("--rom", required=True, help="the ROM the recordings were made on")
    parser.add_argument("--game", default="smb", choices=sorted(GAMES))
    parser.add_argument("--obs", default="pixels", choices=OBS_TYPES)
    parser.add_argument("--frame-skip", type=int, default=4)
    args = parser.parse_args(argv)

    demos = load_demos(args.paths, args.rom, game=args.game, obs_type=args.obs, frame_skip=args.frame_skip)
    print(remap_report(demos, GAMES[args.game].actions))


if __name__ == "__main__":
    main()
