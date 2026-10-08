"""Saving and finding training checkpoints, shared by bc_train.py, train.py and play.py.

Training writes checkpoints while `play.py --follow` reads them from another process, so a reader must never see a
half-written file. save_atomic() writes each checkpoint to a temp file in the same folder, then renames it into place
with os.replace(), which swaps the file in one step: a reader sees either the old checkpoint or the finished new one.

Checkpoints are named `step_<n>.zip`, where `n` is the gradient step (bc_train.py) or the timestep (train.py).
Temp files start with "." and end in ".tmp", so latest_checkpoint() never picks one up. Temp files left behind by a
process killed mid-save aren't cleaned up: another process could still be writing one.
"""

import operator
import os
import re
import time
import uuid
from pathlib import Path
from typing import BinaryIO, Protocol

CHECKPOINT_RE = re.compile(r"step_(\d+)\.zip")

# On Windows, os.replace() fails with PermissionError while another process (e.g. play.py loading the previous
# checkpoint) has the target open. Loading takes well under a second, so retry for up to REPLACE_TRIES × REPLACE_DELAY.
REPLACE_TRIES = 20
REPLACE_DELAY = 0.1


class Saveable(Protocol):
    """Anything with SB3's `save()`, which also accepts an open binary file."""

    def save(self, path: BinaryIO) -> None: ...


def checkpoint_path(directory: str | os.PathLike, step: int) -> Path:
    """Returns the path of the checkpoint for a step.

    Only steps that checkpoint_step() and latest_checkpoint() can read back are accepted, so a bad step fails here
    instead of silently saving a checkpoint `play.py --follow` never finds.

    Args:
        directory: The run's checkpoint folder, e.g. "checkpoints/<run>".
        step: The gradient step or timestep: a non-negative integer, e.g. an int or np.int64.

    Returns:
        `directory/step_<step>.zip`.

    Raises:
        TypeError: `step` isn't an integer, or is a bool (f"{True}" would give "step_True.zip").
        ValueError: `step` is negative.
    """
    if isinstance(step, bool):
        raise TypeError(f"step must be an integer, not a bool: {step!r}")
    # operator.index() accepts Python and numpy integers and raises TypeError for floats, strings and None.
    step = operator.index(step)
    if step < 0:
        raise ValueError(f"step must be >= 0: {step}")
    return Path(directory) / f"step_{step}.zip"


def checkpoint_step(path: str | os.PathLike) -> int:
    """Reads the step from a checkpoint's file name.

    Args:
        path: A checkpoint path, named `step_<n>.zip`.

    Returns:
        `n`.

    Raises:
        ValueError: The file name isn't `step_<n>.zip`.
    """
    name = Path(path).name
    match = CHECKPOINT_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"not a checkpoint name (expected step_<n>.zip): {name!r}")
    return int(match.group(1))


def save_atomic(model: Saveable, path: str | os.PathLike) -> None:
    """Saves a model so that readers only ever see a complete file.

    The model is saved to a temp file in the same folder, flushed to disk, then renamed to `path` in one step. The
    temp file is opened here and passed to `model.save()` as an open file, so it's always closed, and can be deleted
    if saving fails (SB3 doesn't close a file it opened itself if saving fails partway). If anything fails, the temp
    file is deleted and an existing file at `path` is left untouched.

    The temp file is created with open(..., "xb"), which never overwrites an existing file and gives the file the
    normal permissions for the user's umask (usually 0644 on Linux/macOS), the same as a plain `model.save(path)`.
    tempfile.mkstemp() would make it 0600, and os.replace() keeps that, so readers running as another user couldn't
    load the checkpoint.

    Args:
        model: The model to save, e.g. an SB3 PPO model.
        path: Where to save it. Must end in ".zip". The parent folder is created if needed.

    Raises:
        ValueError: `path` doesn't end in ".zip".
        PermissionError: `path` stayed open in another process for the whole retry period.
    """
    path = Path(path)
    if path.suffix != ".zip":
        raise ValueError(f"checkpoint path must end in .zip: {str(path)!r}")
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    # Opened outside the try: if creating it fails, there's nothing of ours to delete.
    f = open(tmp, "xb")
    try:
        with f:
            model.save(f)
            f.flush()
            # Make sure the data is on disk before the rename, so a power cut can't leave an empty file at `path`.
            os.fsync(f.fileno())
        _replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _replace(src: Path, dst: Path) -> None:
    """Renames `src` to `dst` with os.replace(), retrying while `dst` is open in another process.

    Args:
        src: The finished temp file.
        dst: The checkpoint path to replace.

    Raises:
        PermissionError: `dst` was still open after REPLACE_TRIES attempts.
    """
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES - 1:
                raise
            time.sleep(REPLACE_DELAY)


def latest_checkpoint(directory: str | os.PathLike) -> Path | None:
    """Finds the newest checkpoint in a folder.

    Only regular files named exactly `step_<n>.zip` count, so temp files and anything else in the folder are ignored.
    Steps are compared as numbers: step_1000 is newer than step_900.

    Args:
        directory: The run's checkpoint folder.

    Returns:
        The path of the checkpoint with the highest step, or None if the folder doesn't exist or has no checkpoints.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for entry in directory.iterdir():
        match = CHECKPOINT_RE.fullmatch(entry.name)
        if match is None or not entry.is_file():
            continue
        step = int(match.group(1))
        if best is None or step > best[0]:
            best = (step, entry)
    return None if best is None else best[1]
