"""Saving and finding training checkpoints, shared by bc_train.py, train.py and play.py.

Training writes checkpoints while `play.py --follow` reads them from another process, so a reader must never see a
half-written file. save_atomic() writes each checkpoint to a temp file in the same folder, then renames it into place
with os.replace(), which swaps the file in one step: a reader sees either the old checkpoint or the finished new one.

Checkpoints are named `step_<n>.zip`, where `n` is the gradient step (bc_train.py) or the timestep (train.py).
Temp files start with "." and end in ".tmp", so latest_checkpoint() never picks one up. Temp files left behind by a
process killed mid-save aren't cleaned up: another process could still be writing one.
"""

import os
import re
import tempfile
import time
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

    Args:
        directory: The run's checkpoint folder, e.g. "checkpoints/<run>".
        step: The gradient step or timestep.

    Returns:
        `directory/step_<step>.zip`.
    """
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

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
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
