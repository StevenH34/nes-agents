"""Tests for checkpoint saving and lookup.

Fake models stand in for SB3 models, so no SB3, ROM or GPU is needed.
"""

import os

import pytest

from nes_gym import checkpoints
from nes_gym.checkpoints import checkpoint_path, checkpoint_step, latest_checkpoint, save_atomic


class FakeModel:
    """A model whose save() writes fixed bytes to the open file it's given.

    Args:
        data: The bytes to write.
    """

    def __init__(self, data=b"weights"):
        self.data = data

    def save(self, f):
        """Writes `data` to `f`.

        Args:
            f: The open binary file to save to.
        """
        f.write(self.data)


class FailingModel:
    """A model whose save() writes a few bytes, then fails partway."""

    def save(self, f):
        """Writes some bytes to `f`, then raises.

        Args:
            f: The open binary file to save to.

        Raises:
            RuntimeError: Always.
        """
        f.write(b"part")
        raise RuntimeError("boom")


def files(directory):
    """Lists everything in a folder.

    Args:
        directory: The folder to list.

    Returns:
        The sorted names of the folder's entries.
    """
    return sorted(os.listdir(directory))


def locked_replace(monkeypatch, failures):
    """Patches os.replace to raise PermissionError, as on Windows while another process has the target open.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        failures: How many calls fail before calls go through to the real os.replace; None fails every call.

    Returns:
        A list that records the arguments of every call.
    """
    real_replace = os.replace
    calls = []

    def replace(src, dst):
        calls.append((src, dst))
        if failures is None or len(calls) <= failures:
            raise PermissionError("target is open in another process")
        real_replace(src, dst)

    monkeypatch.setattr(checkpoints.os, "replace", replace)
    return calls


def test_checkpoint_path(tmp_path):
    """checkpoint_path names checkpoints step_<n>.zip, and checkpoint_step reads n back."""
    path = checkpoint_path(tmp_path, 42)
    assert path == tmp_path / "step_42.zip"
    assert checkpoint_step(path) == 42


@pytest.mark.parametrize(
    "name", ["step_x.zip", "step_5.zip.tmp", ".step_5.zip.abc.tmp", "step_5", "notes.txt", "step_-1.zip"]
)
def test_checkpoint_step_rejects_bad_names(name):
    """checkpoint_step raises for names that aren't step_<n>.zip."""
    with pytest.raises(ValueError):
        checkpoint_step(name)


def test_save_atomic_writes_file(tmp_path):
    """save_atomic creates the parent folder and leaves only the finished file."""
    path = tmp_path / "run" / "step_1.zip"
    save_atomic(FakeModel(), path)
    assert path.read_bytes() == b"weights"
    assert files(path.parent) == ["step_1.zip"]


def test_save_atomic_replaces_existing(tmp_path):
    """Saving to an existing checkpoint replaces it."""
    path = tmp_path / "step_1.zip"
    save_atomic(FakeModel(b"old"), path)
    save_atomic(FakeModel(b"new"), path)
    assert path.read_bytes() == b"new"
    assert files(tmp_path) == ["step_1.zip"]


def test_save_atomic_failure_keeps_old_file(tmp_path):
    """A failed save raises, leaves the existing checkpoint unchanged and removes its temp file."""
    path = tmp_path / "step_1.zip"
    save_atomic(FakeModel(b"old"), path)
    with pytest.raises(RuntimeError, match="boom"):
        save_atomic(FailingModel(), path)
    assert path.read_bytes() == b"old"
    assert files(tmp_path) == ["step_1.zip"]


def test_save_atomic_failure_without_old_file(tmp_path):
    """A failed first save raises and leaves the folder empty."""
    with pytest.raises(RuntimeError, match="boom"):
        save_atomic(FailingModel(), tmp_path / "step_1.zip")
    assert files(tmp_path) == []


@pytest.mark.parametrize("name", ["step_1", "step_1.pt", "step_1.zip.bak"])
def test_save_atomic_requires_zip(tmp_path, name):
    """save_atomic raises for paths not ending in .zip, without writing anything."""
    with pytest.raises(ValueError):
        save_atomic(FakeModel(), tmp_path / name)
    assert files(tmp_path) == []


def test_save_atomic_retries_while_locked(tmp_path, monkeypatch):
    """save_atomic retries os.replace while the target is open elsewhere, sleeping between attempts."""
    calls = locked_replace(monkeypatch, failures=2)
    sleeps = []
    monkeypatch.setattr(checkpoints.time, "sleep", sleeps.append)
    path = tmp_path / "step_1.zip"
    save_atomic(FakeModel(), path)
    assert path.read_bytes() == b"weights"
    assert files(tmp_path) == ["step_1.zip"]
    assert len(calls) == 3
    assert sleeps == [checkpoints.REPLACE_DELAY] * 2


def test_save_atomic_gives_up_when_locked(tmp_path, monkeypatch):
    """After REPLACE_TRIES attempts save_atomic raises, keeping the old checkpoint and removing its temp file."""
    path = tmp_path / "step_1.zip"
    save_atomic(FakeModel(b"old"), path)
    calls = locked_replace(monkeypatch, failures=None)
    monkeypatch.setattr(checkpoints, "REPLACE_TRIES", 3)
    monkeypatch.setattr(checkpoints.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        save_atomic(FakeModel(b"new"), path)
    assert len(calls) == 3
    assert path.read_bytes() == b"old"
    assert files(tmp_path) == ["step_1.zip"]


def test_save_atomic_passes_open_file(tmp_path):
    """The model is given an open, writable file rather than a path, so save_atomic controls closing it."""
    seen = {}

    class RecordingModel:
        def save(self, f):
            seen["is_path"] = isinstance(f, (str, os.PathLike))
            seen["writable"] = f.writable()
            seen["closed"] = f.closed

    save_atomic(RecordingModel(), tmp_path / "step_1.zip")
    assert seen == {"is_path": False, "writable": True, "closed": False}


@pytest.mark.skipif(os.name != "posix", reason="permission bits only apply on Linux/macOS")
def test_save_atomic_uses_umask_permissions(tmp_path):
    """Checkpoints get the umask's normal permissions, like a plain save, not mkstemp's owner-only 0600."""
    old_umask = os.umask(0o022)
    try:
        path = tmp_path / "step_1.zip"
        save_atomic(FakeModel(), path)
    finally:
        os.umask(old_umask)
    assert path.stat().st_mode & 0o777 == 0o644


def test_latest_checkpoint_missing_dir(tmp_path):
    """A folder that doesn't exist has no latest checkpoint."""
    assert latest_checkpoint(tmp_path / "missing") is None


def test_latest_checkpoint_empty_dir(tmp_path):
    """An empty folder has no latest checkpoint."""
    assert latest_checkpoint(tmp_path) is None


def test_latest_checkpoint_compares_numbers(tmp_path):
    """Steps are compared as numbers, so step_1000 beats step_900."""
    for step in (2, 900, 1000):
        checkpoint_path(tmp_path, step).write_bytes(b"")
    assert latest_checkpoint(tmp_path) == tmp_path / "step_1000.zip"


def test_latest_checkpoint_ignores_other_entries(tmp_path):
    """Temp files, other files and folders are ignored, even with higher steps."""
    for name in ("step_5.zip", ".step_99.zip.abc.tmp", "step_x.zip", "notes.txt", "step_50.zip.bak"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "step_77.zip").mkdir()
    assert latest_checkpoint(tmp_path) == tmp_path / "step_5.zip"
    (tmp_path / "step_5.zip").unlink()
    assert latest_checkpoint(tmp_path) is None


def test_latest_checkpoint_sees_save_atomic(tmp_path):
    """latest_checkpoint finds what save_atomic saves under checkpoint_path."""
    for step in (1, 10):
        save_atomic(FakeModel(), checkpoint_path(tmp_path, step))
    assert latest_checkpoint(tmp_path) == tmp_path / "step_10.zip"
