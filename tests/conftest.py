import os
from pathlib import Path

import pytest

NESTEST_ROM = Path(__file__).resolve().parent / "roms" / "nestest.nes"


@pytest.fixture
def nestest_rom():
    """Provide the path to the bundled nestest ROM.

    Returns:
        Path: Absolute path to ``tests/roms/nestest.nes``.
    """
    return NESTEST_ROM


@pytest.fixture
def smb_rom():
    """Provide the path to a Super Mario Bros. ROM from the environment.

    The ROM is not bundled with the repository, so its location is read from
    the ``SMB_ROM`` environment variable.

    Returns:
        Path: Path to the Super Mario Bros. ROM named by ``SMB_ROM``.

    Raises:
        pytest.skip.Exception: If ``SMB_ROM`` is not set, which skips the
            requesting test.
    """
    path = os.environ.get("SMB_ROM")
    if not path:
        pytest.skip("SMB_ROM is not set")
    return Path(path)
