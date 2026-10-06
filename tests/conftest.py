import os
from pathlib import Path

import pytest

NESTEST_ROM = Path(__file__).resolve().parent / "roms" / "nestest.nes"


@pytest.fixture
def nestest_rom():
    return NESTEST_ROM


@pytest.fixture
def smb_rom():
    """Path to a Super Mario Bros. ROM from $SMB_ROM; the test is skipped when it isn't set."""
    path = os.environ.get("SMB_ROM")
    if not path:
        pytest.skip("SMB_ROM is not set")
    return Path(path)
