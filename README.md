# nes-agents

Train AI agents to play NES games on [nes-emulator-cpp](https://github.com/StevenH34/nes-emulator-cpp).

The approach: an agent first learns by copying recorded human play (behaviour cloning), then
improves with reinforcement learning (PPO with a KL penalty).

The emulator runs through its `nes_py` Python module, which is installed from a pinned tag of nes-emulator-cpp.
Gameplay is recorded in the emulator's desktop app (F10), and trained policies can be exported to ONNX and watched in that app.

## Setup

**Prerequisites**

- Python 3.14+
- A C++23 compiler, CMake and Ninja: installing builds `nes_py` from the emulator's source.
    - Windows: Visual Studio Build Tools with the "Desktop development with C++" workload and the
      "C++ Clang tools for Windows" component
    - macOS: Xcode Command Line Tools (`xcode-select --install`), Xcode 16+ for full C++23 support
    - Linux: GCC 14+ or Clang 18+

**Install** into a virtual environment:

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
python -m pip install -e ".[test]"
```

Extras: `train` (PyTorch, Stable-Baselines3, TensorBoard), `play` (OpenCV playback window), `export` (ONNX),
`test` (pytest). Combine them as needed, e.g. `".[train,play,test]"`.

## ROMs

```
set SMB_ROM=C:\path\to\smb.nes          # Windows (cmd)
$env:SMB_ROM = "C:\path\to\smb.nes"     # Windows (PowerShell)
export SMB_ROM=/path/to/smb.nes         # macOS / Linux
```

Only Mapper 0 games are supported by the emulator so far (e.g. Super Mario Bros., Donkey Kong, Balloon Fight,
Excitebike, Ice Climber, Pac-Man).

## Testing

`python -m pytest`

Tests that need a commercial ROM are skipped unless its environment variable (e.g. `SMB_ROM`) is set. The rest run against the bundled `tests/roms/nestest.nes`.

## License

GPL-3.0, see [LICENSE](LICENSE).
