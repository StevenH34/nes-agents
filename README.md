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

## Recordings to training data

Record your play in the emulator's desktop app with F10. Each clip is saved as a `.nesdemo` file in a `recordings`
folder next to the ROM. To check how the clips convert into `(observation, action)` pairs:

```
python -m nes_gym.demos --rom C:\path\to\smb.nes C:\path\to\recordings
```

Pass `.nesdemo` files or folders, which are searched for `.nesdemo` files. It prints, for each recording, the
number of pairs and segments, how many windows had to be remapped to the game's actions, how many paused windows
were skipped, and the most common remaps. Options: `--game` (default `smb`), `--obs pixels|ram`, `--frame-skip`
(default 4, matching the environment).

How a recording is converted:

- It is replayed from its start state with the exact buttons pressed, so the emulator reproduces the play.
- Each 4-frame window becomes one pair: the screen (or RAM) before the window, and the most common button
  combination during it, mapped to the closest of the game's actions.
- Only gameplay is kept. Deaths, level clears, lives screens, the title screen and pauses are skipped, and each
  stretch of play is marked as a separate segment.

A recording made on a different ROM is rejected. In Python, `load_demos` returns one `Demo` per recording:

```python
from nes_gym.demos import load_demos

demos = load_demos("recordings", "smb.nes")
demo = demos[0]
demo.observations, demo.actions, demo.episode_starts
```

## Testing

`python -m pytest`

Tests that need a commercial ROM are skipped unless its environment variable (e.g. `SMB_ROM`) is set. The rest run against the bundled `tests/roms/nestest.nes`.

## License

GPL-3.0, see [LICENSE](LICENSE).
