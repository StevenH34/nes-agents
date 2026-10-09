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

To train on an NVIDIA GPU, install PyTorch's CUDA build first, so the `train` extra doesn't pull in the CPU-only
one. Pick the CUDA version for your setup on [pytorch.org](https://pytorch.org/get-started/locally/), e.g.:

```
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[train,play,test]"
python -c "import torch; print(torch.cuda.is_available())"
```

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

## Behaviour cloning

`bc_train.py` (needs the `train` extra) trains a policy to copy your recordings. About 20–30 minutes of play, as
many short clips through 1-1 and 1-2, is a reasonable start.

```
python bc_train.py --rom C:\path\to\smb.nes C:\path\to\recordings
```

- The policy is the same CNN that PPO uses, so the result loads straight into later RL training and `play.py`.
- Whole segments (about 10% of the pairs) are held out for validation, so near-identical neighbouring frames can't
  end up on both sides. At least 6 segments are needed.
- Rare actions are weighted up in the loss, so the agent can't do well by only pressing Right.
- The epoch with the lowest validation loss is saved to `bc_policy.zip` (`--out`). Ctrl-C stops early and still
  saves the best epoch finished so far.
- The saved policy then plays 1 argmax episode and 5 sampled ones, and reports the distance reached, how often it
  reached the flag, and the episode length.
- Checkpoints go to `checkpoints/<run>/step_<n>.zip` every 500 gradient steps, and TensorBoard logs to
  `runs/<run>` (`tensorboard --logdir runs`).

Options: `--game` (default `smb`), `--run` (default `bc-<date>-<time>`), `--out`, `--epochs 30`,
`--batch-size 256`, `--lr 1e-4`, `--val-frac 0.1`, `--checkpoint-every 500`, `--eval-episodes 5`,
`--eval-max-steps 3000`, `--device auto` (CUDA if available), `--seed 0`.

## Watching an agent

`play.py` (needs the `train` and `play` extras) shows an agent playing in a window, scaled 3×:

```
python play.py --rom C:\path\to\smb.nes bc_policy.zip
python play.py --rom C:\path\to\smb.nes --follow checkpoints\bc-20261008-120000
```

- With `--follow`, it watches training as it happens: it waits for the first checkpoint, then switches to newer
  ones between episodes. The window title shows the checkpoint's step. Checkpoints are the weights as training
  goes, so late ones may be overfit; `bc_policy.zip` is the best epoch.
- It runs on the CPU by default (`--device`), so it doesn't slow down training on the GPU.
- Actions are the agent's top choice by default. `--stochastic` samples them, as training does.
- The game and the top choices are deterministic, so plain playback plays 1 episode; `--follow` and
  `--stochastic` play until you quit. `--episodes N` overrides this (0 plays forever).
- Press q or Esc, or close the window, to quit.

Options: `--game`, `--max-steps 3000`, `--scale 3`, `--fps 15` (agent steps per second; 15 is real speed).

## Testing

`python -m pytest`

Tests that need a commercial ROM are skipped unless its environment variable (e.g. `SMB_ROM`) is set. The rest run against the bundled `tests/roms/nestest.nes`. Tests that need Stable-Baselines3 are skipped without the `train` extra.

## License

GPL-3.0, see [LICENSE](LICENSE).
