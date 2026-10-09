"""Watches a trained agent play, in an OpenCV window scaled 3×.

Plays episodes with a checkpoint or a saved policy such as bc_policy.zip. With `--follow <checkpoint dir>` it watches
training as it happens: it plays with the newest `step_<n>.zip` in the folder and, between episodes, switches to a
newer one when training has saved it. The window title shows the checkpoint's step.

It runs as its own process, so training never pays for rendering, and on the CPU by default (`--device`) so it
doesn't compete with training for the GPU. Actions are argmax by default, showing what the agent has learned rather
than its exploration noise; `--stochastic` samples them, as training does.

The env and argmax actions are deterministic, so only `--stochastic` or a newer checkpoint makes one episode differ
from the last. Plain playback therefore plays 1 episode by default, and `--follow` or `--stochastic` play until you
quit; `--episodes` overrides both.

The window updates once per agent step (every 4 frames), so at the default 15 steps a second the game runs at real
speed but looks choppier than the emulator. Press q or Esc, or close the window, to quit.

Needs the `play` extra (opencv-python).

Example:
    python play.py --rom smb.nes bc_policy.zip
    python play.py --rom smb.nes --follow checkpoints/bc-20261008-120000
"""

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from nes_gym.checkpoints import CHECKPOINT_RE, checkpoint_step, latest_checkpoint
from nes_gym.cli import float_at_least, non_negative_int, positive_int
from nes_gym.games import GAMES

WINDOW = "nes-agents"
# Seconds between looks for a first checkpoint with --follow.
WAIT_POLL = 1.0


def scale_frame(rgb: np.ndarray, scale: int) -> np.ndarray:
    """Enlarges an RGB frame with sharp pixels and converts it to the BGR order OpenCV shows.

    Args:
        rgb: (H, W, 3) uint8 frame.
        scale: The whole-number enlargement.

    Returns:
        (H × scale, W × scale, 3) uint8, BGR, contiguous.
    """
    big = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return np.ascontiguousarray(big[..., ::-1])


def load_model(path: str | Path, venv, device: str):
    """Loads a saved PPO model and checks it fits the env.

    Args:
        path: The .zip to load.
        venv: The env it will play.
        device: The torch device to run it on.

    Returns:
        The PPO model.

    Raises:
        ValueError: The model's observation or action space doesn't match the env's, e.g. a RAM model on pixels.
    """
    from stable_baselines3 import PPO

    model = PPO.load(path, device=device)
    if model.observation_space != venv.observation_space:
        raise ValueError(
            f"{path}: the model expects observations {model.observation_space}, "
            f"but the env gives {venv.observation_space}"
        )
    if model.action_space != venv.action_space:
        raise ValueError(f"{path}: the model has actions {model.action_space}, the env {venv.action_space}")
    return model


class Follower:
    """Finds and loads the newest checkpoint in a folder that training is writing to.

    Args:
        directory: The run's checkpoint folder.
        loader: Loads a checkpoint path into a model.

    Attributes:
        step: The step of the loaded checkpoint, or None before the first.
    """

    def __init__(self, directory: str | Path, loader: Callable):
        self.directory = directory
        self.loader = loader
        self.step = None

    def poll(self):
        """Loads the newest checkpoint if it's newer than the current one.

        A checkpoint deleted between being found and being loaded is skipped; the next poll looks again.

        Returns:
            (model, step) for a newly loaded checkpoint, or None if there's nothing newer.
        """
        path = latest_checkpoint(self.directory)
        if path is None:
            return None
        step = checkpoint_step(path)
        if self.step is not None and step <= self.step:
            return None
        try:
            model = self.loader(path)
        except FileNotFoundError:
            # Enough while training never deletes or overwrites a step_<n>.zip. If checkpoints are ever pruned
            # (keep the last N), also catch PermissionError (Windows can't open a file being deleted) and
            # zipfile.BadZipFile.
            return None
        self.step = step
        return model, step


def play_episode(model, venv, deterministic: bool, show: Callable[[np.ndarray], bool]) -> dict | None:
    """Plays one episode, showing each frame.

    Args:
        model: The model to play.
        venv: A one-env VecEnv made with render_mode="rgb_array".
        deterministic: True for argmax actions, False to sample them.
        show: Called with the full RGB frame after the reset and after each step except the last (by then SB3 has
            already reset the env); returns False to stop.

    Returns:
        A dict with "distance" (the furthest info["x"], 0 if the game doesn't report it), "flag" (whether
        info["flag_get"] was ever True) and "length" (steps), or None if `show` stopped the episode.
    """
    # SB3 already reset the env when the last episode ended, so this reset is usually a second one. It's kept on
    # purpose: it makes the episode start cleanly on a fresh env or one stopped mid-episode, for one extra reset.
    obs = venv.reset()
    if not show(venv.get_images()[0]):
        return None
    distance, flag, length = 0, False, 0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, dones, infos = venv.step(action)
        length += 1
        distance = max(distance, infos[0].get("x", 0))
        flag = flag or bool(infos[0].get("flag_get", False))
        if dones[0]:
            return {"distance": distance, "flag": flag, "length": length}
        if not show(venv.get_images()[0]):
            return None


def default_episodes(episodes: int | None, follow: bool, stochastic: bool) -> int:
    """Returns how many episodes to play, 0 meaning forever.

    The env and argmax actions are deterministic, so without --follow or --stochastic every episode is an exact
    copy of the first: one is enough. Following plays forever to keep up with new checkpoints, and sampled episodes
    differ, so those play forever too. An explicit --episodes always wins.

    Args:
        episodes: The --episodes value, or None if it wasn't given.
        follow: Whether --follow was given.
        stochastic: Whether --stochastic was given.

    Returns:
        The number of episodes, or 0 for forever.
    """
    if episodes is not None:
        return episodes
    return 0 if follow or stochastic else 1


def main(argv=None) -> None:
    """Plays a saved agent from the command line.

    Args:
        argv: The command-line arguments, or None to use sys.argv.
    """
    parser = argparse.ArgumentParser(description="Watch a trained agent play")
    parser.add_argument("checkpoint", help="a saved .zip, or with --follow the checkpoint folder to watch")
    parser.add_argument("--rom", required=True, help="the game's ROM")
    parser.add_argument("--game", default="smb", choices=sorted(GAMES))
    parser.add_argument("--follow", action="store_true", help="switch to newer checkpoints between episodes")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    parser.add_argument("--device", default="cpu", help="torch device for the policy")
    parser.add_argument(
        "--episodes",
        type=non_negative_int,
        help="episodes to play; 0 plays forever (default: 1 for plain argmax playback, else forever)",
    )
    parser.add_argument("--max-steps", type=positive_int, default=3000, help="steps before an episode is cut off")
    parser.add_argument("--scale", type=positive_int, default=3)
    parser.add_argument("--fps", type=float_at_least(0.1), default=15.0, help="agent steps per second, at least 0.1")
    args = parser.parse_args(argv)
    target = Path(args.checkpoint)
    if args.follow and target.exists() and not target.is_dir():
        parser.error(f"--follow takes a checkpoint folder, got the file {args.checkpoint!r}")
    if not args.follow and not target.is_file():
        parser.error(f"no such file: {args.checkpoint!r}")

    from nes_gym.vec_env import make_vec_env

    venv = make_vec_env(args.rom, args.game, max_episode_steps=args.max_steps, render_mode="rgb_array")
    last_shown = time.perf_counter()

    def show(frame: np.ndarray) -> bool:
        """Draws a frame, waits to keep the --fps pace, and returns False if the viewer quit."""
        nonlocal last_shown
        cv2.imshow(WINDOW, scale_frame(frame, args.scale))
        wait_ms = max(1, round((1 / args.fps - (time.perf_counter() - last_shown)) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        last_shown = time.perf_counter()
        if key in (ord("q"), 27):  # 27 is Esc.
            return False
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1

    def set_title(step: int | None) -> None:
        """Shows the checkpoint (and its step, if known) in the window title."""
        name = target.name if args.follow else target.stem
        cv2.setWindowTitle(WINDOW, name if step is None else f"{name} step {step}")

    episodes = default_episodes(args.episodes, args.follow, args.stochastic)

    def load(path: Path):
        """Loads a model, reporting one that doesn't fit the env as a usage error rather than a traceback."""
        try:
            return load_model(path, venv, args.device)
        except ValueError as e:
            parser.error(str(e))

    cv2 = None  # Imported once the model has loaded; see below.
    try:
        if args.follow:
            follower = Follower(target, load)
            print(f"waiting for a checkpoint in {target} (Ctrl+C to stop) ...")
            while (loaded := follower.poll()) is None:
                time.sleep(WAIT_POLL)
            model, step = loaded
            print(f"loaded step {step}")
        else:
            model = load(target)
            step = checkpoint_step(target) if CHECKPOINT_RE.fullmatch(target.name) else None
        # Imported and opened only now, so a wrong model is reported first. OpenCV handles window events only inside
        # waitKey(), so a window open during the wait above would show "Not Responding" and ignore q, Esc and the
        # close button.
        import cv2

        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        set_title(step)

        episode = 0
        while episodes == 0 or episode < episodes:
            if args.follow and (loaded := follower.poll()) is not None:
                model, step = loaded
                print(f"loaded step {step}")
                set_title(step)
            result = play_episode(model, venv, deterministic=not args.stochastic, show=show)
            if result is None:
                break
            episode += 1
            print(
                f"episode {episode}: distance {result['distance']}, flag {result['flag']}, length {result['length']}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        venv.close()
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
