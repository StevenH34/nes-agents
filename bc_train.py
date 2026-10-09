"""Behaviour cloning: trains PPO's CNN policy to copy recorded human play, then plays it in the environment.

The policy is the same SB3 ActorCriticCnnPolicy that PPO uses, so `bc_policy.zip` loads straight into
`train.py --init` and `play.py`. Only the action-choosing parts are trained (the CNN, the policy MLP and the action
head), with class-weighted cross-entropy so rare actions count as much as NOOP and plain Right. The value head is
left untouched for Phase 5 to warm up.

Each epoch is validated on whole held-out segments (see nes_gym.bc), and the weights with the lowest weighted
validation loss are saved to `--out`. Checkpoints go to `checkpoints/<run>/step_<n>.zip` (n = gradient steps), so
`play.py --follow` can watch cloning, and TensorBoard logs go to `runs/<run>`. Checkpoints hold the weights as
training goes, overfitting included; `--out` holds the best epoch, so watch that with `play.py bc_policy.zip`.

Example:
    python bc_train.py --rom smb.nes path/to/recordings
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

from nes_gym.bc import Dataset, balanced_accuracy, class_weights, split_segments
from nes_gym.checkpoints import checkpoint_path, save_atomic
from nes_gym.cli import fraction, non_negative_int, positive_int
from nes_gym.demos import load_demos, mask_name, remap_report
from nes_gym.games import GAMES
from nes_gym.vec_env import make_vec_env

# Gradient steps per logged mean training loss.
LOG_EVERY = 50


@dataclass
class Seeds:
    """Separate random sources derived from one seed, so changing one use can't change the others.

    Attributes:
        split: Shuffles the segments for the train/validation split.
        shuffle: Shuffles the training pairs each epoch.
        init: Seeds PPO's starting weights.
        eval: Seeds torch before the sampled evaluation episodes.
    """

    split: np.random.Generator
    shuffle: np.random.Generator
    init: int
    eval: int


def derive_seeds(seed: int) -> Seeds:
    """Derives the split, shuffle, init and eval random sources from one seed.

    Args:
        seed: The run's seed.

    Returns:
        The Seeds.
    """
    split, shuffle, init, eval_ = np.random.SeedSequence(seed).spawn(4)
    return Seeds(
        split=np.random.default_rng(split),
        shuffle=np.random.default_rng(shuffle),
        init=int(init.generate_state(1)[0]),
        eval=int(eval_.generate_state(1)[0]),
    )


def bc_parameters(policy) -> list[torch.nn.Parameter]:
    """Returns the parameters behaviour cloning trains: everything that chooses actions, nothing of the value head.

    That's the CNN (shared with the value head), the policy MLP and the action head; it leaves out
    `mlp_extractor.value_net` and `value_net`.

    Args:
        policy: An SB3 ActorCriticPolicy with a shared features extractor, e.g. PPO("CnnPolicy", ...).policy.

    Returns:
        The parameters, each once.
    """
    modules = (policy.features_extractor, policy.mlp_extractor.policy_net, policy.action_net)
    return [param for module in modules for param in module.parameters()]


def _logits(policy, dataset: Dataset, indices: np.ndarray) -> torch.Tensor:
    """Returns the policy's action scores for a batch of pairs.

    The stacks go to the device as raw uint8; SB3 converts them to float and divides by 255 itself.
    """
    obs = torch.from_numpy(dataset.stack(indices)).to(policy.device)
    return policy.get_distribution(obs).distribution.logits


@torch.no_grad()
def validate(policy, dataset: Dataset, val_idx: np.ndarray, weights: torch.Tensor, batch_size: int) -> dict:
    """Scores the policy on the validation pairs.

    Args:
        policy: The SB3 policy.
        dataset: The dataset.
        val_idx: The validation pair indices.
        weights: (n_actions,) class weights on the policy's device.
        batch_size: The number of pairs per forward pass.

    Returns:
        A dict with "loss" (class-weighted, as in training: the sum of weight × loss over the sum of weights),
        "unweighted_loss", "accuracy" and "balanced_accuracy".
    """
    policy.set_training_mode(False)
    # Sums stay on the device and are read back once at the end: reading a GPU value makes the CPU wait for the GPU,
    # which would stop the next batch being stacked while the GPU works.
    sums = torch.zeros(3, dtype=torch.float64, device=policy.device)  # weighted loss, total weight, unweighted loss
    preds = []
    for start in range(0, len(val_idx), batch_size):
        batch = val_idx[start : start + batch_size]
        logits = _logits(policy, dataset, batch)
        target = torch.from_numpy(dataset.actions[batch]).to(policy.device)
        sums += torch.stack(
            [
                F.cross_entropy(logits, target, weight=weights, reduction="sum"),
                weights[target].sum(),
                F.cross_entropy(logits, target, reduction="sum"),
            ]
        ).double()
        preds.append(logits.argmax(dim=1))
    weighted_sum, weight_total, unweighted_sum = sums.tolist()
    pred = torch.cat(preds).cpu().numpy()
    target = dataset.actions[val_idx]
    n_actions = len(weights)
    return {
        # NaN if every validation action is missing from training (all weights 0).
        "loss": weighted_sum / weight_total if weight_total else float("nan"),
        "unweighted_loss": unweighted_sum / len(val_idx),
        "accuracy": float(np.mean(pred == target)),
        "balanced_accuracy": balanced_accuracy(pred, target, n_actions),
    }


def train(
    model: PPO,
    dataset: Dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    weights: np.ndarray,
    *,
    shuffle_rng: np.random.Generator,
    epochs: int,
    batch_size: int,
    lr: float,
    checkpoint_dir: str | Path,
    checkpoint_every: int,
    writer: SummaryWriter | None = None,
) -> tuple[int, float, bool]:
    """Trains the model's policy on the demo pairs, leaving it with the best epoch's weights.

    Ctrl-C (KeyboardInterrupt) stops training early but still restores the best finished epoch, so it can be saved.
    If no epoch has finished yet, there's nothing to keep and the KeyboardInterrupt is raised again.

    Args:
        model: The PPO model whose policy is trained.
        dataset: The dataset.
        train_idx: The training pair indices.
        val_idx: The validation pair indices.
        weights: (n_actions,) class weights from the training pairs.
        shuffle_rng: Shuffles the training pairs each epoch.
        epochs: The number of passes over the training pairs.
        batch_size: The number of pairs per gradient step. Each epoch drops the leftover pairs after the last full
            batch, unless there are fewer than one batch's worth.
        lr: Adam's learning rate.
        checkpoint_dir: Where `step_<n>.zip` checkpoints go.
        checkpoint_every: Saves a checkpoint every this many gradient steps.
        writer: TensorBoard writer, or None to log nothing.

    Returns:
        (best_epoch, best_loss, interrupted): the 1-based epoch with the lowest weighted validation loss, that loss,
        and whether Ctrl-C stopped training early.

    Raises:
        KeyboardInterrupt: Ctrl-C before the first epoch finished.
    """
    policy = model.policy
    optimizer = torch.optim.Adam(bc_parameters(policy), lr=lr)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=policy.device)
    best_epoch, best_loss, best_state = 0, float("inf"), None
    step = 0
    interrupted = False
    loss_sum = torch.zeros((), device=policy.device)
    loss_count = 0

    def flush_loss():
        """Logs the mean training loss since the last flush, at the current step."""
        nonlocal loss_sum, loss_count
        if loss_count and writer is not None:
            writer.add_scalar("train/loss", (loss_sum / loss_count).item(), step)
        loss_sum = torch.zeros((), device=policy.device)
        loss_count = 0

    try:
        for epoch in range(1, epochs + 1):
            policy.set_training_mode(True)
            order = shuffle_rng.permutation(train_idx)
            # Drop the last partial batch: a tiny batch still gets a full Adam step. The order is reshuffled every
            # epoch, so different pairs are dropped each time. With fewer pairs than one batch, use them all.
            n_batches = max(1, len(order) // batch_size)
            for start in range(0, n_batches * batch_size, batch_size):
                batch = order[start : start + batch_size]
                logits = _logits(policy, dataset, batch)
                target = torch.from_numpy(dataset.actions[batch]).to(policy.device)
                loss = F.cross_entropy(logits, target, weight=weights_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                # Summed on the device and logged as a mean every LOG_EVERY steps: loss.item() every step would
                # make the CPU wait for the GPU instead of stacking the next batch.
                loss_sum += loss.detach()
                loss_count += 1
                if loss_count == LOG_EVERY:
                    flush_loss()
                if step % checkpoint_every == 0:
                    save_atomic(model, checkpoint_path(checkpoint_dir, step))
            flush_loss()

            metrics = validate(policy, dataset, val_idx, weights_t, batch_size)
            if writer is not None:
                for name, value in metrics.items():
                    writer.add_scalar(f"val/{name}", value, step)
            print(
                f"epoch {epoch}/{epochs} (step {step}): val loss {metrics['loss']:.4f}, "
                f"accuracy {metrics['accuracy']:.1%}, balanced {metrics['balanced_accuracy']:.1%}"
            )
            # The first epoch is always kept. The loss is NaN either every epoch or never (it depends only on the
            # fixed validation actions and weights), so `<` being False for NaN can't block a real improvement.
            if best_state is None or metrics["loss"] < best_loss:
                best_epoch, best_loss = epoch, metrics["loss"]
                best_state = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
    except KeyboardInterrupt:
        if best_state is None:
            raise
        print(f"interrupted at step {step}; keeping epoch {best_epoch}")
        interrupted = True

    policy.load_state_dict(best_state)
    return best_epoch, best_loss, interrupted


def evaluate(model: PPO, deterministic: bool) -> dict:
    """Plays one episode in the model's own env (one env, capped by its TimeLimit).

    Args:
        model: The model to play.
        deterministic: True for argmax actions, False to sample them.

    Returns:
        A dict with "distance" (the furthest info["x"], 0 if the game doesn't report it), "flag" (whether
        info["flag_get"] was ever True) and "length" (steps).
    """
    venv = model.get_env()
    obs = venv.reset()
    distance, flag, length = 0, False, 0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, dones, infos = venv.step(action)
        length += 1
        distance = max(distance, infos[0].get("x", 0))
        flag = flag or bool(infos[0].get("flag_get", False))
        if dones[0]:
            return {"distance": distance, "flag": flag, "length": length}


def _action_counts(actions: np.ndarray, game_actions: tuple[int, ...]) -> str:
    """Formats per-action pair counts, e.g. "NOOP 120, R 300"."""
    counts = np.bincount(actions, minlength=len(game_actions))
    return ", ".join(f"{mask_name(mask)} {count}" for mask, count in zip(game_actions, counts))


def main(argv=None) -> None:
    """Runs behaviour cloning from the command line.

    Args:
        argv: The command-line arguments, or None to use sys.argv.
    """
    parser = argparse.ArgumentParser(description="Clone recorded play into a PPO CNN policy")
    parser.add_argument("demos", nargs="+", help=".nesdemo files or folders containing them")
    parser.add_argument("--rom", required=True, help="the ROM the recordings were made on")
    parser.add_argument("--game", default="smb", choices=sorted(GAMES))
    parser.add_argument("--run", default=time.strftime("bc-%Y%m%d-%H%M%S"), help="names checkpoints/ and runs/")
    parser.add_argument("--out", default="bc_policy.zip", help="where the best weights are saved")
    parser.add_argument("--epochs", type=positive_int, default=30)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=fraction, default=0.1, help="strictly between 0 and 1")
    parser.add_argument(
        "--checkpoint-every", type=positive_int, default=500, help="gradient steps between checkpoints"
    )
    parser.add_argument(
        "--eval-episodes", type=non_negative_int, default=5, help="sampled episodes, after 1 argmax episode"
    )
    parser.add_argument("--eval-max-steps", type=positive_int, default=3000)
    parser.add_argument("--device", default="auto", help='"auto" (CUDA if available), "cuda" or "cpu"')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    # Checked now: otherwise these only fail after every recording has been replayed, or after training (--out).
    if Path(args.out).suffix != ".zip":
        parser.error(f"--out must end in .zip, got {args.out!r}")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        try:
            device = torch.device(args.device)
        except RuntimeError:
            parser.error(f"unknown --device {args.device!r}")
        if device.type == "cuda" and not torch.cuda.is_available():
            parser.error("--device cuda was given, but CUDA isn't available")
    seeds = derive_seeds(args.seed)
    game_actions = GAMES[args.game].actions

    demos = load_demos(args.demos, args.rom, game=args.game, obs_type="pixels")
    print(remap_report(demos, game_actions))
    dataset = Dataset(demos)
    del demos  # Dataset holds its own copy of the frames.
    train_idx, val_idx = split_segments(dataset, rng=seeds.split, val_frac=args.val_frac)
    weights = class_weights(dataset.actions[train_idx], len(game_actions))
    n_segments = len(dataset.segments()[0])
    print(f"{len(dataset)} pairs in {n_segments} segments ({dataset.skipped} empty recordings skipped)")
    print(f"train: {len(train_idx)} pairs: {_action_counts(dataset.actions[train_idx], game_actions)}")
    print(f"val:   {len(val_idx)} pairs: {_action_counts(dataset.actions[val_idx], game_actions)}")

    venv = writer = None
    try:
        venv = make_vec_env(args.rom, args.game, max_episode_steps=args.eval_max_steps)
        model = PPO("CnnPolicy", venv, seed=seeds.init, device=device)
        print(f"training on {model.device}")
        writer = SummaryWriter(f"runs/{args.run}")
        best_epoch, best_loss, interrupted = train(
            model,
            dataset,
            train_idx,
            val_idx,
            weights,
            shuffle_rng=seeds.shuffle,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_dir=f"checkpoints/{args.run}",
            checkpoint_every=args.checkpoint_every,
            writer=writer,
        )
        save_atomic(model, args.out)
        print(f"saved epoch {best_epoch} (val loss {best_loss:.4f}) to {args.out}")
        print(f"watch the saved policy with: python play.py --rom {args.rom} {args.out}")
        if interrupted:
            print("skipping evaluation after Ctrl-C")
            return

        torch.manual_seed(seeds.eval)
        argmax = evaluate(model, deterministic=True)
        for name, value in argmax.items():
            writer.add_scalar(f"eval/argmax/{name}", float(value))
        print(f"argmax:  distance {argmax['distance']}, flag {argmax['flag']}, length {argmax['length']}")

        if args.eval_episodes:
            sampled = [evaluate(model, deterministic=False) for _ in range(args.eval_episodes)]
            distances = [episode["distance"] for episode in sampled]
            flags = sum(episode["flag"] for episode in sampled)
            mean_length = float(np.mean([episode["length"] for episode in sampled]))
            writer.add_scalar("eval/sampled/mean_distance", float(np.mean(distances)))
            writer.add_scalar("eval/sampled/best_distance", float(max(distances)))
            writer.add_scalar("eval/sampled/flag_rate", flags / len(sampled))
            writer.add_scalar("eval/sampled/mean_length", mean_length)
            print(
                f"sampled: mean distance {np.mean(distances):.0f}, best {max(distances)}, "
                f"flag {flags} of {len(sampled)}, mean length {mean_length:.0f}"
            )
    finally:
        if writer is not None:
            writer.close()
        if venv is not None:
            venv.close()


if __name__ == "__main__":
    main()
