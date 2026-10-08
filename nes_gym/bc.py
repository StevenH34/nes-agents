"""Behaviour-cloning data: frame stacks, a train/validation split and class weights, in numpy only.

load_demos() holds every pair in memory as one (84, 84) uint8 frame. Dataset joins them and builds each batch's
4-frame stacks on demand from indices, so memory stays at one frame per pair instead of four. Stacks restart at
each segment's first pair with zero padding, as SB3's VecFrameStack does after a reset, so the policy sees the same
observations in training as in the environment.

Pairs inside a segment are near-copies of their neighbours, so the validation set holds out whole segments: a
random split of pairs would put almost the same frames on both sides and make validation look better than it is.
"""

from collections.abc import Sequence

import nes_py
import numpy as np

from nes_gym.demos import Demo

FRAME_SHAPE = (nes_py.OBS_SIZE, nes_py.OBS_SIZE)


class Dataset:
    """Pixel demos joined into one array, with frame stacks built per batch.

    Args:
        demos: The demos from load_demos(..., obs_type="pixels"). Empty ones are skipped.
        n_stack: The number of frames in each stack, matching VecFrameStack.

    Raises:
        ValueError: `n_stack` is less than 1, every demo is empty, or a demo isn't pixels, has observations,
            actions and episode_starts of different lengths, or doesn't start a segment on its first pair.

    Attributes:
        frames: (N, 84, 84) uint8, every pair's frame.
        actions: (N,) int64 action indices.
        episode_starts: (N,) bool, True on the first pair of each segment.
        start_index: (N,) int64, the index of the first pair in each pair's segment.
        n_stack: The number of frames in each stack.
        skipped: The number of empty demos skipped.
    """

    def __init__(self, demos: Sequence[Demo], n_stack: int = 4):
        if n_stack < 1:
            raise ValueError(f"n_stack must be at least 1, got {n_stack}")
        kept = [demo for demo in demos if len(demo)]
        if not kept:
            raise ValueError("every demo is empty")
        for demo in kept:
            # uint8 too: SB3 scales the raw frames by 1/255, so float frames would be scaled wrongly.
            if demo.observations.shape[1:] != (*FRAME_SHAPE, 1) or demo.observations.dtype != np.uint8:
                raise ValueError(
                    f"{demo.path}: observations are {demo.observations.dtype} {demo.observations.shape[1:]}, "
                    f"not pixels (uint8 {(*FRAME_SHAPE, 1)})"
                )
            # load_demo() guarantees these, but a Demo can be built by hand. A length mismatch would pair every
            # later frame with the wrong action.
            if not len(demo.observations) == len(demo.actions) == len(demo.episode_starts):
                raise ValueError(
                    f"{demo.path}: {len(demo.observations)} observations, {len(demo.actions)} actions and "
                    f"{len(demo.episode_starts)} episode_starts differ in length"
                )
            # Checked so a stack can never reach back into the previous demo.
            if not demo.episode_starts[0]:
                raise ValueError(f"{demo.path}: the first pair doesn't start a segment")

        self.frames = np.concatenate([demo.observations[..., 0] for demo in kept])
        self.actions = np.concatenate([demo.actions for demo in kept]).astype(np.int64)
        self.episode_starts = np.concatenate([demo.episode_starts for demo in kept]).astype(bool)
        positions = np.arange(len(self.actions))
        self.start_index = np.maximum.accumulate(np.where(self.episode_starts, positions, 0))
        self.n_stack = n_stack
        self.skipped = len(demos) - len(kept)

    def __len__(self) -> int:
        """Returns the number of pairs."""
        return len(self.actions)

    def segments(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns each segment's first pair index and one past its last, as two int64 arrays."""
        starts = np.flatnonzero(self.episode_starts)
        ends = np.append(starts[1:], len(self))
        return starts, ends

    def stack(self, indices) -> np.ndarray:
        """Builds the frame stacks for a batch of pairs.

        Args:
            indices: (B,) pair indices.

        Returns:
            (B, n_stack, 84, 84) uint8, oldest frame first. Frames from before a pair's segment start are zero.

        Raises:
            IndexError: The indices aren't integers (casting would truncate floats and turn a boolean mask into 0s
                and 1s), or an index is negative or not less than len(self). Negative indices aren't wrapped.
        """
        idx = np.asarray(indices)
        if not np.issubdtype(idx.dtype, np.integer):
            raise IndexError(f"pair indices must be integers, got {idx.dtype}")
        idx = idx.astype(np.intp).reshape(-1)
        if idx.size and (idx.min() < 0 or idx.max() >= len(self)):
            raise IndexError(f"pair indices must be in [0, {len(self)})")
        offsets = idx[:, None] - np.arange(self.n_stack - 1, -1, -1)
        valid = offsets >= self.start_index[idx][:, None]
        # Invalid offsets can be negative; read the pair's own frame there instead, then zero it.
        out = self.frames[np.where(valid, offsets, idx[:, None])]
        out[~valid] = 0
        return out


def split_segments(
    dataset: Dataset,
    *,
    rng: np.random.Generator,
    val_frac: float = 0.1,
    min_val_segments: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Splits the pairs into training and validation sets without cutting any segment.

    The segments are shuffled, then held out in that order until they reach both `val_frac` of all pairs and
    `min_val_segments` segments. The held-out share can therefore overshoot `val_frac`: by up to one segment, or
    by more when `min_val_segments` long segments are needed.

    Args:
        dataset: The dataset to split.
        rng: The random source for the shuffle.
        val_frac: The share of pairs to hold out, strictly between 0 and 1.
        min_val_segments: The fewest segments the validation set may have.

    Returns:
        (train_idx, val_idx): sorted int64 pair indices.

    Raises:
        ValueError: `val_frac` isn't strictly between 0 and 1, `min_val_segments` is less than 1, or holding out
            enough segments would leave no training pairs.
    """
    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac must be strictly between 0 and 1, got {val_frac}")
    if min_val_segments < 1:
        raise ValueError(f"min_val_segments must be at least 1, got {min_val_segments}")
    starts, ends = dataset.segments()
    if len(starts) <= min_val_segments:
        raise ValueError(
            f"need more than {min_val_segments} segments to hold out {min_val_segments} for validation and still "
            f"train, got {len(starts)}; record more clips or lower min_val_segments"
        )
    target = val_frac * len(dataset)
    held = []
    total = 0
    for segment in rng.permutation(len(starts)):
        if total >= target and len(held) >= min_val_segments:
            break
        held.append(segment)
        total += ends[segment] - starts[segment]
    if len(held) == len(starts):
        raise ValueError(f"all {len(starts)} segments would be held out, leaving nothing to train on")

    is_val = np.zeros(len(dataset), dtype=bool)
    for segment in held:
        is_val[starts[segment] : ends[segment]] = True
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def _check_actions(actions, n_actions: int, name: str) -> np.ndarray:
    """Returns `actions` as a 1-D int64 array, raising ValueError if it's empty or out of range."""
    actions = np.asarray(actions)
    if actions.ndim != 1 or actions.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {actions.shape}")
    if not np.issubdtype(actions.dtype, np.integer):
        raise ValueError(f"{name} must be integers, got {actions.dtype}")
    if actions.min() < 0 or actions.max() >= n_actions:
        raise ValueError(f"{name} must be in [0, {n_actions})")
    return actions.astype(np.int64)


def class_weights(actions, n_actions: int, max_weight: float = 10.0) -> np.ndarray:
    """Weights each action by its inverse frequency, so rare actions count as much as common ones in the loss.

    Args:
        actions: (N,) action indices, e.g. the training pairs' actions.
        n_actions: The number of actions.
        max_weight: The cap, so an action seen a handful of times can't dominate the loss.

    Returns:
        (n_actions,) float32. Balanced data gives 1.0 for every action; an action that never appears gets 0.

    Raises:
        ValueError: `actions` is empty, not integers or out of range, or `max_weight` isn't positive.
    """
    if not max_weight > 0:  # Also rejects NaN.
        raise ValueError(f"max_weight must be positive, got {max_weight}")
    actions = _check_actions(actions, n_actions, "actions")
    counts = np.bincount(actions, minlength=n_actions)
    present = counts > 0
    weights = np.zeros(n_actions)
    weights[present] = len(actions) / (present.sum() * counts[present])
    return np.minimum(weights, max_weight).astype(np.float32)


def balanced_accuracy(pred, target, n_actions: int) -> float:
    """The mean, over the actions present in `target`, of the share of each action's pairs predicted correctly.

    Unlike plain accuracy, always predicting the most common action scores only 1 / (number of actions present).

    Args:
        pred: (N,) predicted action indices.
        target: (N,) true action indices.
        n_actions: The number of actions.

    Returns:
        A float in [0, 1].

    Raises:
        ValueError: `pred` and `target` differ in shape, or either is empty, not integers or out of range.
    """
    pred = _check_actions(pred, n_actions, "pred")
    target = _check_actions(target, n_actions, "target")
    if pred.shape != target.shape:
        raise ValueError(f"pred and target differ in shape: {pred.shape} and {target.shape}")
    counts = np.bincount(target, minlength=n_actions)
    correct = np.bincount(target[pred == target], minlength=n_actions)
    present = counts > 0
    return float(np.mean(correct[present] / counts[present]))
