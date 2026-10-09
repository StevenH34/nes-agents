"""Tests for the behaviour-cloning dataset, split and class weights.

Demos are built by hand with each frame filled with a known value, so no ROM, SB3 or GPU is needed.
"""

import numpy as np
import pytest

from nes_gym.bc import Dataset, balanced_accuracy, class_weights, split_segments
from nes_gym.demos import Demo


def make_demo(starts, first_value=1, path="demo.nesdemo", actions=None):
    """Builds a pixel Demo whose frames are filled with consecutive values.

    Args:
        starts: Per-pair episode_starts flags.
        first_value: The fill value of the first frame; frame i is filled with (first_value + i) % 256.
        path: The demo's path.
        actions: Per-pair actions, or None for all zeros.

    Returns:
        The Demo.
    """
    n = len(starts)
    values = (np.arange(first_value, first_value + n) % 256).astype(np.uint8)
    observations = np.broadcast_to(values[:, None, None, None], (n, 84, 84, 1)).copy()
    actions = np.zeros(n, dtype=np.int64) if actions is None else np.asarray(actions, dtype=np.int64)
    return Demo(path, observations, actions, np.asarray(starts, dtype=bool))


def stack_values(dataset, indices):
    """Returns each stacked frame's fill value, checking every frame is uniform.

    Args:
        dataset: The Dataset.
        indices: The pair indices to stack.

    Returns:
        (B, n_stack) int array of fill values.
    """
    stacks = dataset.stack(indices)
    assert stacks.dtype == np.uint8
    assert stacks.shape == (len(indices), dataset.n_stack, 84, 84)
    assert (stacks == stacks[..., :1, :1]).all()
    return stacks[..., 0, 0].astype(int)


def segmented_dataset(lengths):
    """Builds a one-demo Dataset with segments of the given lengths.

    Args:
        lengths: The number of pairs in each segment.

    Returns:
        The Dataset.
    """
    starts = []
    for length in lengths:
        starts += [True] + [False] * (length - 1)
    return Dataset([make_demo(starts, first_value=0)])


# Dataset


def test_dataset_joins_demos():
    a = make_demo([True, False, True], first_value=1, actions=[0, 1, 2])
    b = make_demo([True, False], first_value=10, actions=[3, 4])
    dataset = Dataset([a, b])

    assert len(dataset) == 5
    assert dataset.frames.shape == (5, 84, 84)
    assert dataset.frames.dtype == np.uint8
    assert dataset.frames[:, 0, 0].tolist() == [1, 2, 3, 10, 11]
    assert dataset.actions.dtype == np.int64
    assert dataset.actions.tolist() == [0, 1, 2, 3, 4]
    assert dataset.episode_starts.tolist() == [True, False, True, True, False]
    assert dataset.skipped == 0


def test_start_index():
    a = make_demo([True, False, False, True, False])
    b = make_demo([True, False, True])
    dataset = Dataset([a, b])

    assert dataset.start_index.tolist() == [0, 0, 0, 3, 3, 5, 5, 7]


def test_segments():
    dataset = Dataset([make_demo([True, False, False, True]), make_demo([True, False])])

    starts, ends = dataset.segments()
    assert starts.tolist() == [0, 3, 4]
    assert ends.tolist() == [3, 4, 6]


def test_empty_demos_are_skipped():
    empty = make_demo([])
    dataset = Dataset([empty, make_demo([True, False]), empty])

    assert len(dataset) == 2
    assert dataset.skipped == 2


def test_all_empty_demos_raise():
    with pytest.raises(ValueError, match="empty"):
        Dataset([make_demo([]), make_demo([])])


def test_ram_demo_is_rejected():
    ram = Demo("ram.nesdemo", np.zeros((3, 2048), dtype=np.uint8), np.zeros(3, dtype=np.int64), np.ones(3, bool))
    with pytest.raises(ValueError, match="ram.nesdemo.*not pixels"):
        Dataset([make_demo([True]), ram])


def test_float_pixel_demo_is_rejected():
    demo = make_demo([True, False], path="float.nesdemo")
    demo.observations = demo.observations.astype(np.float32)

    with pytest.raises(ValueError, match="float.nesdemo.*not pixels"):
        Dataset([demo])


def test_demo_not_starting_a_segment_is_rejected():
    with pytest.raises(ValueError, match="bad.nesdemo.*doesn't start a segment"):
        Dataset([make_demo([True]), make_demo([False, True], path="bad.nesdemo")])


@pytest.mark.parametrize("field", ["observations", "actions", "episode_starts"])
def test_demo_length_mismatch_is_rejected(field):
    demo = make_demo([True, False, False], path="bad.nesdemo")
    setattr(demo, field, getattr(demo, field)[:-1])

    with pytest.raises(ValueError, match="bad.nesdemo.*differ in length"):
        Dataset([make_demo([True]), demo])


def test_n_stack_must_be_positive():
    with pytest.raises(ValueError, match="n_stack"):
        Dataset([make_demo([True])], n_stack=0)


# Dataset.stack


def test_stack_inside_segment_is_oldest_first():
    dataset = Dataset([make_demo([True] + [False] * 5, first_value=1)])

    assert stack_values(dataset, [4, 5]).tolist() == [[2, 3, 4, 5], [3, 4, 5, 6]]


def test_stack_pads_at_segment_start():
    dataset = Dataset([make_demo([True, False, False, True, False, False], first_value=1)])

    assert stack_values(dataset, [3, 4, 5]).tolist() == [[0, 0, 0, 4], [0, 0, 4, 5], [0, 4, 5, 6]]


def test_stack_pads_at_recording_start():
    a = make_demo([True, False, False, False], first_value=1)
    b = make_demo([True, False], first_value=10)
    dataset = Dataset([a, b])

    assert stack_values(dataset, [0, 4, 5]).tolist() == [[0, 0, 0, 1], [0, 0, 0, 10], [0, 0, 10, 11]]


def test_stack_keeps_index_order_and_repeats():
    dataset = Dataset([make_demo([True] + [False] * 4, first_value=1)])

    assert stack_values(dataset, [4, 0, 4]).tolist() == [[2, 3, 4, 5], [0, 0, 0, 1], [2, 3, 4, 5]]


def test_stack_n_stack():
    dataset = Dataset([make_demo([True, False, False], first_value=1)], n_stack=2)

    assert stack_values(dataset, [0, 2]).tolist() == [[0, 1], [2, 3]]


def test_stack_empty_batch():
    dataset = Dataset([make_demo([True, False])])

    assert dataset.stack(np.array([], dtype=np.int64)).shape == (0, 4, 84, 84)


@pytest.mark.parametrize("index", [-1, 3])
def test_stack_rejects_out_of_range_indices(index):
    dataset = Dataset([make_demo([True, False, False])])

    with pytest.raises(IndexError):
        dataset.stack([0, index])


@pytest.mark.parametrize("indices", [np.array([0.0, 2.9]), np.array([True, False, True])])
def test_stack_rejects_non_integer_indices(indices):
    dataset = Dataset([make_demo([True, False, False])])

    with pytest.raises(IndexError, match="integers"):
        dataset.stack(indices)


# split_segments


def check_split(dataset, train_idx, val_idx):
    """Checks a split covers every pair once and never cuts a segment.

    Args:
        dataset: The split Dataset.
        train_idx: The training pair indices.
        val_idx: The validation pair indices.
    """
    assert train_idx.dtype == np.int64 and val_idx.dtype == np.int64
    assert (np.diff(train_idx) > 0).all() and (np.diff(val_idx) > 0).all()
    assert np.array_equal(np.sort(np.concatenate([train_idx, val_idx])), np.arange(len(dataset)))
    is_val = np.zeros(len(dataset), dtype=bool)
    is_val[val_idx] = True
    for start, end in zip(*dataset.segments()):
        assert is_val[start:end].all() or not is_val[start:end].any()


def test_split_holds_out_whole_segments_near_val_frac():
    lengths = np.random.default_rng(1).integers(5, 40, size=100)
    dataset = segmented_dataset(lengths)

    train_idx, val_idx = split_segments(dataset, rng=np.random.default_rng(0), val_frac=0.1)

    check_split(dataset, train_idx, val_idx)
    target = 0.1 * len(dataset)
    assert target <= len(val_idx) < target + lengths.max()


def test_split_is_reproducible():
    dataset = segmented_dataset([10] * 50)

    first = split_segments(dataset, rng=np.random.default_rng(7))
    second = split_segments(dataset, rng=np.random.default_rng(7))

    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_split_holds_out_min_val_segments_past_val_frac():
    # 10% of 40 equal segments is 4, so the fifth is held out too.
    dataset = segmented_dataset([10] * 40)

    train_idx, val_idx = split_segments(dataset, rng=np.random.default_rng(0), val_frac=0.1)

    check_split(dataset, train_idx, val_idx)
    assert len(val_idx) == 50


def test_split_min_val_segments_can_overshoot_val_frac():
    dataset = segmented_dataset([10] * 20)

    train_idx, val_idx = split_segments(dataset, rng=np.random.default_rng(0), val_frac=0.1)

    check_split(dataset, train_idx, val_idx)
    assert len(val_idx) == 50


@pytest.mark.parametrize("n_segments", [1, 5])
def test_split_raises_with_too_few_segments(n_segments):
    dataset = segmented_dataset([10] * n_segments)

    with pytest.raises(ValueError, match="need more than 5 segments"):
        split_segments(dataset, rng=np.random.default_rng(0), val_frac=0.1)


def test_split_raises_when_nothing_is_left_to_train_on():
    dataset = segmented_dataset([10, 10])

    with pytest.raises(ValueError, match="nothing to train on"):
        split_segments(dataset, rng=np.random.default_rng(0), val_frac=0.9, min_val_segments=1)


@pytest.mark.parametrize("val_frac", [0, 1, -0.1, 1.5])
def test_split_rejects_bad_val_frac(val_frac):
    with pytest.raises(ValueError, match="val_frac"):
        split_segments(segmented_dataset([10] * 10), rng=np.random.default_rng(0), val_frac=val_frac)


def test_split_rejects_bad_min_val_segments():
    with pytest.raises(ValueError, match="min_val_segments"):
        split_segments(segmented_dataset([10] * 10), rng=np.random.default_rng(0), min_val_segments=0)


# class_weights


def test_class_weights_balanced():
    weights = class_weights([0, 1, 2, 0, 1, 2], 3)

    assert weights.dtype == np.float32
    np.testing.assert_allclose(weights, [1, 1, 1])


def test_class_weights_imbalanced():
    np.testing.assert_allclose(class_weights([0] * 6 + [1] * 2, 2), [8 / 12, 8 / 4], rtol=1e-6)


def test_class_weights_cap():
    np.testing.assert_allclose(class_weights([0] * 99 + [1], 2), [100 / 198, 10], rtol=1e-6)
    np.testing.assert_allclose(class_weights([0] * 99 + [1], 2, max_weight=20), [100 / 198, 20], rtol=1e-6)


def test_class_weights_missing_action_is_zero():
    np.testing.assert_allclose(class_weights([0, 0, 2, 2], 3), [1, 0, 1])


@pytest.mark.parametrize(
    "actions, n_actions, max_weight",
    [
        ([], 2, 10.0),
        ([0, 2], 2, 10.0),
        ([-1, 0], 2, 10.0),
        ([0.0, 1.0], 2, 10.0),
        ([0, 1], 2, 0.0),
        ([0, 1], 2, float("nan")),
    ],
)
def test_class_weights_rejects_bad_input(actions, n_actions, max_weight):
    with pytest.raises(ValueError):
        class_weights(actions, n_actions, max_weight=max_weight)


# balanced_accuracy


def test_balanced_accuracy_perfect():
    assert balanced_accuracy([0, 1, 2], [0, 1, 2], 3) == 1.0


def test_balanced_accuracy_ignores_imbalance():
    # Plain accuracy would be 0.75; action 1 is never predicted.
    assert balanced_accuracy([0, 0, 0, 0], [0, 0, 0, 1], 2) == 0.5


def test_balanced_accuracy_skips_actions_absent_from_target():
    assert balanced_accuracy([0, 1, 2, 2], [0, 1, 1, 1], 3) == pytest.approx((1 + 1 / 3) / 2)


@pytest.mark.parametrize("pred, target", [([0, 1], [0]), ([], []), ([0, 3], [0, 1])])
def test_balanced_accuracy_rejects_bad_input(pred, target):
    with pytest.raises(ValueError):
        balanced_accuracy(pred, target, 3)
