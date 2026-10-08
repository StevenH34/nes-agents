"""Tests for bc_train.py's parameter selection, seeds, training loop and evaluation.

Needs stable-baselines3 (the `train` extra); skipped without it. Training uses a fake 3-action env and hand-built
demos, so no recordings are needed; evaluation uses the null game on the committed nestest ROM. Everything runs on
the CPU.
"""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

pytest.importorskip("stable_baselines3")

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

import bc_train  # noqa: E402
from nes_gym.bc import Dataset, class_weights  # noqa: E402
from nes_gym.checkpoints import checkpoint_path, save_atomic  # noqa: E402
from nes_gym.demos import Demo  # noqa: E402
from nes_gym.vec_env import make_vec_env, wrap_vec_env  # noqa: E402

N_ACTIONS = 3


class ThreeActionEnv(gym.Env):
    """A blank-screen env with 3 actions and 5-step episodes, so the policy has a real choice to learn."""

    observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
    action_space = spaces.Discrete(N_ACTIONS)

    def reset(self, *, seed=None, options=None):
        """Starts a blank episode."""
        super().reset(seed=seed)
        self.t = 0
        return np.zeros((84, 84, 1), dtype=np.uint8), {}

    def step(self, action):
        """Ends the episode after 5 steps."""
        self.t += 1
        return np.zeros((84, 84, 1), dtype=np.uint8), 0.0, self.t >= 5, False, {}


def make_model(**policy_kwargs):
    """Builds a CPU PPO CnnPolicy model on the fake env.

    Args:
        policy_kwargs: Passed to PPO as policy_kwargs.

    Returns:
        The model.
    """
    venv = wrap_vec_env(DummyVecEnv([ThreeActionEnv]))
    return PPO("CnnPolicy", venv, device="cpu", seed=0, policy_kwargs=policy_kwargs or None)


def make_dataset(n_segments=6, length=8):
    """Builds a Dataset of random frames and actions in equal-length segments.

    Args:
        n_segments: The number of segments.
        length: The pairs per segment.

    Returns:
        The Dataset.
    """
    rng = np.random.default_rng(0)
    n = n_segments * length
    frames = rng.integers(0, 256, size=(n, 84, 84, 1), dtype=np.uint8)
    actions = rng.integers(0, N_ACTIONS, size=n)
    starts = np.zeros(n, dtype=bool)
    starts[::length] = True
    return Dataset([Demo("fake.nesdemo", frames, actions, starts)])


def run_train(model, dataset, tmp_path):
    """Trains for 2 epochs on pairs 0–31, validating on 32–47: 4 steps per epoch, checkpoints every 3 steps.

    Args:
        model: The model to train.
        dataset: A make_dataset() Dataset.
        tmp_path: Checkpoints go to tmp_path / "ckpt".

    Returns:
        train()'s (best_epoch, best_loss, interrupted).
    """
    train_idx, val_idx = np.arange(32), np.arange(32, 48)
    return bc_train.train(
        model,
        dataset,
        train_idx,
        val_idx,
        class_weights(dataset.actions[train_idx], N_ACTIONS),
        shuffle_rng=np.random.default_rng(0),
        epochs=2,
        batch_size=8,
        lr=1e-3,
        checkpoint_dir=tmp_path / "ckpt",
        checkpoint_every=3,
    )


def value_keys(state):
    """Returns the state-dict keys of the value head."""
    return [key for key in state if key.startswith(("value_net.", "mlp_extractor.value_net."))]


@pytest.mark.parametrize("net_arch", [None, {"pi": [16], "vf": [16]}])
def test_bc_parameters_is_policy_minus_value_parts(net_arch):
    """Selects every parameter except the value MLP and value head, each once.

    Args:
        net_arch: The policy's net_arch; the default CnnPolicy has empty MLPs, so one case gives them layers.
    """
    policy = make_model(**({"net_arch": net_arch} if net_arch else {})).policy
    selected = bc_train.bc_parameters(policy)
    value_ids = {id(p) for p in policy.value_net.parameters()}
    value_ids |= {id(p) for p in policy.mlp_extractor.value_net.parameters()}

    assert len(selected) == len({id(p) for p in selected})
    assert {id(p) for p in selected} == {id(p) for p in policy.parameters()} - value_ids
    assert {id(p) for p in policy.features_extractor.parameters()} <= {id(p) for p in selected}
    assert {id(p) for p in policy.action_net.parameters()} <= {id(p) for p in selected}


def test_derive_seeds_is_reproducible_and_separate():
    """The same seed gives the same sources; the sources differ from each other."""
    a, b = bc_train.derive_seeds(0), bc_train.derive_seeds(0)
    assert (a.init, a.eval) == (b.init, b.eval)
    assert np.array_equal(a.split.integers(0, 1 << 30, 8), b.split.integers(0, 1 << 30, 8))
    assert np.array_equal(a.shuffle.integers(0, 1 << 30, 8), b.shuffle.integers(0, 1 << 30, 8))

    c = bc_train.derive_seeds(0)
    assert c.init != c.eval
    assert not np.array_equal(c.split.integers(0, 1 << 30, 8), c.shuffle.integers(0, 1 << 30, 8))
    assert bc_train.derive_seeds(1).init != c.init


def test_train_updates_action_parts_and_checkpoints(tmp_path):
    """Changes the action head but not the value head, checkpoints by step, and the result saves and loads."""
    model = make_model()
    dataset = make_dataset()
    train_idx, val_idx = np.arange(32), np.arange(32, 48)
    before = {key: value.clone() for key, value in model.policy.state_dict().items()}

    best_epoch, best_loss, interrupted = run_train(model, dataset, tmp_path)

    assert best_epoch in (1, 2)
    assert not interrupted
    assert np.isfinite(best_loss)
    after = model.policy.state_dict()
    keys = value_keys(after)
    assert keys
    for key in keys:
        assert torch.equal(after[key], before[key]), key
    assert not torch.equal(after["action_net.weight"], before["action_net.weight"])
    # 32 pairs / 8 per batch × 2 epochs = 8 steps, so checkpoints at steps 3 and 6.
    assert sorted(p.name for p in (tmp_path / "ckpt").iterdir()) == ["step_3.zip", "step_6.zip"]

    metrics = bc_train.validate(model.policy, dataset, val_idx, torch.ones(N_ACTIONS), batch_size=5)
    assert set(metrics) == {"loss", "unweighted_loss", "accuracy", "balanced_accuracy"}
    # With all weights 1, the weighted loss is the plain mean loss.
    assert metrics["loss"] == pytest.approx(metrics["unweighted_loss"], rel=1e-5)
    assert 0 <= metrics["accuracy"] <= 1 and 0 <= metrics["balanced_accuracy"] <= 1

    out = tmp_path / "bc_policy.zip"
    save_atomic(model, out)
    loaded = PPO.load(out, device="cpu")
    assert torch.equal(loaded.policy.state_dict()["action_net.weight"], after["action_net.weight"])
    assert checkpoint_path(tmp_path / "ckpt", 3).exists()


@pytest.mark.parametrize("n_train, steps", [(33, 4), (5, 1)])
def test_train_drops_last_partial_batch(tmp_path, n_train, steps):
    """Each epoch does only full batches, or one batch of everything when there's less than a batch.

    Args:
        n_train: The number of training pairs (batch size 8).
        steps: The expected gradient steps in one epoch, counted by checkpointing every step.
    """
    dataset = make_dataset()
    train_idx, val_idx = np.arange(n_train), np.arange(40, 48)
    bc_train.train(
        make_model(),
        dataset,
        train_idx,
        val_idx,
        class_weights(dataset.actions[train_idx], N_ACTIONS),
        shuffle_rng=np.random.default_rng(0),
        epochs=1,
        batch_size=8,
        lr=1e-3,
        checkpoint_dir=tmp_path,
        checkpoint_every=1,
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"step_{n}.zip" for n in range(1, steps + 1))


class FakeWriter:
    """Records add_scalar() calls as (tag, value, step)."""

    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step=None):
        """Records one scalar."""
        self.scalars.append((tag, value, step))


def test_train_logs_mean_loss_every_log_every_steps_and_at_epoch_end(tmp_path, monkeypatch):
    """With LOG_EVERY=3 and 4 steps per epoch, the training loss is logged at steps 3, 4, 7 and 8."""
    monkeypatch.setattr(bc_train, "LOG_EVERY", 3)
    dataset = make_dataset()
    train_idx, val_idx = np.arange(32), np.arange(32, 48)
    writer = FakeWriter()
    bc_train.train(
        make_model(),
        dataset,
        train_idx,
        val_idx,
        class_weights(dataset.actions[train_idx], N_ACTIONS),
        shuffle_rng=np.random.default_rng(0),
        epochs=2,
        batch_size=8,
        lr=1e-3,
        checkpoint_dir=tmp_path,
        checkpoint_every=100,
        writer=writer,
    )

    losses = [(value, step) for tag, value, step in writer.scalars if tag == "train/loss"]
    assert [step for _, step in losses] == [3, 4, 7, 8]
    assert all(isinstance(value, float) and np.isfinite(value) for value, _ in losses)
    val_steps = sorted({step for tag, _, step in writer.scalars if tag.startswith("val/")})
    assert val_steps == [4, 8]


def interrupt_on_call(monkeypatch, n):
    """Makes the n-th (1-based) call to bc_train._logits raise KeyboardInterrupt, as Ctrl-C would.

    Each make_dataset() epoch in run_train() calls _logits 6 times: 4 training batches, then 2 validation batches.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        n: The call that raises.
    """
    real = bc_train._logits
    calls = 0

    def logits(*args):
        nonlocal calls
        calls += 1
        if calls == n:
            raise KeyboardInterrupt
        return real(*args)

    monkeypatch.setattr(bc_train, "_logits", logits)


def test_train_ctrl_c_keeps_best_finished_epoch(tmp_path, monkeypatch):
    """Ctrl-C in epoch 2 returns epoch 1's weights with interrupted=True."""
    real_validate = bc_train.validate
    snapshots = []

    def validate(policy, *args):
        metrics = real_validate(policy, *args)
        snapshots.append(policy.action_net.weight.detach().clone())
        return metrics

    monkeypatch.setattr(bc_train, "validate", validate)
    interrupt_on_call(monkeypatch, 7)  # The first training batch of epoch 2.
    model = make_model()

    best_epoch, _, interrupted = run_train(model, make_dataset(), tmp_path)

    assert (best_epoch, interrupted) == (1, True)
    assert len(snapshots) == 1
    assert torch.equal(model.policy.action_net.weight, snapshots[0])


def test_train_ctrl_c_in_first_epoch_raises(tmp_path, monkeypatch):
    """Ctrl-C before any epoch finishes has nothing to keep, so it's raised again."""
    interrupt_on_call(monkeypatch, 2)

    with pytest.raises(KeyboardInterrupt):
        run_train(make_model(), make_dataset(), tmp_path)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--out", "bc_policy"], "--out must end in .zip"),
        (["--val-frac", "0"], "--val-frac"),
        (["--val-frac", "1.5"], "--val-frac"),
        (["--device", "gpu"], "unknown --device"),
        (["--epochs", "0"], "--epochs: must be at least 1"),
        (["--epochs", "two"], "--epochs: invalid int value"),
        (["--eval-episodes", "-1"], "--eval-episodes: must be at least 0"),
    ],
)
def test_main_rejects_bad_arguments_before_loading(tmp_path, capsys, extra, message):
    """Bad arguments give a usage error before any recording is loaded.

    The demo path doesn't exist, so reaching load_demos() would raise something other than SystemExit.

    Args:
        extra: The bad arguments.
        message: Text the usage error must contain.
    """
    argv = [str(tmp_path / "missing"), "--rom", str(tmp_path / "missing.nes"), *extra]
    with pytest.raises(SystemExit) as exc:
        bc_train.main(argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_main_rejects_cuda_without_cuda(tmp_path, capsys, monkeypatch):
    """--device cuda on a machine without CUDA gives a usage error before loading."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    argv = [str(tmp_path / "missing"), "--rom", str(tmp_path / "missing.nes"), "--device", "cuda"]
    with pytest.raises(SystemExit) as exc:
        bc_train.main(argv)
    assert exc.value.code == 2
    assert "CUDA isn't available" in capsys.readouterr().err


@pytest.mark.parametrize("deterministic", [True, False])
def test_evaluate_plays_one_capped_episode(nestest_rom, deterministic):
    """Plays until TimeLimit ends the episode; the null game reports no distance or flag.

    Args:
        deterministic: Argmax or sampled actions.
    """
    venv = make_vec_env(nestest_rom, "null", max_episode_steps=5)
    try:
        model = PPO("CnnPolicy", venv, device="cpu", seed=0)
        assert bc_train.evaluate(model, deterministic) == {"distance": 0, "flag": False, "length": 5}
    finally:
        venv.close()
