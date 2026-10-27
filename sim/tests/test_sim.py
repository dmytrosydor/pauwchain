import random
import torch

from sim import attacks
from sim.core import TASK_ID, Task, block_seed, miner_id
from sim.mechanisms import clear_replica_cache, run
from sim.simulator import Config, report, simulate
from sim.verify_stats import wilson
from worker.data import Dataset
from worker.model import init_state


def synthetic() -> Dataset:
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 256, (1200, 28, 28), dtype=torch.uint8, generator=g)
    w = torch.randn(28 * 28, 10, generator=g)
    y = (x.flatten(1).float() @ w).argmax(1)
    return Dataset(x[:1000], y[:1000], x[1000:], y[1000:])


DS = synthetic()
TASK = Task(DS, K=40, k=10, batch_size=32, lr=0.05)
W0 = init_state(1)
ME, OTHER = miner_id(0), miner_id(1)
SEED = block_seed(bytes(32), ME, 0)


def verdict(kind: str, mech: str, params=None, seed_mode="miner", **attack_kw) -> bool:
    if seed_mode == "task":
        my_seed = other_seed = block_seed(bytes(32), TASK_ID, 0)
    else:
        my_seed, other_seed = SEED, block_seed(bytes(32), OTHER, 0)
    victim = attacks.honest(TASK, W0, OTHER, other_seed)
    blk = victim if kind == "honest" else attacks.make(kind, TASK, W0, ME, my_seed, victim=victim, **attack_kw)
    return run(mech, TASK, blk, W0, random.Random(3), params or {}).rejected


def test_block_shapes():
    victim = attacks.honest(TASK, W0, OTHER, SEED)
    for kind in attacks.ATTACKS:
        blk = attacks.make(kind, TASK, W0, ME, SEED, victim=victim)
        assert len(blk.ckpts) == TASK.m + 1, kind


def test_b_accepts_honest_rejects_cheap_attacks():
    assert not verdict("honest", "B")
    for kind in ("lazy", "plagiarism", "poison"):
        assert verdict(kind, "B"), kind


def test_b_exact_accepts_same_device():
    assert not verdict("honest", "B", {"B": {"delta": 0.0}})


def test_b_catches_partial_and_one_jump_when_q_covers_all():
    full = {"B": {"q": TASK.m}}
    assert verdict("partial", "B", full)
    assert verdict("one_jump", "B", full)


def test_one_jump_passes_b_when_jump_not_sampled():
    blk = attacks.one_jump(TASK, W0, ME, SEED, jump_at=0)
    rng = random.Random(0)
    # Check only transitions 1..m−1: a jump in transition 0 is not visible.
    from sim.mechanisms import mech_b
    import sim.mechanisms as mech
    original = mech.pick_transitions
    mech.pick_transitions = lambda m, q, r: [1, 2]
    try:
        assert not mech_b(TASK, blk, W0, rng).rejected
    finally:
        mech.pick_transitions = original


def test_spoof_tolerance_passes_b_but_not_residual_test():
    assert not verdict("spoof_tolerance", "B", mix=0.0)
    assert verdict("spoof_tolerance", "R", {"B": {"q": 3}}, mix=0.0)


def test_c_accepts_honest_rejects_lazy():
    clear_replica_cache()
    assert not verdict("honest", "C", seed_mode="task")
    assert verdict("lazy", "C", seed_mode="task")


def test_c_plagiarism_without_commit_reveal_passes():
    # Without commit-reveal a copy of another replica equals the honest result, so C cannot tell them apart.
    clear_replica_cache()
    assert not verdict("plagiarism", "C", seed_mode="task")


def test_a_rejects_metric_lie():
    assert verdict("metric_lie", "A")


def test_d_rejects_lazy():
    assert verdict("lazy", "D")


def test_combination_rejects_if_any_part_rejects():
    assert verdict("lazy", "A+B")
    assert not verdict("honest", "A+B")


def test_wilson_bounds():
    lo, hi = wilson(0, 15)
    assert lo == 0.0 and 0.15 < hi < 0.25
    lo, hi = wilson(15, 15)
    assert hi == 1.0 and lo > 0.75


def test_simulator_lazy_never_wins(monkeypatch):
    monkeypatch.setattr("sim.simulator.load_dataset", lambda name: DS)
    cfg = Config(rounds=2, n_honest=2, attackers=[{"kind": "lazy"}], K=20, k=10, baseline=False)
    res = simulate(cfg)
    rep = report(cfg, res)
    assert rep["rejected_share"]["lazy"] == 1.0
    assert rep["wins"] == {"honest": 2}


def test_simulator_verifies_late_candidates(monkeypatch):
    # plagiarism arrives after the honest block; with verify_all it is still verified for the statistics.
    monkeypatch.setattr("sim.simulator.load_dataset", lambda name: DS)
    cfg = Config(rounds=1, n_honest=2, attackers=[{"kind": "plagiarism"}], K=20, k=10, baseline=False)
    rep = report(cfg, simulate(cfg))
    assert rep["submissions"].get("plagiarism") == 1
    assert rep["rejected_share"]["plagiarism"] == 1.0
