import random
import torch

from sim import attacks
from sim.core import Task, block_seed, miner_id
from sim.mechanisms import run
from sim.simulator import Config, report, simulate
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


def verdict(kind: str, mech: str, **kw) -> bool:
    victim = attacks.honest(TASK, W0, OTHER, block_seed(bytes(32), OTHER, 0))
    blk = victim if kind == "honest" else attacks.make(kind, TASK, W0, ME, SEED, victim=victim)
    return run(mech, TASK, blk, W0, random.Random(3), **kw).rejected


def test_block_shapes():
    for kind in attacks.ATTACKS:
        victim = attacks.honest(TASK, W0, OTHER, SEED)
        blk = attacks.make(kind, TASK, W0, ME, SEED, victim=victim)
        assert len(blk.ckpts) == TASK.m + 1, kind


def test_b_accepts_honest_rejects_cheap_attacks():
    assert not verdict("honest", "B")
    for kind in ("lazy", "plagiarism", "poison"):
        assert verdict(kind, "B"), kind


def test_b_catches_partial_when_q_covers_all():
    assert verdict("partial", "B", q=TASK.m)


def test_spoof_tolerance_passes_b():
    # Атака 5c задумана так, щоб пройти B: зсув β·α·‖Δ‖ менший за поріг α·‖Δ‖.
    assert not verdict("spoof_tolerance", "B")


def test_c_accepts_honest_rejects_lazy():
    assert not verdict("honest", "C")
    assert verdict("lazy", "C")


def test_c_plagiarism_needs_commit_reveal():
    assert verdict("plagiarism", "C", commit_reveal=True)
    assert not verdict("plagiarism", "C", commit_reveal=False)


def test_d_rejects_lazy():
    assert verdict("lazy", "D")


def test_simulator_lazy_never_wins(monkeypatch):
    monkeypatch.setattr("sim.simulator.load_dataset", lambda name: DS)
    cfg = Config(rounds=2, n_honest=2, attackers=[{"kind": "lazy"}], K=20, k=10, baseline=False)
    res = simulate(cfg)
    rep = report(cfg, res)
    assert rep["rejected_share"]["lazy"] == 1.0
    assert rep["wins"] == {"honest": 2}
