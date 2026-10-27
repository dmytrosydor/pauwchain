import random
import torch

from worker.canonical import canonical_weight_hash, derive_seed
from worker.data import Dataset
from worker.model import init_state
from worker.store import load_state, save_state
from worker.train import hash_states, train_block
from worker.verify import pick_transitions, verify_a, verify_b

B, LR, K, k = 32, 0.05, 40, 10
RULE = {"kind": "relative", "alpha": 0.3}


def synthetic() -> Dataset:
    # Синтетичні 28×28 з мітками за фіксованою проєкцією: вчиться, але без завантаження MNIST.
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 256, (1200, 28, 28), dtype=torch.uint8, generator=g)
    w = torch.randn(28 * 28, 10, generator=g)
    y = (x.flatten(1).float() @ w).argmax(1)
    return Dataset(x[:1000], y[:1000], x[1000:], y[1000:])


DS = synthetic()
SEED = derive_seed(bytes(32), b"\x01" * 32, 0)


def honest():
    return train_block(init_state(1), DS, SEED, K, k, B, LR)


def check(ckpts, j, seed=SEED):
    return verify_b(ckpts[j], ckpts[j + 1], DS, seed, j * k, k, B, LR, RULE)


def test_genesis_deterministic():
    assert canonical_weight_hash(init_state(1)) == canonical_weight_hash(init_state(1))
    assert canonical_weight_hash(init_state(1)) != canonical_weight_hash(init_state(2))


def test_block_shape():
    ckpts = honest()
    assert len(ckpts) == K // k + 1
    assert canonical_weight_hash(ckpts[0]) == canonical_weight_hash(init_state(1))


def test_recompute_bitwise_equal():
    assert hash_states(honest()) == hash_states(honest())


def test_honest_accepted():
    ckpts = honest()
    for j in range(K // k):
        r = check(ckpts, j)
        assert r["verdict"] == "accept" and r["distance"] == 0.0


def test_lazy_rejected():
    ckpts = honest()
    g = torch.Generator().manual_seed(5)
    lazy_to = {n: t + 1e-4 * torch.randn(t.shape, generator=g) for n, t in ckpts[1].items()}
    assert verify_b(ckpts[1], lazy_to, DS, SEED, k, k, B, LR, RULE)["verdict"] == "reject"
    copy_to = {n: t.clone() for n, t in ckpts[1].items()}
    assert verify_b(ckpts[1], copy_to, DS, SEED, k, k, B, LR, RULE)["verdict"] == "reject"


def test_plagiarism_rejected():
    # Чужі точки під своїм miner_id: seed інший, тож перерахунок розходиться.
    ckpts = honest()
    thief_seed = derive_seed(bytes(32), b"\x02" * 32, 0)
    assert check(ckpts, 0, seed=thief_seed)["verdict"] == "reject"


def test_pick_transitions_private():
    picks = pick_transitions(20, 2, random.Random(7))
    assert len(picks) == 2 and len(set(picks)) == 2 and all(0 <= j < 20 for j in picks)


def test_verify_a():
    ckpts = honest()
    from worker.train import accuracy, to_micro
    true = to_micro(accuracy(ckpts[-1], DS.x_test, DS.y_test))
    ok = verify_a(ckpts[-1], DS, true, 200, 0.0, random.Random(1))
    assert ok["verdict"] == "accept"
    lie = verify_a(ckpts[-1], DS, min(true + 300_000, 1_000_000), 200, 0.05, random.Random(1))
    assert lie["verdict"] == "reject"


def test_store_roundtrip(tmp_path):
    state = init_state(3)
    h = save_state(str(tmp_path), state)
    assert h == canonical_weight_hash(state).hex()
    assert canonical_weight_hash(load_state(str(tmp_path), h)).hex() == h
