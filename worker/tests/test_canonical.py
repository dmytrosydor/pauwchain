import hashlib
import torch
from worker.canonical import canonical_weight_hash, derive_seed, block_header_bytes


def test_sha256_sanity():
    assert hashlib.sha256(b"abc").hexdigest().startswith("ba7816bf")


def test_order_independent():
    a = {"w": torch.zeros(2, 3), "b": torch.ones(3)}
    b = {"b": torch.ones(3), "w": torch.zeros(2, 3)}
    assert canonical_weight_hash(a) == canonical_weight_hash(b)


def test_value_sensitive():
    a = {"w": torch.zeros(2, 2)}
    b = {"w": torch.zeros(2, 2)}
    b["w"][0, 0] = 1e-7
    assert canonical_weight_hash(a) != canonical_weight_hash(b)


def test_seed_binds_to_miner():
    prev, m1, m2 = bytes(32), b"\x01" * 32, b"\x02" * 32
    assert derive_seed(prev, m1, 7) != derive_seed(prev, m2, 7)
    assert derive_seed(prev, m1, 7) != derive_seed(prev, m2, 8)
    assert derive_seed(prev, m1, 7) == derive_seed(prev, m1, 7)


def test_header_length():
    header = dict(
        version=1,
        height=0,
        prev_hash="00" * 32,
        task_id="11" * 32,
        round=0,
        miner_id="22" * 32,
        timestamp=0,
        w_start="33" * 32,
        w_end="44" * 32,
        ckpt_root="55" * 32,
        K=500,
        k=25,
        metric_micro=912300,
    )
    assert len(block_header_bytes(header)) == 236
