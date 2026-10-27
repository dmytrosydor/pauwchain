"""Cross-language contract: Python must produce the same bytes and hashes as recorded in docs/vectors.json and
docs/golden.json. C++ checks the same files (node/tests/test_main.cpp)."""
import hashlib
import json
import os

import numpy as np
import torch

from worker.canonical import block_header_bytes, canonical_weight_hash, derive_seed
from worker.train import ckpt_root

DOCS = os.path.join(os.path.dirname(__file__), "..", "..", "docs")


def load(name):
    with open(os.path.join(DOCS, name)) as f:
        return json.load(f)


def test_golden_header_hash():
    g = load("golden.json")
    assert hashlib.sha256(block_header_bytes(g["header"])).hexdigest() == g["block_hash"]


def test_header_vectors():
    for v in load("vectors.json")["headers"]:
        raw = block_header_bytes(v["header"])
        assert raw.hex() == v["bytes_hex"]
        assert hashlib.sha256(raw).hexdigest() == v["block_hash"]


def test_ckpt_root_vectors():
    for v in load("vectors.json")["ckpt_roots"]:
        assert ckpt_root([bytes.fromhex(h) for h in v["hashes"]]).hex() == v["root"]


def test_seed_vectors():
    for v in load("vectors.json")["seeds"]:
        seed = derive_seed(bytes.fromhex(v["prev_block_hash"]), bytes.fromhex(v["miner_id"]), v["round"])
        assert seed == v["seed"]


def test_weight_hash_vectors():
    for v in load("vectors.json")["weights"]:
        state = {}
        for name, t in v["tensors"].items():
            arr = np.frombuffer(bytes.fromhex(t["le_hex"]), dtype="<f4").reshape(t["shape"])
            state[name] = torch.from_numpy(arr.copy())
        assert canonical_weight_hash(state).hex() == v["hash"]
