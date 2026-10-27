import hashlib
import struct
from typing import Dict
import torch


def canonical_weight_hash(state: Dict[str, torch.Tensor]) -> bytes:
    hasher = hashlib.sha256()

    for name in sorted(state.keys()):
        t = state[name].detach().cpu().contiguous()
        if t.dtype != torch.float32:
            raise ValueError(f"Expected float32, not {t.dtype}")
        nb = name.encode("ascii")
        hasher.update(struct.pack("<I", len(nb)) + nb)
        hasher.update(bytes([1, t.dim()]))
        for d in t.shape:
            hasher.update(struct.pack("<I", d))
        hasher.update(t.numpy().tobytes())

    return hasher.digest()


def derive_seed(prev_block_hash: bytes, miner_id: bytes, round_no: int) -> int:
    if len(prev_block_hash) != 32 or len(miner_id) != 32:
        raise ValueError("Hashes must be 32 bytes")
    h = hashlib.sha256(
        prev_block_hash + miner_id + struct.pack("<Q", round_no)
    ).digest()
    return struct.unpack("<Q", h[:8])[0]


def block_header_bytes(header: dict) -> bytes:
    parts = [
        struct.pack("<I", header["version"]),
        struct.pack("<Q", header["height"]),
        bytes.fromhex(header["prev_hash"]),
        bytes.fromhex(header["task_id"]),
        struct.pack("<Q", header["round"]),
        bytes.fromhex(header["miner_id"]),
        struct.pack("<Q", header["timestamp"]),
        bytes.fromhex(header["w_start"]),
        bytes.fromhex(header["w_end"]),
        bytes.fromhex(header["ckpt_root"]),
        struct.pack("<I", header["K"]),
        struct.pack("<I", header["k"]),
        struct.pack("<q", header["metric_micro"]),  # signed 64-bit integer: <q
    ]
    raw = b"".join(parts)
    if len(raw) != 236:
        raise ValueError(f"Header must be 236 bytes, got {len(raw)}")
    return raw
