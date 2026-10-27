from dataclasses import dataclass, field, replace
from typing import Dict, List
import torch

from worker.canonical import derive_seed
from worker.data import Dataset

State = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class Task:
    ds: Dataset
    K: int = 200
    k: int = 20
    batch_size: int = 64
    lr: float = 0.05
    device: str = "cpu"

    @property
    def m(self) -> int:
        return self.K // self.k

    def on(self, device: str) -> "Task":
        return replace(self, device=device)


@dataclass
class Block:
    """A miner's claim: checkpoints C₀..C_m and the claimed metric. kind is for bookkeeping, hidden from validators."""

    miner_id: bytes
    seed: int
    ckpts: List[State]
    metric_micro: int
    kind: str = "honest"
    cost_steps: int = 0
    meta: Dict = field(default_factory=dict)

    @property
    def w_start(self) -> State:
        return self.ckpts[0]

    @property
    def w_end(self) -> State:
        return self.ckpts[-1]


TASK_ID = b"\xee" * 32  # task identifier: the seed for replication C derives from it (D8)


def miner_id(i: int) -> bytes:
    return bytes([i + 1]) * 32


def block_seed(prev_hash: bytes, miner: bytes, round_no: int) -> int:
    return derive_seed(prev_hash, miner, round_no)


def clone(state: State) -> State:
    return {n: t.detach().clone() for n, t in state.items()}


def add(a: State, b: State, scale: float = 1.0) -> State:
    return {n: a[n] + scale * b[n] for n in a}


def sub(a: State, b: State) -> State:
    return {n: a[n] - b[n] for n in a}
