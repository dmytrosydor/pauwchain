import os
from typing import Dict
import torch
from safetensors.torch import load_file, save_file

from worker.canonical import canonical_weight_hash

State = Dict[str, torch.Tensor]


def save_state(store_dir: str, state: State) -> str:
    """Writes weights to store/<canonical hash>.safetensors and returns the hash in hex."""
    state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    h = canonical_weight_hash(state).hex()
    path = os.path.join(store_dir, f"{h}.safetensors")
    if not os.path.exists(path):
        os.makedirs(store_dir, exist_ok=True)
        save_file(state, path)
    return h


def load_state(store_dir: str, h: str) -> State:
    state = load_file(os.path.join(store_dir, f"{h}.safetensors"))
    if canonical_weight_hash(state).hex() != h:
        raise ValueError(f"store object {h} does not match its hash")
    return state
