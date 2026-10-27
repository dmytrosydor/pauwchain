import random
from typing import Dict, List
import torch

from worker.data import Dataset, val_set
from worker.train import accuracy, train_steps

State = Dict[str, torch.Tensor]


def flat(state: State) -> torch.Tensor:
    return torch.cat([state[n].detach().cpu().double().flatten() for n in sorted(state)])


def pick_transitions(m: int, q: int, rng: random.Random) -> List[int]:
    """The validator picks q of m transitions with its own private RNG (D5)."""
    return sorted(rng.sample(range(m), min(q, m)))


def verify_b(
    ckpt_from: State,
    ckpt_to: State,
    ds: Dataset,
    seed: int,
    step_from: int,
    k: int,
    batch_size: int,
    lr: float,
    threshold_rule: Dict,
    device: str = "cpu",
    return_residual: bool = False,
) -> Dict:
    """Mechanism B: recompute k steps from ckpt_from and compare with the claimed ckpt_to.

    distance = ‖W_recomputed − W_claimed‖.
    Threshold (D6): relative is α·‖W_recomputed − W_from‖, absolute is a fixed δ.
    """
    recomputed, _ = train_steps(ckpt_from, ds, seed, step_from, k, batch_size, lr, device)
    w_rec, w_to = flat(recomputed), flat(ckpt_to)
    distance = (w_rec - w_to).norm().item()
    update = (w_rec - flat(ckpt_from)).norm().item()

    kind = threshold_rule["kind"]
    if kind == "relative":
        threshold = threshold_rule["alpha"] * update
    elif kind == "absolute":
        threshold = threshold_rule["delta"]
    else:
        raise ValueError(f"unknown threshold kind: {kind}")

    out = {
        "verdict": "accept" if distance <= threshold else "reject",
        "distance": distance,
        "threshold": threshold,
        "update_norm": update,
    }
    if return_residual:
        out["residual"] = w_to - w_rec  # residual r_j for the direction test
    return out


def verify_a(
    state: State,
    ds: Dataset,
    claimed_metric_micro: int,
    subset: int,
    eps: float,
    rng: random.Random,
    device: str = "cpu",
) -> Dict:
    """Mechanism A: accuracy on a random subsample of the validation split must be within ε of the claimed one (D9)."""
    x_val, y_val = val_set(ds)
    n = x_val.shape[0]
    idx = torch.tensor(sorted(rng.sample(range(n), min(subset, n))))
    measured = accuracy(state, x_val[idx], y_val[idx], device)
    claimed = claimed_metric_micro / 1_000_000
    return {
        "verdict": "accept" if abs(measured - claimed) <= eps else "reject",
        "measured": measured,
        "claimed": claimed,
    }
