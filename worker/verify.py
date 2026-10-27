import random
from typing import Dict, List
import torch

from worker.data import Dataset
from worker.train import accuracy, train_steps

State = Dict[str, torch.Tensor]


def flat(state: State) -> torch.Tensor:
    return torch.cat([state[n].detach().cpu().double().flatten() for n in sorted(state)])


def pick_transitions(m: int, q: int, rng: random.Random) -> List[int]:
    """Валідатор обирає q із m переходів власним приватним генератором (Р5)."""
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
) -> Dict:
    """Механізм B: перерахувати k кроків від ckpt_from і порівняти з заявленою ckpt_to.

    distance = ‖W_перерах − W_заявл‖.
    Поріг (Р6): relative — α·‖W_перерах − W_from‖, absolute — фіксоване δ.
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

    return {
        "verdict": "accept" if distance <= threshold else "reject",
        "distance": distance,
        "threshold": threshold,
        "update_norm": update,
    }


def verify_a(
    state: State,
    ds: Dataset,
    claimed_metric_micro: int,
    subset: int,
    eps: float,
    rng: random.Random,
    device: str = "cpu",
) -> Dict:
    """Механізм A: точність на випадковій підвибірці тесту має бути в межах ε від заявленої (Р9)."""
    n = ds.x_test.shape[0]
    idx = torch.tensor(sorted(rng.sample(range(n), min(subset, n))))
    measured = accuracy(state, ds.x_test[idx], ds.y_test[idx], device)
    claimed = claimed_metric_micro / 1_000_000
    return {
        "verdict": "accept" if abs(measured - claimed) <= eps else "reject",
        "measured": measured,
        "claimed": claimed,
    }
