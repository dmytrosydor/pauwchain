"""Механізми верифікації A, B, C, D у вигляді, зручному для симулятора й експериментів.

Кожен механізм повертає Verdict: чи відхилено блок, скільки кроків навчання-еквівалента
витратив валідатор і скільки секунд це зайняло.
"""
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F

from sim.core import Block, State, Task
from worker.canonical import canonical_weight_hash
from worker.data import batch_indices, permutation, to_input
from worker.model import mnist_cnn
from worker.train import train_block
from worker.verify import flat, pick_transitions, verify_a, verify_b


@dataclass
class Verdict:
    rejected: bool
    cost_steps: float
    seconds: float
    detail: Dict = field(default_factory=dict)


def _structural(task: Task, blk: Block, w_start: State) -> Optional[str]:
    if len(blk.ckpts) != task.m + 1:
        return "wrong checkpoint count"
    if canonical_weight_hash(blk.w_start) != canonical_weight_hash(w_start):
        return "w_start does not match chain tip"  # Р1
    return None


def mech_a(task: Task, blk: Block, w_start: State, rng: random.Random, subset: int = 2000,
           eps: float = 0.02, prev_metric_micro: Optional[int] = None) -> Verdict:
    """A: метрика на випадковій підвибірці тесту; і не гірше за попередній блок більше ніж на ε."""
    t0 = time.perf_counter()
    bad = _structural(task, blk, w_start)
    if bad:
        return Verdict(True, 0, time.perf_counter() - t0, {"reason": bad})
    r = verify_a(blk.w_end, task.ds, blk.metric_micro, subset, eps, rng, task.device)
    rejected = r["verdict"] == "reject"
    if prev_metric_micro is not None and r["measured"] < prev_metric_micro / 1e6 - eps:
        rejected = True
    # Прямий прохід ≈ 1/3 кроку навчання на тому самому обсязі даних.
    cost = subset / task.batch_size / 3
    return Verdict(rejected, cost, time.perf_counter() - t0, r)


def mech_b(task: Task, blk: Block, w_start: State, rng: random.Random, q: int = 2,
           rule: Optional[Dict] = None) -> Verdict:
    """B: перерахунок q приватно обраних переходів (Р5, Р6)."""
    rule = rule or {"kind": "relative", "alpha": 0.3}
    t0 = time.perf_counter()
    bad = _structural(task, blk, w_start)
    if bad:
        return Verdict(True, 0, time.perf_counter() - t0, {"reason": bad})
    checks = []
    for j in pick_transitions(task.m, q, rng):
        r = verify_b(blk.ckpts[j], blk.ckpts[j + 1], task.ds, blk.seed, j * task.k, task.k,
                     task.batch_size, task.lr, rule, task.device)
        checks.append({"j": j, **r})
    rejected = any(c["verdict"] == "reject" for c in checks)
    return Verdict(rejected, len(checks) * task.k, time.perf_counter() - t0, {"checks": checks})


_replicas: Dict = {}


def _replica_end(task: Task, w_start: State, seed: int) -> torch.Tensor:
    # Репліки на одному пристрої дають побітово однаковий результат, тож у симуляції його можна
    # порахувати раз; вартість (n−1)·K усе одно враховується у Verdict.
    key = (canonical_weight_hash(w_start), seed, task.device, task.K, task.k, task.lr, task.batch_size)
    if key not in _replicas:
        end = train_block(w_start, task.ds, seed, task.K, task.k, task.batch_size, task.lr,
                          task.device)[-1]
        _replicas[key] = flat(end)
    return _replicas[key]


def mech_c(task: Task, blk: Block, w_start: State, replicas: int = 3, alpha: float = 0.3,
           commit_reveal: bool = True) -> Verdict:
    """C: реплікація. n−1 інших вузлів виконують те саме завдання з тим самим seed (Р8),
    заявлений W_end порівнюється з медіаною реплік у межах α·‖W_медіана − W_start‖.

    Без commit-reveal плагіатор просто копіює розкриту репліку — і проходить.
    З commit-reveal копіювати нічого, тож він подає без роботи (еквівалент лінивого).
    """
    t0 = time.perf_counter()
    bad = _structural(task, blk, w_start)
    if bad:
        return Verdict(True, 0, time.perf_counter() - t0, {"reason": bad})
    n_other = replicas - 1
    stacked = torch.stack([_replica_end(task, w_start, blk.seed) for _ in range(n_other)])
    median = stacked.median(dim=0).values
    if blk.kind == "plagiarism" and not commit_reveal:
        claimed = median  # скопійовано з розкритої репліки
    elif blk.kind == "plagiarism":
        claimed = flat(w_start)
    else:
        claimed = flat(blk.w_end)
    distance = (claimed - median).norm().item()
    threshold = alpha * (median - flat(w_start)).norm().item()
    return Verdict(distance > threshold, n_other * task.K, time.perf_counter() - t0,
                   {"distance": distance, "threshold": threshold})


def _grad(state: State, task: Task, seed: int, steps: List[int]) -> torch.Tensor:
    model = mnist_cnn()
    model.load_state_dict(state)
    model.to(task.device)
    perm = permutation(seed, task.ds.x_train.shape[0])
    model.zero_grad()
    for s in steps:
        idx = batch_indices(perm, s, task.batch_size)
        x = to_input(task.ds.x_train[idx]).to(task.device)
        y = task.ds.y_train[idx].to(task.device)
        F.cross_entropy(model(x), y).backward()
    named = dict(model.named_parameters())
    return torch.cat([named[n].grad.detach().cpu().double().flatten() for n in sorted(named)])


def _params_flat(state: State) -> torch.Tensor:
    # state_dict mnist_cnn містить лише параметри, тож порядок збігається з named_parameters.
    return flat(state)


def mech_d(task: Task, blk: Block, w_start: State, rng: random.Random, q: int = 2,
           n_batches: int = 4, tau: float = 0.1) -> Verdict:
    """D: градієнтна узгодженість. cos(ΔW_заявл, −∇L) на перших n_batches призначених батчах переходу ≥ τ."""
    t0 = time.perf_counter()
    bad = _structural(task, blk, w_start)
    if bad:
        return Verdict(True, 0, time.perf_counter() - t0, {"reason": bad})
    checks = []
    for j in pick_transitions(task.m, q, rng):
        g = _grad(blk.ckpts[j], task, blk.seed, [j * task.k + i for i in range(n_batches)])
        delta = _params_flat(blk.ckpts[j + 1]) - _params_flat(blk.ckpts[j])
        denom = delta.norm() * g.norm()
        cos = float(-(delta @ g) / denom) if denom > 0 else 0.0
        checks.append({"j": j, "cos": cos})
    rejected = any(c["cos"] < tau for c in checks)
    return Verdict(rejected, len(checks) * n_batches, time.perf_counter() - t0, {"checks": checks})


MECHANISMS = {"A": mech_a, "B": mech_b, "C": mech_c, "D": mech_d}


def run(name: str, task: Task, blk: Block, w_start: State, rng: random.Random, **kw) -> Verdict:
    if name == "B+D":
        b = mech_b(task, blk, w_start, rng, **kw.get("B", {}))
        d = mech_d(task, blk, w_start, rng, **kw.get("D", {}))
        return Verdict(b.rejected or d.rejected, b.cost_steps + d.cost_steps,
                       b.seconds + d.seconds, {"B": b.detail, "D": d.detail})
    if name == "C":
        return mech_c(task, blk, w_start, **kw)
    return MECHANISMS[name](task, blk, w_start, rng, **kw)
