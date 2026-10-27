"""Механізми верифікації A, B, C, D, R у вигляді, зручному для симулятора й експериментів.

Кожен механізм повертає Verdict: чи відхилено блок, скільки кроків навчання-еквівалента
витратив валідатор і скільки секунд це зайняло. Валідатор бачить лише блок (точки, seed,
заявлену метрику) і вершину ланцюга; тип атаки йому невідомий.
"""
import itertools
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

# Відкалібровано в E1 (k = 20, CPU ↔ MPS): найбільша відносна похибка чесного вузла 0,0084,
# FRR = 0 при α = 0,05. Для k = 50 запас менший — див. docs/strategy.md, розділ 10.
DEFAULT_ALPHA = 0.05


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
           eps: float = 0.02, prev_metric_micro: Optional[int] = None,
           min_gain: Optional[float] = None) -> Verdict:
    """A (Р9): заявлена метрика збігається з виміряною на випадковій валідаційній підвибірці в межах ε;
    модель не гірша за вершину ланцюга більше ніж на ε; за min_gain — ще й покращилась щонайменше на min_gain.
    """
    t0 = time.perf_counter()
    bad = _structural(task, blk, w_start)
    if bad:
        return Verdict(True, 0, time.perf_counter() - t0, {"reason": bad})
    r = verify_a(blk.w_end, task.ds, blk.metric_micro, subset, eps, rng, task.device)
    rejected = r["verdict"] == "reject"
    if prev_metric_micro is not None:
        prev = prev_metric_micro / 1e6
        if r["measured"] < prev - eps:
            rejected = True
        if min_gain is not None and r["measured"] < prev + min_gain:
            rejected = True
    # Прямий прохід ≈ 1/3 кроку навчання на тому самому обсязі даних.
    cost = subset / task.batch_size / 3
    return Verdict(rejected, cost, time.perf_counter() - t0, r)


def _mean_pairwise_cos(vectors: List[torch.Tensor]) -> Optional[float]:
    vs = [v for v in vectors if v.norm() > 0]
    if len(vs) < 2:
        return None
    cos = [float(a @ b / (a.norm() * b.norm())) for a, b in itertools.combinations(vs, 2)]
    return sum(cos) / len(cos)


def mech_b(task: Task, blk: Block, w_start: State, rng: random.Random, q: int = 2,
           alpha: float = DEFAULT_ALPHA, delta: Optional[float] = None,
           residual_tau: Optional[float] = None) -> Verdict:
    """B (Р5, Р6): перерахунок q приватно обраних переходів.

    Поріг відносний: ‖W_перерах − W_заявл‖ ≤ α·‖W_перерах − W_from‖; якщо задано delta — абсолютний
    (delta = 0 — точний збіг, «B-exact»).
    residual_tau вмикає тест напрямку залишків (механізм R): у чесного вузла залишки r_j = W_заявл − W_перерах
    — це шум обчислень і між переходами майже некорельовані; атакувальник, що зсуває кожну точку в межах
    порогу в одному напрямку, дає співнапрямлені залишки. Відхилити, якщо середній попарний cos > residual_tau.
    """
    rule = {"kind": "absolute", "delta": delta} if delta is not None else {"kind": "relative", "alpha": alpha}
    t0 = time.perf_counter()
    bad = _structural(task, blk, w_start)
    if bad:
        return Verdict(True, 0, time.perf_counter() - t0, {"reason": bad})
    checks, residuals = [], []
    for j in pick_transitions(task.m, q, rng):
        r = verify_b(blk.ckpts[j], blk.ckpts[j + 1], task.ds, blk.seed, j * task.k, task.k,
                     task.batch_size, task.lr, rule, task.device, return_residual=True)
        residuals.append(r.pop("residual"))
        checks.append({"j": j, **r})
    rejected = any(c["verdict"] == "reject" for c in checks)
    detail = {"checks": checks}
    if residual_tau is not None:
        cos = _mean_pairwise_cos(residuals)
        detail["residual_cos"] = cos
        if cos is not None and cos > residual_tau:
            rejected = True
    return Verdict(rejected, len(checks) * task.k, time.perf_counter() - t0, detail)


_replicas: Dict = {}


def _replica_end(task: Task, w_start: State, seed: int):
    """W_end чесної репліки. На одному пристрої результат побітово однаковий, тому кеш точний;
    час обчислення зберігається, щоб вартість у секундах лишалась чесною."""
    key = (canonical_weight_hash(w_start), seed, task.device, task.K, task.k, task.lr, task.batch_size)
    if key not in _replicas:
        t0 = time.perf_counter()
        end = train_block(w_start, task.ds, seed, task.K, task.k, task.batch_size, task.lr,
                          task.device)[-1]
        _replicas[key] = (flat(end), time.perf_counter() - t0)
    return _replicas[key]


def clear_replica_cache() -> None:
    _replicas.clear()


def mech_c(task: Task, blk: Block, w_start: State, replicas: int = 3, alpha: float = DEFAULT_ALPHA,
           replica_devices: Optional[List[str]] = None) -> Verdict:
    """C (Р8): реплікація. n−1 інших вузлів виконують те саме завдання з тим самим seed завдання;
    W_end заявника порівнюється з поелементною медіаною всіх n результатів (разом із заявленим),
    поріг α·‖W_медіана − W_start‖. Репліки можуть рахувати на різних пристроях (replica_devices).

    Блок для C має бути побудований із seed завдання — це забезпечує той, хто генерує блоки
    (sim.matrix / sim.simulator), бо від правила seed залежить, що атакувальник може скопіювати.
    """
    if replicas % 2 == 0:
        raise ValueError("replicas must be odd so the median is a real middle value")
    t0 = time.perf_counter()
    bad = _structural(task, blk, w_start)
    if bad:
        return Verdict(True, 0, time.perf_counter() - t0, {"reason": bad})
    devices = replica_devices or [task.device]
    ends, seconds = [], 0.0
    for i in range(replicas - 1):
        end, s = _replica_end(task.on(devices[i % len(devices)]), w_start, blk.seed)
        ends.append(end)
        seconds += s
    claimed = flat(blk.w_end)
    median = torch.stack([claimed] + ends).median(dim=0).values
    distance = (claimed - median).norm().item()
    threshold = alpha * (median - flat(w_start)).norm().item()
    return Verdict(distance > threshold, (replicas - 1) * task.K, time.perf_counter() - t0 + seconds,
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
    """D (Р10): cos(ΔW_заявл, −∇L) на перших n_batches призначених батчах переходу ≥ τ."""
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


DEFAULT_RESIDUAL_TAU = 0.5


def _one(name: str, task: Task, blk: Block, w_start: State, rng: random.Random, params: Dict) -> Verdict:
    p = params.get(name, {})
    if name == "A":
        return mech_a(task, blk, w_start, rng, **p)
    if name == "B":
        return mech_b(task, blk, w_start, rng, **p)
    if name == "R":
        merged = {"residual_tau": DEFAULT_RESIDUAL_TAU, **params.get("B", {}), **p}
        return mech_b(task, blk, w_start, rng, **merged)
    if name == "C":
        return mech_c(task, blk, w_start, **p)
    if name == "D":
        return mech_d(task, blk, w_start, rng, **p)
    raise ValueError(f"unknown mechanism {name}")


def run(name: str, task: Task, blk: Block, w_start: State, rng: random.Random,
        params: Optional[Dict] = None) -> Verdict:
    """name — один механізм ("B") або комбінація через «+» ("B+D", "A+R"); params — за літерою механізму.
    Комбінацію відхилено, якщо відхилив хоч один складник."""
    params = params or {}
    parts = [_one(n, task, blk, w_start, rng, params) for n in name.split("+")]
    if len(parts) == 1:
        return parts[0]
    return Verdict(any(v.rejected for v in parts), sum(v.cost_steps for v in parts),
                   sum(v.seconds for v in parts), {n: v.detail for n, v in zip(name.split("+"), parts)})
