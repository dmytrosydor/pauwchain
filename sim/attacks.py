"""Моделі поведінки майнера: сумлінна робота й атаки 1–5 (docs/research/technical.md, розділ 8)."""
import math
import random
from dataclasses import replace
from typing import List
import torch

from sim.core import Block, State, Task, add, clone, sub
from sim.mechanisms import DEFAULT_ALPHA
from worker.data import Dataset, val_set
from worker.train import accuracy, to_micro, train_block, train_steps


def _metric(task: Task, state: State) -> int:
    return to_micro(accuracy(state, *val_set(task.ds), task.device))


def honest(task: Task, w_start: State, miner: bytes, seed: int) -> Block:
    ckpts = train_block(w_start, task.ds, seed, task.K, task.k, task.batch_size, task.lr, task.device)
    return Block(miner, seed, ckpts, _metric(task, ckpts[-1]), "honest", task.K)


def lazy(task: Task, w_start: State, miner: bytes, seed: int, sigma: float = 1e-4,
         claimed_metric: int = 0) -> Block:
    """Атака 1: не тренує, подає старий стан із мінімальним шумом."""
    g = torch.Generator().manual_seed(seed & 0x7FFFFFFF)
    ckpts = [clone(w_start)]
    for _ in range(task.m):
        prev = ckpts[-1]
        ckpts.append({n: t + sigma * torch.randn(t.shape, generator=g) for n, t in prev.items()})
    return Block(miner, seed, ckpts, claimed_metric or _metric(task, ckpts[-1]), "lazy", 0)


def plagiarism(victim: Block, miner: bytes, seed: int) -> Block:
    """Атака 2: копіює чужі точки й метрику, підставляє свій miner_id."""
    return Block(miner, seed, [clone(c) for c in victim.ckpts], victim.metric_micro, "plagiarism", 0)


def partial(task: Task, w_start: State, miner: bytes, seed: int, frac: float = 0.1) -> Block:
    """Атака 3: чесно рахує перші ⌈frac·m⌉ переходів, решту екстраполює лінійно."""
    h = max(1, math.ceil(frac * task.m))
    _, done = train_steps(w_start, task.ds, seed, 0, h * task.k, task.batch_size, task.lr,
                          task.device, ckpt_every=task.k)
    ckpts = [clone(w_start)] + done
    step = {n: t / h for n, t in sub(ckpts[-1], ckpts[0]).items()}
    while len(ckpts) < task.m + 1:
        ckpts.append(add(ckpts[-1], step))
    return Block(miner, seed, ckpts, _metric(task, ckpts[-1]), "partial", h * task.k,
                 {"honest_transitions": h})


def flip_labels(ds: Dataset, p: float, seed: int = 0) -> Dataset:
    """y′ = (y+1) mod 10 для частки p навчальних міток."""
    n = ds.y_train.shape[0]
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(n, generator=g) < p
    y = ds.y_train.clone()
    y[mask] = (y[mask] + 1) % 10
    return replace(ds, y_train=y)


def poison(task: Task, w_start: State, miner: bytes, seed: int, p: float = 1.0) -> Block:
    """Атака 4: тренує сумлінно, але на зіпсованих мітках (атака на якість моделі)."""
    bad = replace(task, ds=flip_labels(task.ds, p))
    ckpts = train_block(w_start, bad.ds, seed, task.K, task.k, task.batch_size, task.lr, task.device)
    return Block(miner, seed, ckpts, _metric(task, ckpts[-1]), "poison", task.K, {"p": p})


def spoof_tiny(task: Task, w_start: State, miner: bytes, seed: int, scale: float = 0.05) -> Block:
    """Атака 5a: «нескінченно малі оновлення» — один крок з lr·scale на перехід замість k кроків."""
    ckpts = [clone(w_start)]
    for j in range(task.m):
        nxt, _ = train_steps(ckpts[-1], task.ds, seed, j * task.k, 1, task.batch_size,
                             task.lr * scale, task.device)
        ckpts.append(nxt)
    return Block(miner, seed, ckpts, _metric(task, ckpts[-1]), "spoof_tiny", task.m)


def spoof_tolerance(task: Task, w_start: State, miner: bytes, seed: int, alpha: float = DEFAULT_ALPHA,
                    beta: float = 0.9, p: float = 1.0, mix: float = 0.0) -> Block:
    """Атака 5c: експлуатація толерантності B. Адаптивна: атакувальник знає поріг валідатора α.

    На кожному переході чесний результат зсувається на β·α·‖Δ‖ — трохи менше за поріг, тож B не бачить
    різниці. Напрямок зсуву: до отруєної моделі (mix = 0), випадковий (mix = 1) або їх суміш. Випадкова
    складова — спроба обійти тест напрямку залишків R ціною меншої шкоди.
    """
    bad = flip_labels(task.ds, p)
    g = torch.Generator().manual_seed(seed & 0x7FFFFFFF)
    ckpts = [clone(w_start)]
    for j in range(task.m):
        cur = ckpts[-1]
        good, _ = train_steps(cur, task.ds, seed, j * task.k, task.k, task.batch_size, task.lr, task.device)
        evil, _ = train_steps(cur, bad, seed, j * task.k, task.k, task.batch_size, task.lr, task.device)
        target = _unit(sub(evil, good))
        noise = _unit({n: torch.randn(t.shape, generator=g) for n, t in good.items()})
        direction = _unit({n: (1 - mix) * target[n] + mix * noise[n] for n in good})
        shift = beta * alpha * _norm(sub(good, cur))
        ckpts.append(add(good, direction, shift))
    return Block(miner, seed, ckpts, _metric(task, ckpts[-1]), "spoof_tolerance", 2 * task.K,
                 {"alpha": alpha, "beta": beta, "mix": mix})


def one_jump(task: Task, w_start: State, miner: bytes, seed: int, p: float = 1.0,
             lr_boost: float = 10.0, jump_at: int = None) -> Block:
    """Адаптивна атака «один стрибок»: m−1 переходів чесні, а один (j*) замінено на k кроків
    на отруєних мітках зі збільшеним кроком навчання. B ловить її, лише якщо випадково вибере саме j*:
    імовірність пропуску одним валідатором 1 − q/m, V незалежними — (1 − q/m)^V.
    """
    j_star = jump_at if jump_at is not None else random.Random(seed).randrange(task.m)
    bad = flip_labels(task.ds, p)
    ckpts = [clone(w_start)]
    for j in range(task.m):
        ds, lr = (bad, task.lr * lr_boost) if j == j_star else (task.ds, task.lr)
        nxt, _ = train_steps(ckpts[-1], ds, seed, j * task.k, task.k, task.batch_size, lr, task.device)
        ckpts.append(nxt)
    return Block(miner, seed, ckpts, _metric(task, ckpts[-1]), "one_jump", task.K, {"jump_at": j_star})


def metric_lie(task: Task, w_start: State, miner: bytes, seed: int, inflate: float = 0.05,
               sigma: float = 1e-4) -> Block:
    """Ліниве повторення із завищеною заявленою метрикою: «модель покращилась на inflate»."""
    blk = lazy(task, w_start, miner, seed, sigma)
    blk.metric_micro = min(1_000_000, blk.metric_micro + round(inflate * 1_000_000))
    blk.kind = "metric_lie"
    blk.meta["inflate"] = inflate
    return blk


def _norm(state: State) -> float:
    return math.sqrt(sum(float((t.double() ** 2).sum()) for t in state.values()))


def _unit(state: State) -> State:
    n = _norm(state)
    return {k: v / n for k, v in state.items()} if n > 0 else state


ATTACKS = ["lazy", "metric_lie", "plagiarism", "partial", "poison", "spoof_tiny", "spoof_tolerance", "one_jump"]


def make(kind: str, task: Task, w_start: State, miner: bytes, seed: int,
         victim: Block = None, **kw) -> Block:
    if kind == "honest":
        return honest(task, w_start, miner, seed)
    if kind == "plagiarism":
        if victim is None:
            raise ValueError("plagiarism needs a victim block")
        return plagiarism(victim, miner, seed)
    fn = {"lazy": lazy, "metric_lie": metric_lie, "partial": partial, "poison": poison,
          "spoof_tiny": spoof_tiny, "spoof_tolerance": spoof_tolerance, "one_jump": one_jump}[kind]
    return fn(task, w_start, miner, seed, **kw)


def kinds() -> List[str]:
    return ["honest"] + ATTACKS
