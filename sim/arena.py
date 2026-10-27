"""Арена атак: будь-хто пише свою стратегію обману і перевіряє, чи пройде вона механізми A, B, R, C, D.

Стратегія — Python-файл у теці стратегій (за замовчуванням arena/strategies/) з двома іменами:

    NAME = "my_attack"
    def produce(ctx) -> list:      # точки C₀..C_m, C₀ = ctx.w_start
        ...

Через ctx стратегія бачить усе, що бачив би справжній майнер: стартові ваги, свій seed і порядок даних,
параметри завдання, а також параметри перевірки (q, α, τ) — тобто атакувальник адаптивний.
Кожен крок навчання через ctx.honest / ctx.poisoned рахується: так видно, скільки роботи атака зекономила.
Необов'язково: claim_metric(ctx, w_end) -> int — заявлена точність ×10⁶ (інакше заявляється чесно виміряна).

python -m sim.arena --trials 3 --out experiments/runs/arena
Код стратегій виконується у цьому ж процесі: запускайте лише стратегії, яким довіряєте.
"""
import argparse
import csv
import importlib.util
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from sim import attacks
from sim.core import TASK_ID, Block, State, Task, block_seed, clone, miner_id
from sim.matrix import warm_states
from sim.mechanisms import DEFAULT_ALPHA, DEFAULT_RESIDUAL_TAU, run
from sim.provenance import write_manifest
from sim.verify_stats import wilson
from worker.data import load_dataset, test_set, val_set
from worker.train import accuracy, to_micro, train_steps

MECHS = ["A", "B", "R", "C", "D"]


@dataclass
class Context:
    """Те, що бачить і може робити стратегія."""

    task: Task
    w_start: State
    seed: int
    verifier: Dict  # параметри перевірки, які атакувальник знає: q, alpha, residual_tau
    rng: random.Random
    steps_used: int = 0
    _bad: Dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return self.task.K

    @property
    def k(self) -> int:
        return self.task.k

    @property
    def m(self) -> int:
        return self.task.m

    def honest(self, state: State, j: int, steps: Optional[int] = None) -> State:
        """Чесні кроки переходу j (за замовчуванням усі k), рахуються в steps_used."""
        n = steps or self.k
        out, _ = train_steps(state, self.task.ds, self.seed, j * self.k, n, self.task.batch_size,
                             self.task.lr, self.task.device)
        self.steps_used += n
        return out

    def poisoned(self, state: State, j: int, p: float = 1.0, lr_scale: float = 1.0) -> State:
        """Перехід j на мітках, зіпсованих із часткою p (y → y+1 mod 10), з кроком lr·lr_scale."""
        if p not in self._bad:
            self._bad[p] = attacks.flip_labels(self.task.ds, p)
        out, _ = train_steps(state, self._bad[p], self.seed, j * self.k, self.k, self.task.batch_size,
                             self.task.lr * lr_scale, self.task.device)
        self.steps_used += self.k
        return out

    def noise_like(self, state: State, scale: float) -> State:
        g = torch.Generator().manual_seed(self.rng.randrange(2**31))
        return {n: scale * torch.randn(t.shape, generator=g) for n, t in state.items()}


@dataclass
class Strategy:
    name: str
    produce: Callable[[Context], List[State]]
    claim_metric: Optional[Callable[[Context, State], int]]
    description: str
    path: str


def load_strategies(folder: str) -> List[Strategy]:
    out = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        path = os.path.join(folder, fname)
        spec = importlib.util.spec_from_file_location(f"arena_strategy_{fname[:-3]}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out.append(Strategy(getattr(mod, "NAME", fname[:-3]), mod.produce, getattr(mod, "claim_metric", None),
                            (mod.__doc__ or "").strip().splitlines()[0] if mod.__doc__ else "", path))
    return out


def check_shape(ckpts: List[State], w_start: State, m: int) -> Optional[str]:
    if not isinstance(ckpts, list) or len(ckpts) != m + 1:
        return f"expected {m + 1} checkpoints"
    for c in ckpts:
        if set(c) != set(w_start) or any(c[n].shape != w_start[n].shape for n in w_start):
            return "checkpoint has wrong tensors"
    return None


def play(strategy: Strategy, task: Task, w_start: State, seed: int, verifier: Dict,
         rng: random.Random) -> Tuple[Optional[Block], Context, Optional[str]]:
    ctx = Context(task, clone(w_start), seed, verifier, rng)
    try:
        ckpts = strategy.produce(ctx)
    except Exception as e:  # noqa: BLE001 — помилка в чужій стратегії дискваліфікує її, а не арену
        return None, ctx, f"{type(e).__name__}: {e}"
    bad = check_shape(ckpts, w_start, task.m)
    if bad:
        return None, ctx, bad
    ckpts = [{n: t.detach().cpu().float().contiguous() for n, t in c.items()} for c in ckpts]
    metric = (strategy.claim_metric(ctx, ckpts[-1]) if strategy.claim_metric
              else to_micro(accuracy(ckpts[-1], *val_set(task.ds))))
    return Block(miner_id(200), seed, ckpts, metric, strategy.name, ctx.steps_used), ctx, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="arena/strategies")
    ap.add_argument("--out", default="experiments/runs/arena")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--mechs", default=",".join(MECHS))
    ap.add_argument("--task-seeds", default="42")
    ap.add_argument("--starts", default="0,3")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--validators", type=int, default=1, help="V незалежних валідаторів на блок")
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    write_manifest(args.out, vars(args))
    ds = load_dataset(args.dataset)
    task = Task(ds)
    strategies = load_strategies(args.strategies)
    mechs = args.mechs.split(",")
    verifier = {"q": args.q, "alpha": args.alpha, "residual_tau": DEFAULT_RESIDUAL_TAU,
                "validators": args.validators}
    params = {"B": {"q": args.q, "alpha": args.alpha}, "C": {"alpha": args.alpha}}

    rows = []
    for task_seed in [int(s) for s in args.task_seeds.split(",")]:
        for r, (w_start, prev) in warm_states(task, task_seed, [int(s) for s in args.starts.split(",")]).items():
            prev_metric = to_micro(accuracy(w_start, *val_set(ds)))
            mparams = {**params, "A": {"prev_metric_micro": prev_metric}}
            for trial in range(args.trials):
                for mech in mechs:
                    # Для C усі виконують те саме завдання з seed завдання (Р8), для решти seed прив'язаний до майнера.
                    seed = block_seed(prev, TASK_ID if mech == "C" else miner_id(200), r * 100 + trial)
                    honest_end = attacks.honest(task, w_start, miner_id(200), seed).w_end
                    honest_acc = accuracy(honest_end, *test_set(ds))
                    for st in strategies:
                        rng = random.Random(f"{task_seed}/{r}/{trial}/{st.name}/{mech}")
                        blk, ctx, err = play(st, task, w_start, seed, verifier, rng)
                        if err:
                            rows.append({"strategy": st.name, "mechanism": mech, "task_seed": task_seed,
                                         "start_round": r, "trial": trial, "passed": 0, "work": 0.0,
                                         "damage": 0.0, "error": err})
                            continue
                        passed = all(not run(mech, task, blk, w_start, random.Random(f"{rng.random()}/{v}"),
                                             mparams).rejected for v in range(args.validators))
                        acc = accuracy(blk.w_end, *test_set(ds))
                        rows.append({"strategy": st.name, "mechanism": mech, "task_seed": task_seed,
                                     "start_round": r, "trial": trial, "passed": int(passed),
                                     "work": round(ctx.steps_used / task.K, 3),
                                     "damage": round(honest_acc - acc, 4), "error": ""})
                print(f"seed {task_seed} start {r} trial {trial} done", flush=True)

    with open(os.path.join(args.out, "arena.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    board = scoreboard(rows, [s.name for s in strategies], mechs)
    with open(os.path.join(args.out, "scoreboard.json"), "w") as f:
        json.dump(board, f, indent=2, ensure_ascii=False)
    print_board(board, mechs)


def scoreboard(rows, names, mechs) -> Dict:
    agg = defaultdict(lambda: {"passed": 0, "n": 0, "work": [], "damage": [], "errors": 0})
    for x in rows:
        a = agg[x["strategy"], x["mechanism"]]
        a["n"] += 1
        a["passed"] += x["passed"]
        a["errors"] += bool(x["error"])
        a["work"].append(x["work"])
        a["damage"].append(x["damage"])
    board = {}
    for s in names:
        board[s] = {}
        for m in mechs:
            a = agg[s, m]
            lo, hi = wilson(a["passed"], a["n"])
            work = sum(a["work"]) / len(a["work"]) if a["work"] else 0.0
            dmg = sum(a["damage"]) / len(a["damage"]) if a["damage"] else 0.0
            rate = a["passed"] / a["n"] if a["n"] else 0.0
            board[s][m] = {"pass_rate": rate, "ci": [lo, hi], "work": work, "damage": dmg, "errors": a["errors"],
                           # Що атакувальник отримує, коли проходить: зекономлену роботу і шкоду моделі.
                           "saved_work": rate * max(0.0, 1 - work), "expected_damage": rate * max(0.0, dmg)}
    return board


def print_board(board, mechs) -> None:
    print("\nчастка блоків, що пройшли перевірку (робота — частка від K, шкода — падіння точності)")
    print(f"{'стратегія':26s}" + "".join(f"{m:>7s}" for m in mechs) + f"{'робота':>9s}{'шкода':>9s}")
    for s, per in board.items():
        any_m = next(iter(per.values()))
        print(f"{s:26s}" + "".join(f"{per[m]['pass_rate']:7.2f}" for m in mechs)
              + f"{any_m['work']:9.2f}{any_m['damage']:9.3f}")


if __name__ == "__main__":
    main()
