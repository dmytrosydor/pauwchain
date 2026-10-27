"""Attack arena: anyone writes their own cheating strategy and checks whether it passes mechanisms A, B, R, C, D.

A strategy is a Python file in the strategies folder (arena/strategies/ by default) defining two names:

    NAME = "my_attack"
    def produce(ctx) -> list:      # checkpoints C₀..C_m, C₀ = ctx.w_start
        ...

Through ctx the strategy sees everything a real miner would: the starting weights, its seed and data order,
the task parameters, and also the verification parameters (q, α, τ), i.e. the attacker is adaptive.
Every training step through ctx.honest / ctx.poisoned is counted, which shows how much work the attack saved.
Optional: claim_metric(ctx, w_end) -> int, the claimed accuracy ×10⁶ (otherwise the honestly measured one is claimed).

python -m sim.arena --trials 3 --out experiments/runs/arena
Strategy code runs in this same process: run only strategies you trust.
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
    """What a strategy can see and do."""

    task: Task
    w_start: State
    seed: int
    verifier: Dict  # verification parameters known to the attacker: q, alpha, residual_tau
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
        """Honest steps of transition j (all k by default), counted in steps_used."""
        n = steps or self.k
        out, _ = train_steps(state, self.task.ds, self.seed, j * self.k, n, self.task.batch_size,
                             self.task.lr, self.task.device)
        self.steps_used += n
        return out

    def poisoned(self, state: State, j: int, p: float = 1.0, lr_scale: float = 1.0) -> State:
        """Transition j on labels corrupted with fraction p (y → y+1 mod 10), with learning rate lr·lr_scale."""
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
    except Exception as e:  # noqa: BLE001 — an error in a third-party strategy disqualifies it, not the arena
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
    ap.add_argument("--validators", type=int, default=1, help="V independent validators per block")
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
                    # For C everyone runs the same task with the task seed (D8), otherwise the seed is per miner.
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
                           # What the attacker gains when it passes: saved work and damage to the model.
                           "saved_work": rate * max(0.0, 1 - work), "expected_damage": rate * max(0.0, dmg)}
    return board


def print_board(board, mechs) -> None:
    print("\nshare of blocks that passed verification (work = fraction of K, damage = accuracy drop)")
    print(f"{'strategy':26s}" + "".join(f"{m:>7s}" for m in mechs) + f"{'work':>9s}{'damage':>9s}")
    for s, per in board.items():
        any_m = next(iter(per.values()))
        print(f"{s:26s}" + "".join(f"{per[m]['pass_rate']:7.2f}" for m in mechs)
              + f"{any_m['work']:9.2f}{any_m['damage']:9.3f}")


if __name__ == "__main__":
    main()
