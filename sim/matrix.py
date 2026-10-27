"""Експеримент E2: матриця «атака × механізм».

Для кожної стартової точки (стан моделі після r сумлінних раундів) і кожного повтору
будуємо сумлінний блок і блоки всіх атак, проганяємо кожен механізм і пишемо рядок у CSV.
Для рядка kind=honest частка відхилень — це FRR.

python -m sim.matrix --out experiments/runs/e2 --trials 5 --starts 0,3,8
"""
import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict

import torch

from sim import attacks
from sim.core import Task, block_seed, miner_id
from sim.mechanisms import run
from worker.canonical import canonical_weight_hash
from worker.data import load_dataset
from worker.model import init_state
from worker.train import accuracy, to_micro, train_block

MECHS = ["A", "B", "C", "D", "B+D"]


def warm_states(task: Task, task_seed: int, starts):
    """Стани моделі після r сумлінних раундів для кожного r зі starts."""
    state, out = init_state(task_seed), {}
    prev = bytes(32)
    for r in range(max(starts) + 1):
        if r in starts:
            out[r] = (state, prev)
        seed = block_seed(prev, miner_id(0), r)
        state = train_block(state, task.ds, seed, task.K, task.k, task.batch_size, task.lr, task.device)[-1]
        prev = canonical_weight_hash(state)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/runs/e2")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--starts", default="0,3,8")
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--miner-device", default="cpu")
    ap.add_argument("--validator-device", default="cpu")
    ap.add_argument("--mechs", default=",".join(MECHS))
    ap.add_argument("--kinds", default=",".join(attacks.kinds()))
    ap.add_argument("--params", default="{}", help="JSON: параметри механізмів, напр. {\"B\": {\"q\": 1}}")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    ds = load_dataset(args.dataset)
    miner_task = Task(ds, args.K, args.k, lr=args.lr, device=args.miner_device)
    val_task = miner_task.on(args.validator_device)
    starts = [int(s) for s in args.starts.split(",")]
    mechs = args.mechs.split(",")
    kinds = args.kinds.split(",")
    params = json.loads(args.params)

    rows = []
    t_start = time.perf_counter()
    for r, (w_start, prev) in warm_states(miner_task, 42, starts).items():
        base_metric = to_micro(accuracy(w_start, ds.x_test, ds.y_test))
        for trial in range(args.trials):
            victim_id, me = miner_id(10 + trial), miner_id(100 + trial)
            victim = attacks.honest(miner_task, w_start, victim_id, block_seed(prev, victim_id, r))
            my_seed = block_seed(prev, me, r)
            for kind in kinds:
                blk = victim if kind == "honest" else attacks.make(
                    kind, miner_task, w_start, me, my_seed, victim=victim)
                for mech in mechs:
                    rng = random.Random(f"{r}/{trial}/{kind}/{mech}")
                    kw = params.get(mech, {})
                    if mech == "A":
                        kw = {"prev_metric_micro": base_metric, **kw}
                    if mech == "B+D":
                        kw = {"B": params.get("B", {}), "D": params.get("D", {})}
                    v = run(mech, val_task, blk, w_start, rng, **kw)
                    rows.append({
                        "start_round": r, "trial": trial, "kind": kind, "mechanism": mech,
                        "rejected": int(v.rejected), "verify_steps": v.cost_steps,
                        "verify_s": round(v.seconds, 4), "miner_steps": blk.cost_steps, "K": args.K,
                        "detail": json.dumps(v.detail, default=float)[:500],
                    })
            print(f"start {r} trial {trial} done, {time.perf_counter() - t_start:.0f}s", flush=True)

    path = os.path.join(args.out, "matrix.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    summary(rows, mechs, kinds)
    print(f"\n{len(rows)} rows → {path}")


def summary(rows, mechs, kinds) -> None:
    rej, n, cost = defaultdict(int), defaultdict(int), defaultdict(float)
    for x in rows:
        key = (x["kind"], x["mechanism"])
        rej[key] += x["rejected"]
        n[key] += 1
        cost[x["mechanism"]] += x["verify_steps"] / x["K"]
    print("\nчастка відхилених (для honest — FRR)")
    print(f"{'':16s}" + "".join(f"{m:>8s}" for m in mechs))
    for kind in kinds:
        print(f"{kind:16s}" + "".join(f"{rej[kind, m] / n[kind, m]:8.2f}" for m in mechs))
    per_mech = {m: sum(n[k, m] for k in kinds) for m in mechs}
    print(f"{'overhead V/K':16s}" + "".join(f"{cost[m] / per_mech[m]:8.3f}" for m in mechs))


if __name__ == "__main__":
    main()
