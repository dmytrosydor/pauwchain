"""Experiment E4: verification parameters versus detection, false rejections and cost.

Axes (one at a time, the rest at defaults):
  B.q      — transitions recomputed per validator (k = 20, m = 10)
  B.k      — checkpoint interval at fixed K = 200 (validator on another device, so FRR is visible)
  A.subset — validation sample size for the metric check
  C.n      — number of replicas
  D.tau    — gradient-consistency threshold

python -m sim.sweep_e4 --out experiments/runs/e4
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict

from sim.core import Task
from sim.matrix import build_blocks, warm_states
from sim.mechanisms import run
from sim.provenance import write_manifest
from sim.verify_stats import wilson
from worker.data import load_dataset, val_set
from worker.train import accuracy, to_micro

SUITE = [("honest", {}), ("lazy", {}), ("metric_lie", {"inflate": 0.02}), ("partial", {"frac": 0.5}),
         ("partial", {"frac": 0.9}), ("one_jump", {}), ("spoof_tolerance", {"mix": 0.0})]

AXES = (
    [{"axis": "B.q", "value": q, "mech": "B", "params": {"B": {"q": q}}, "k": 20} for q in (1, 2, 3, 5)]
    + [{"axis": "B.k", "value": k, "mech": "B", "params": {"B": {"q": 2}}, "k": k, "validator": "mps"}
       for k in (10, 20, 50)]
    + [{"axis": "A.subset", "value": s, "mech": "A", "params": {"A": {"subset": s}}, "k": 20}
       for s in (200, 1000, 5000)]
    + [{"axis": "C.n", "value": n, "mech": "C", "params": {"C": {"replicas": n}}, "k": 20} for n in (3, 5)]
    + [{"axis": "D.tau", "value": t, "mech": "D", "params": {"D": {"tau": t}}, "k": 20} for t in (0.0, 0.1, 0.2)]
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/runs/e4")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--task-seeds", default="42,43")
    ap.add_argument("--starts", default="0,3")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--K", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    write_manifest(args.out, vars(args))
    ds = load_dataset(args.dataset)
    starts = [int(s) for s in args.starts.split(",")]
    cache = {}  # (k, seed_mode, task_seed, r, trial) → blocks, so axes share the same blocks
    rows = []
    for cfg in AXES:
        task = Task(ds, args.K, cfg["k"])
        vtask = task.on(cfg.get("validator", "cpu"))
        seed_mode = "task" if cfg["mech"] == "C" else "miner"
        for task_seed in [int(s) for s in args.task_seeds.split(",")]:
            for r, (w_start, prev) in warm_states(task, task_seed, starts).items():
                params = {**cfg["params"]}
                if cfg["mech"] == "A":
                    params["A"] = {"prev_metric_micro": to_micro(accuracy(w_start, *val_set(ds))), **params["A"]}
                for trial in range(args.trials):
                    key = (cfg["k"], seed_mode, task_seed, r, trial)
                    if key not in cache:
                        cache[key] = build_blocks(task, SUITE, w_start, prev, r, trial, seed_mode, True)[1]
                    for lab, kind, blk in cache[key]:
                        rng = random.Random(f"{cfg['axis']}/{cfg['value']}/{task_seed}/{r}/{trial}/{lab}")
                        v = run(cfg["mech"], vtask, blk, w_start, rng, params)
                        rows.append({"axis": cfg["axis"], "value": cfg["value"], "attack": lab, "kind": kind,
                                     "rejected": int(v.rejected), "verify_steps": v.cost_steps, "K": args.K})
        print(f"{cfg['axis']}={cfg['value']} done", flush=True)

    with open(os.path.join(args.out, "e4.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(rows, args.out)


def summarize(rows, out_dir) -> None:
    rej, n, cost = defaultdict(int), defaultdict(int), defaultdict(list)
    for x in rows:
        key = (x["axis"], x["value"], x["attack"])
        rej[key] += x["rejected"]
        n[key] += 1
        cost[x["axis"], x["value"]].append(x["verify_steps"] / x["K"])
    table = defaultdict(dict)
    for (axis, value, attack), hits in rej.items():
        lo, hi = wilson(hits, n[axis, value, attack])
        table[f"{axis}={value}"][attack] = {"rate": hits / n[axis, value, attack], "ci": [lo, hi]}
    for (axis, value), cs in cost.items():
        table[f"{axis}={value}"]["cost_K"] = sum(cs) / len(cs)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(table, f, indent=2)
    attacks = list(dict.fromkeys(x["attack"] for x in rows))
    print("\nrejected share per setting (honest = FRR); cost as a fraction of K")
    print(f"{'setting':16s}" + "".join(f"{a[:14]:>15s}" for a in attacks) + f"{'cost':>8s}")
    for setting, per in table.items():
        cells = "".join(f"{per[a]['rate']:15.2f}" if a in per else f"{'-':>15s}" for a in attacks)
        print(f"{setting:16s}{cells}{per['cost_K']:8.3f}")


if __name__ == "__main__":
    main()
