"""Experiment E2: the "attack (at varying strength) × mechanism" matrix.

For every task_seed, every starting point (model state after r honest rounds) and every trial
we build an honest block and the attack blocks from the suite, run every mechanism and write a CSV row.
The honest row gives FRR. For mechanism C blocks are built separately, from the task seed (D8): this
determines what the plagiarist can copy.

python -m sim.matrix --out experiments/runs/e2 --task-seeds 42,43,44 --starts 0,3,8 --trials 3
"""
import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict

from sim import attacks
from sim.core import TASK_ID, Task, block_seed, miner_id
from sim.mechanisms import clear_replica_cache, run
from sim.provenance import write_manifest
from sim.verify_stats import wilson
from worker.canonical import canonical_weight_hash
from worker.data import load_dataset, test_set, val_set
from worker.model import init_state
from worker.train import accuracy, to_micro, train_block

MECHS = ["A", "B", "R", "C", "D", "B+D"]

# Attack suite: (kind, parameters). Each attack's strength varies to get curves rather than 0/1.
SUITES = {
    "main": [
        ("honest", {}),
        ("lazy", {}),
        ("metric_lie", {"inflate": 0.05}),
        ("plagiarism", {}),
        ("partial", {"frac": 0.1}), ("partial", {"frac": 0.5}), ("partial", {"frac": 0.9}),
        ("poison", {"p": 0.1}), ("poison", {"p": 0.3}), ("poison", {"p": 1.0}),
        ("spoof_tiny", {"scale": 0.05}), ("spoof_tiny", {"scale": 0.5}),
        ("spoof_tolerance", {"mix": 0.0}), ("spoof_tolerance", {"mix": 0.5}), ("spoof_tolerance", {"mix": 1.0}),
        ("one_jump", {}),
    ],
    "quick": [("honest", {}), ("lazy", {}), ("plagiarism", {}), ("partial", {"frac": 0.1}),
              ("poison", {"p": 1.0}), ("spoof_tolerance", {}), ("one_jump", {})],
}


def label(kind: str, params: dict) -> str:
    return kind + ("(" + ",".join(f"{k}={v}" for k, v in params.items()) + ")" if params else "")


def warm_states(task: Task, task_seed: int, starts):
    """Model states after r honest rounds for every r in starts."""
    state, out = init_state(task_seed), {}
    prev = bytes(32)
    for r in range(max(starts) + 1):
        if r in starts:
            out[r] = (state, prev)
        seed = block_seed(prev, miner_id(0), r)
        state = train_block(state, task.ds, seed, task.K, task.k, task.batch_size, task.lr, task.device)[-1]
        prev = canonical_weight_hash(state)
    return out


def build_blocks(task, suite, w_start, prev, r, trial, seed_mode, commit_reveal):
    """Blocks for every attack in the suite. seed_mode: "miner" (D2) or "task" (D8, for C)."""
    victim_id, me = miner_id(10 + trial), miner_id(100 + trial)
    if seed_mode == "task":
        victim_seed = my_seed = block_seed(prev, TASK_ID, r)
    else:
        victim_seed, my_seed = block_seed(prev, victim_id, r), block_seed(prev, me, r)
    victim = attacks.honest(task, w_start, victim_id, victim_seed)
    blocks = []
    for kind, params in suite:
        if kind == "honest":
            blk = victim
        elif kind == "plagiarism" and seed_mode == "task" and commit_reveal:
            # Commit-reveal: commitments precede the reveal, nothing to copy; all that is left is to skip the work.
            blk = attacks.lazy(task, w_start, me, my_seed)
            blk.kind = "plagiarism"
        else:
            blk = attacks.make(kind, task, w_start, me, my_seed, victim=victim, **params)
        blocks.append((label(kind, params), kind, blk))
    return victim, blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/runs/e2")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--suite", default="main", choices=sorted(SUITES))
    ap.add_argument("--task-seeds", default="42")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--starts", default="0,3,8")
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--miner-device", default="cpu")
    ap.add_argument("--validator-device", default="cpu")
    ap.add_argument("--replica-devices", default="cpu", help="devices for C replicas, e.g. cpu,mps")
    ap.add_argument("--no-commit-reveal", action="store_true")
    ap.add_argument("--mechs", default=",".join(MECHS))
    ap.add_argument("--params", default="{}", help='JSON keyed by mechanism letter, e.g. {"B": {"q": 1}}')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    write_manifest(args.out, vars(args))
    ds = load_dataset(args.dataset)
    miner_task = Task(ds, args.K, args.k, lr=args.lr, device=args.miner_device)
    val_task = miner_task.on(args.validator_device)
    mechs = args.mechs.split(",")
    params = json.loads(args.params)
    params.setdefault("C", {})["replica_devices"] = args.replica_devices.split(",")
    suite = SUITES[args.suite]
    commit_reveal = not args.no_commit_reveal

    rows = []
    t_start = time.perf_counter()
    for task_seed in [int(s) for s in args.task_seeds.split(",")]:
        starts = [int(s) for s in args.starts.split(",")]
        for r, (w_start, prev) in warm_states(miner_task, task_seed, starts).items():
            mparams = {**params, "A": {"prev_metric_micro": to_micro(accuracy(w_start, *val_set(ds))),
                                       **params.get("A", {})}}
            for trial in range(args.trials):
                sets = {}
                if any(m != "C" for m in mechs):
                    sets["miner"] = build_blocks(miner_task, suite, w_start, prev, r, trial, "miner", commit_reveal)
                if "C" in mechs:
                    sets["task"] = build_blocks(miner_task, suite, w_start, prev, r, trial, "task", commit_reveal)
                for mech in mechs:
                    victim, blocks = sets["task" if mech == "C" else "miner"]
                    honest_acc = accuracy(victim.w_end, *test_set(ds))
                    for lab, kind, blk in blocks:
                        rng = random.Random(f"{task_seed}/{r}/{trial}/{lab}/{mech}")
                        v = run(mech, val_task, blk, w_start, rng, mparams)
                        acc = honest_acc if blk is victim else accuracy(blk.w_end, *test_set(ds))
                        rows.append({
                            "task_seed": task_seed, "start_round": r, "trial": trial, "attack": lab,
                            "kind": kind, "mechanism": mech, "rejected": int(v.rejected),
                            "verify_steps": v.cost_steps, "verify_s": round(v.seconds, 4),
                            "miner_steps": blk.cost_steps, "K": args.K,
                            "test_acc": round(acc, 4), "damage": round(honest_acc - acc, 4),
                            "detail": json.dumps(v.detail, default=float)[:400],
                        })
                print(f"seed {task_seed} start {r} trial {trial}: {time.perf_counter() - t_start:.0f}s", flush=True)
            clear_replica_cache()

    path = os.path.join(args.out, "matrix.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary(rows, mechs, [label(k, p) for k, p in suite], args.out)
    print(f"\n{len(rows)} rows → {path}")


def summary(rows, mechs, labels, out_dir) -> None:
    rej, n, cost, dmg = defaultdict(int), defaultdict(int), defaultdict(float), defaultdict(list)
    for x in rows:
        key = (x["attack"], x["mechanism"])
        rej[key] += x["rejected"]
        n[key] += 1
        cost[x["mechanism"]] += x["verify_steps"] / x["K"]
        if x["mechanism"] != "C":  # in the C set, plagiarism under commit-reveal is a different block
            dmg[x["attack"]].append(x["damage"])
    table = {}
    print("\nrejected share [95% Wilson CI]; honest row = FRR")
    print(f"{'':30s}" + "".join(f"{m:>18s}" for m in mechs) + f"{'damage':>9s}")
    for lab in labels:
        cells = []
        for m in mechs:
            k_, n_ = rej[lab, m], n[lab, m]
            lo, hi = wilson(k_, n_)
            table.setdefault(lab, {})[m] = {"rate": k_ / n_ if n_ else None, "ci": [lo, hi], "n": n_}
            cells.append(f"{k_ / n_:5.2f} [{lo:.2f},{hi:.2f}]" if n_ else "-")
        mean_dmg = sum(dmg[lab]) / len(dmg[lab]) if dmg[lab] else 0.0
        table[lab]["damage_mean"] = mean_dmg
        print(f"{lab:30s}" + "".join(f"{c:>18s}" for c in cells) + f"{mean_dmg:9.3f}")
    per_mech = {m: sum(n[lab, m] for lab in labels) for m in mechs}
    overhead = {m: cost[m] / per_mech[m] for m in mechs}
    print(f"{'verification cost / K':30s}" + "".join(f"{overhead[m]:18.3f}" for m in mechs))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"table": table, "overhead": overhead}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
