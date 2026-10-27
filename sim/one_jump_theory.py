"""Атака «один стрибок»: теорія проти виміру.

Для кожного блоку перераховуємо ВСІ переходи один раз і знаходимо ті, що не проходять B. Далі
тисячі разів моделюємо вибір q переходів V валідаторами справжнім pick_transitions і порівнюємо
частку виявлення з формулою 1 − (1 − q/m)^V.

python -m sim.one_jump_theory --out experiments/runs/one_jump --blocks 20
"""
import argparse
import csv
import json
import os
import random

from sim import attacks
from sim.core import Task, block_seed, miner_id
from sim.mechanisms import DEFAULT_ALPHA
from sim.provenance import write_manifest
from worker.data import load_dataset
from worker.model import init_state
from worker.verify import pick_transitions, verify_b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/runs/one_jump")
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    write_manifest(args.out, vars(args))
    task = Task(load_dataset("mnist"))
    rule = {"kind": "relative", "alpha": DEFAULT_ALPHA}
    bad_sets = []
    for b in range(args.blocks):
        w0 = init_state(1000 + b)
        seed = block_seed(bytes(32), miner_id(b), 0)
        blk = attacks.one_jump(task, w0, miner_id(b), seed)
        bad = {j for j in range(task.m)
               if verify_b(blk.ckpts[j], blk.ckpts[j + 1], task.ds, seed, j * task.k, task.k,
                           task.batch_size, task.lr, rule)["verdict"] == "reject"}
        bad_sets.append((blk.meta["jump_at"], bad))
        print(f"block {b}: jump_at={blk.meta['jump_at']} failing={sorted(bad)}", flush=True)

    rows, rng = [], random.Random(7)
    for q in (1, 2, 3, 5):
        for V in (1, 3, 5):
            hits = 0
            for i in range(args.draws):
                _, bad = bad_sets[i % len(bad_sets)]
                hits += any(set(pick_transitions(task.m, q, rng)) & bad for _ in range(V))
            rows.append({"q": q, "V": V, "measured": round(hits / args.draws, 4),
                         "theory": round(1 - (1 - q / task.m) ** V, 4)})
    with open(os.path.join(args.out, "theory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"failing_sets": [sorted(b) for _, b in bad_sets]}))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
