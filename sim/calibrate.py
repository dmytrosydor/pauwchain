"""Експеримент E1: калібрування порогів B і D, FRR між різним залізом, ROC.

Для кожного стану моделі (після r сумлінних раундів), кожного k і кожного пристрою валідатора
перераховуємо ВСІ переходи сумлінного блоку та «фальшиві» переходи атак і пишемо:
- ratio = ‖W_перерах − W_заявл‖ / ‖W_перерах − W_from‖ (механізм B, відносна відстань);
- cos(ΔW_заявл, −∇L) на перших n_batches батчах переходу (механізм D).

python -m sim.calibrate --out experiments/runs/e1 --devices cpu,mps --ks 10,20,50 --starts 0,1,2,4,8
"""
import argparse
import csv
import json
import os

from sim import attacks
from sim.core import Block, Task, block_seed, miner_id
from sim.matrix import warm_states
from sim.mechanisms import _grad, _params_flat
from sim.verify_stats import roc
from worker.data import load_dataset
from worker.train import train_steps
from worker.verify import flat

# Які переходи атаки фальшиві: для partial — лише ті, що після чесного префікса.
ATTACKS = ["lazy", "partial", "spoof_tiny", "spoof_tolerance", "plagiarism"]


def fake_transitions(blk: Block, m: int):
    if blk.kind == "partial":
        return range(blk.meta["honest_transitions"], m)
    return range(m)


def measure(task: Task, blk: Block, j: int, n_batches: int):
    rec, _ = train_steps(blk.ckpts[j], task.ds, blk.seed, j * task.k, task.k, task.batch_size,
                         task.lr, task.device)
    w_rec, w_to, w_from = flat(rec), flat(blk.ckpts[j + 1]), flat(blk.ckpts[j])
    update = (w_rec - w_from).norm().item()
    ratio = (w_rec - w_to).norm().item() / update if update > 0 else float("inf")
    g = _grad(blk.ckpts[j], task, blk.seed, [j * task.k + i for i in range(n_batches)])
    delta = _params_flat(blk.ckpts[j + 1]) - _params_flat(blk.ckpts[j])
    denom = (delta.norm() * g.norm()).item()
    cos = float(-(delta @ g) / denom) if denom > 0 else 0.0
    return ratio, cos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/runs/e1")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--devices", default="cpu,mps")
    ap.add_argument("--ks", default="10,20,50")
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--starts", default="0,1,2,4,8")
    ap.add_argument("--miners", type=int, default=2)
    ap.add_argument("--n-batches", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset(args.dataset)
    devices = args.devices.split(",")
    starts = [int(s) for s in args.starts.split(",")]
    rows = []
    for k in [int(x) for x in args.ks.split(",")]:
        mtask = Task(ds, args.K, k, lr=args.lr, device="cpu")
        for r, (w_start, prev) in warm_states(mtask, 42, starts).items():
            for mi in range(args.miners):
                me, other = miner_id(120 + mi), miner_id(140 + mi)
                seed = block_seed(prev, me, r)
                victim = attacks.honest(mtask, w_start, other, block_seed(prev, other, r))
                blocks = [attacks.honest(mtask, w_start, me, seed)]
                blocks += [attacks.make(a, mtask, w_start, me, seed, victim=victim) for a in ATTACKS]
                for dev in devices:
                    vtask = mtask.on(dev)
                    for blk in blocks:
                        js = range(mtask.m) if blk.kind == "honest" else fake_transitions(blk, mtask.m)
                        for j in js:
                            ratio, cos = measure(vtask, blk, j, args.n_batches)
                            rows.append({"k": k, "start_round": r, "miner": mi, "device": dev,
                                         "kind": blk.kind, "j": j, "ratio": ratio, "cos": cos})
            print(f"k={k} start={r} done ({len(rows)} rows)", flush=True)

    with open(os.path.join(args.out, "transitions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    summary = roc(rows)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    for key, s in summary.items():
        print(key, json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
