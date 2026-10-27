"""Експеримент E5: шкода атаки на толерантність (5c) залежно від порогу α.

Атакувальники знають α валідатора (адаптивна 5c) і мають удвічі швидше залізо, тож регулярно
виграють перегони. Для кожного α і механізму (B, R) — точність спільної моделі за раундами
проти мережі без атакувальників.

python -m sim.sweep_e5 --out experiments/runs/e5
"""
import argparse
import csv
import json
import os

from sim.provenance import write_manifest
from sim.simulator import Config, report, simulate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/runs/e5")
    ap.add_argument("--alphas", default="0.01,0.05,0.1,0.3")
    ap.add_argument("--mechs", default="B,R")
    ap.add_argument("--nodes", type=int, default=6)
    ap.add_argument("--attackers", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--mix", type=float, default=0.0)
    ap.add_argument("--max-validators", type=int, default=3)
    ap.add_argument("--seeds", default="0", help="повтори: seed симуляції і task_seed = 42 + seed")
    ap.add_argument("--dataset", default="mnist")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    write_manifest(args.out, vars(args))
    finals, per_round = [], []
    runs = [("none", 0.0, "B", 0)] + [(m, float(a), m, args.attackers)
                                       for m in args.mechs.split(",") for a in args.alphas.split(",")]
    for seed in [int(s) for s in args.seeds.split(",")]:
        for name, alpha, mech, n_bad in runs:
            attack = {"kind": "spoof_tolerance", "alpha": alpha, "mix": args.mix, "speed": 2.0}
            cfg = Config(name=f"{name}_a{alpha}_s{seed}", dataset=args.dataset, rounds=args.rounds,
                         n_honest=args.nodes - n_bad, attackers=[dict(attack) for _ in range(n_bad)],
                         mechanism=mech, mech_params={"B": {"alpha": alpha or 0.05}},
                         max_validators=args.max_validators, seed=seed, task_seed=42 + seed,
                         baseline=False, verify_all=False)
            print(f"== {cfg.name}", flush=True)
            res = simulate(cfg)
            rep = report(cfg, res)
            bad_wins = sum(v for k, v in rep["wins"].items() if k not in ("honest", "none"))
            finals.append({"seed": seed, "mechanism": name, "alpha": alpha,
                           "final_accuracy": rep["final_accuracy"],
                           "attacker_win_share": round(bad_wins / args.rounds, 3),
                           "attacker_rejected_share": rep["rejected_share"].get("spoof_tolerance")})
            for r in res["rounds"]:
                per_round.append({"seed": seed, "mechanism": name, "alpha": alpha, **r})
    for fname, rows in (("final.csv", finals), ("rounds.csv", per_round)):
        with open(os.path.join(args.out, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for x in finals:
        print(json.dumps(x))


if __name__ == "__main__":
    main()
