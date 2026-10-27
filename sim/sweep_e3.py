"""Experiment E3: share of malicious nodes.

10 nodes, 0–50% attackers, several "mechanism × attack" pairs. We measure the shared model's
accuracy over rounds and the share of rounds won by attackers (= share of rewards).

python -m sim.sweep_e3 --out experiments/runs/e3
"""
import argparse
import csv
import json
import os

from sim.provenance import write_manifest
from sim.simulator import Config, centralized, report, simulate
from worker.data import load_dataset

SCENARIOS = [
    {"name": "A_lazy", "mechanism": "A", "attack": {"kind": "lazy"}},
    {"name": "B_lazy", "mechanism": "B", "attack": {"kind": "lazy"}},
    {"name": "A_poison30", "mechanism": "A", "attack": {"kind": "poison", "p": 0.3}},
    {"name": "B_poison30", "mechanism": "B", "attack": {"kind": "poison", "p": 0.3}},
    {"name": "B_one_jump", "mechanism": "B", "attack": {"kind": "one_jump"}},
    # 5c costs twice the honest work; speed = 2 is an attacker with twice-as-fast hardware.
    {"name": "B_spoof_tolerance_x2", "mechanism": "B", "attack": {"kind": "spoof_tolerance", "speed": 2.0}},
    {"name": "R_spoof_tolerance_x2", "mechanism": "R", "attack": {"kind": "spoof_tolerance", "speed": 2.0}},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/runs/e3")
    ap.add_argument("--nodes", type=int, default=10)
    ap.add_argument("--fractions", default="0,0.1,0.2,0.3,0.5")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--scenarios", default=",".join(s["name"] for s in SCENARIOS))
    ap.add_argument("--max-validators", type=int, default=3)
    ap.add_argument("--seeds", default="0", help="repeats: simulation seed and task_seed = 42 + seed")
    ap.add_argument("--dataset", default="mnist")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    write_manifest(args.out, vars(args))
    chosen = [s for s in SCENARIOS if s["name"] in args.scenarios.split(",")]
    per_round, finals = [], []
    for seed in [int(x) for x in args.seeds.split(",")]:
        for sc in chosen:
            for frac in [float(f) for f in args.fractions.split(",")]:
                n_bad = round(frac * args.nodes)
                cfg = Config(name=f"{sc['name']}_{int(frac * 100)}_s{seed}", dataset=args.dataset,
                             rounds=args.rounds, n_honest=args.nodes - n_bad,
                             attackers=[dict(sc["attack"]) for _ in range(n_bad)], mechanism=sc["mechanism"],
                             max_validators=args.max_validators, seed=seed, task_seed=42 + seed,
                             baseline=False, verify_all=False)
                print(f"== {cfg.name}", flush=True)
                res = simulate(cfg)
                rep = report(cfg, res)
                bad_wins = sum(v for k, v in rep["wins"].items() if k not in ("honest", "none"))
                accs = [r["accuracy"] for r in res["rounds"]]
                finals.append({"seed": seed, "scenario": sc["name"], "mechanism": sc["mechanism"], "fraction": frac,
                               "final_accuracy": rep["final_accuracy"], "min_accuracy": min(accs),
                               "attacker_win_share": round(bad_wins / args.rounds, 3),
                               "honest_rejected_share": rep["rejected_share"].get("honest", 0.0),
                               "verify_overhead": rep["verify_overhead"],
                               "useful_work_share": rep["useful_work_share"]})
                for r in res["rounds"]:
                    per_round.append({"seed": seed, "scenario": sc["name"], "fraction": frac, **r})

    for name, rows in (("final.csv", finals), ("rounds.csv", per_round)):
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    base = centralized(Config(rounds=args.rounds), load_dataset(args.dataset))
    with open(os.path.join(args.out, "baseline.json"), "w") as f:
        json.dump({"baseline": base, "config": vars(args)}, f, indent=2)
    for x in finals:
        print(json.dumps(x, ensure_ascii=False))


if __name__ == "__main__":
    main()
