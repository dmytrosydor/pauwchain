"""Експеримент E3: частка зловмисних вузлів.

10 вузлів, частка атакувальників 0–50 %, кілька пар «механізм × атака». Міряємо точність
спільної моделі за раундами і частку раундів, які виграли атакувальники (= частка нагород).

python -m sim.sweep_e3 --out experiments/runs/e3
"""
import argparse
import csv
import json
import os

from sim.simulator import Config, centralized, report, simulate
from worker.data import load_dataset

SCENARIOS = [
    {"name": "A_lazy", "mechanism": "A", "attack": {"kind": "lazy"}},
    {"name": "B_lazy", "mechanism": "B", "attack": {"kind": "lazy"}},
    {"name": "A_poison30", "mechanism": "A", "attack": {"kind": "poison", "p": 0.3}},
    {"name": "B_poison30", "mechanism": "B", "attack": {"kind": "poison", "p": 0.3}},
    # 5c удвічі дорожча за чесну роботу; speed = 2 — атакувальник із удвічі швидшим залізом.
    {"name": "B_spoof_tolerance_x2", "mechanism": "B", "attack": {"kind": "spoof_tolerance", "speed": 2.0}},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/runs/e3")
    ap.add_argument("--nodes", type=int, default=10)
    ap.add_argument("--fractions", default="0,0.1,0.2,0.3,0.5")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--scenarios", default=",".join(s["name"] for s in SCENARIOS))
    ap.add_argument("--max-validators", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    chosen = [s for s in SCENARIOS if s["name"] in args.scenarios.split(",")]
    per_round, finals = [], []
    for sc in chosen:
        for frac in [float(f) for f in args.fractions.split(",")]:
            n_bad = round(frac * args.nodes)
            cfg = Config(name=f"{sc['name']}_{int(frac * 100)}", rounds=args.rounds,
                         n_honest=args.nodes - n_bad, attackers=[dict(sc["attack"]) for _ in range(n_bad)],
                         mechanism=sc["mechanism"], max_validators=args.max_validators, baseline=False)
            print(f"== {cfg.name}", flush=True)
            res = simulate(cfg)
            rep = report(cfg, res)
            bad_wins = sum(v for k, v in rep["wins"].items() if k not in ("honest", "none"))
            finals.append({"scenario": sc["name"], "mechanism": sc["mechanism"], "fraction": frac,
                           "final_accuracy": rep["final_accuracy"],
                           "attacker_win_share": round(bad_wins / args.rounds, 3),
                           "honest_rejected_share": rep["rejected_share"].get("honest", 0.0),
                           "verify_overhead": rep["verify_overhead"],
                           "useful_work_share": rep["useful_work_share"]})
            for r in res["rounds"]:
                per_round.append({"scenario": sc["name"], "fraction": frac, **r})

    for name, rows in (("final.csv", finals), ("rounds.csv", per_round)):
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    base = centralized(Config(rounds=args.rounds), load_dataset("mnist"))
    with open(os.path.join(args.out, "baseline.json"), "w") as f:
        json.dump({"baseline": base, "config": vars(args)}, f, indent=2)
    for x in finals:
        print(json.dumps(x, ensure_ascii=False))


if __name__ == "__main__":
    main()
