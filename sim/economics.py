"""Economics of cheating based on the measured E2 matrix.

The cost unit is K training steps (one honest block). N miners compete for a block with reward R
in the same units (the race, Q1). An honest miner wins 1/N of the rounds on average.
An attack with cost c < 1 submits before the honest miners and wins the round if not caught (w = 1);
an attack with c ≥ 1 arrives no earlier than the honest miners (w = 1/N).
  p   = 1 − (1 − p₁)^(V·(1 − s))        p₁ is detection by one validator (from E2), V validators,
                                        each skips verification with probability s (verifier's dilemma)
  expected cheater payoff  (1 − p)·w·R − c − p·S
  expected honest payoff   R/N − 1
  minimum stake S* = ((1 − p)·w·R − c − R/N + 1) / p
S* ≤ 0: cheating does not pay even without a stake; p = 0: no stake will help.

python -m sim.economics experiments/runs/e2 [N] [R]
"""
import csv
import json
import os
import sys
from collections import defaultdict


def stake(p1: float, cost: float, reward: float, miners: int, validators: int, skip: float) -> float:
    p = 1 - (1 - p1) ** (validators * (1 - skip))
    if p <= 0:
        return float("inf")
    w = 1.0 if cost < 1 else 1.0 / miners
    return ((1 - p) * w * reward - cost - reward / miners + 1) / p


def main(run_dir: str, miners: int = 4, reward: float = 8.0) -> None:
    with open(os.path.join(run_dir, "matrix.csv")) as f:
        rows = list(csv.DictReader(f))
    rej, n, cost = defaultdict(int), defaultdict(int), defaultdict(list)
    for r in rows:
        key = (r["attack"], r["mechanism"])
        rej[key] += int(r["rejected"])
        n[key] += 1
        cost[r["attack"]].append(float(r["miner_steps"]) / float(r["K"]))
    attacks = [a for a in dict.fromkeys(r["attack"] for r in rows) if a != "honest"]
    mechs = list(dict.fromkeys(r["mechanism"] for r in rows))
    out = []
    for a in attacks:
        c = sum(cost[a]) / len(cost[a])
        for m in mechs:
            p1 = rej[a, m] / n[a, m]
            for V, s in ((1, 0.0), (3, 0.0), (3, 0.5)):
                out.append({"attack": a, "mechanism": m, "cost_K": round(c, 3), "p1": round(p1, 3),
                            "validators": V, "skip": s,
                            "stake_min": round(stake(p1, c, reward, miners, V, s), 3)})
    path = os.path.join(run_dir, "economics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(json.dumps([x for x in out if x["validators"] == 3 and x["skip"] == 0.0], ensure_ascii=False, indent=1))
    print(f"→ {path}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4, float(sys.argv[3]) if len(sys.argv) > 3 else 8.0)
