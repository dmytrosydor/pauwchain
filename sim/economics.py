"""Економіка обману на основі виміряної матриці E2.

Одиниця вартості — K кроків навчання (один сумлінний блок). N майнерів змагаються за блок з нагородою R
у тих самих одиницях (перегони, В1). Сумлінний майнер виграє в середньому 1/N раундів.
Атака з вартістю c < 1 подає блок раніше за сумлінних і виграє раунд, якщо її не спіймали (w = 1);
атака з c ≥ 1 приходить не раніше за сумлінних (w = 1/N).
  p   = 1 − (1 − p₁)^(V·(1 − s))        p₁ — виявлення одним валідатором (з E2), V валідаторів,
                                        кожен пропускає перевірку з імовірністю s (дилема верифікатора)
  очікуваний виграш шахрая  (1 − p)·w·R − c − p·S
  очікуваний виграш чесного R/N − 1
  мінімальна застава S* = ((1 − p)·w·R − c − R/N + 1) / p
S* ≤ 0 — обман невигідний і без застави; p = 0 — жодна застава не допоможе.

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
