"""Зведення для E1: FRR і частка виявлення на рівні одного переходу як функція порогу."""
from collections import defaultdict
from typing import Dict, List

# α = 0 — точний збіг (B-exact); логарифмічна сітка до 1, бо чесна похибка ~1e-6…1e-2.
ALPHAS = [0, 1e-6, 1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
TAUS = [-0.2, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4]


def _rates(values: List[float], thresholds, reject_if_above: bool) -> Dict[str, float]:
    out = {}
    for t in thresholds:
        hits = sum((v > t) if reject_if_above else (v < t) for v in values)
        out[str(t)] = round(hits / len(values), 4) if values else None
    return out


def roc(rows: List[Dict]) -> Dict:
    """Для кожного (k, пристрій валідатора): частка відхилених переходів за B (ratio > α) і D (cos < τ)."""
    groups = defaultdict(lambda: defaultdict(lambda: {"ratio": [], "cos": []}))
    for r in rows:
        g = groups[f"k={r['k']}/{r['device']}"][r["kind"]]
        g["ratio"].append(float(r["ratio"]))
        g["cos"].append(float(r["cos"]))
    out = {}
    for key, kinds in groups.items():
        out[key] = {
            "B_reject_share_by_alpha": {kind: _rates(v["ratio"], ALPHAS, True) for kind, v in kinds.items()},
            "D_reject_share_by_tau": {kind: _rates(v["cos"], TAUS, False) for kind, v in kinds.items()},
            "honest_ratio_max": max(kinds["honest"]["ratio"]) if "honest" in kinds else None,
            "n": {kind: len(v["ratio"]) for kind, v in kinds.items()},
        }
    return out


def wilson(hits: int, n: int, z: float = 1.96):
    """95 % довірчий інтервал Вільсона для частки hits/n."""
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
