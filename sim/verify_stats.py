"""Зведення для E1: FRR і частка виявлення на рівні одного переходу як функція порогу."""
from collections import defaultdict
from typing import Dict, List

ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
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
