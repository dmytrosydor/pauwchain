"""Single-process simulator of network rounds (no C++ or TCP).

Round: all miners start from the chain tip, blocks arrive in order of work "cost"
(lazy instantly, honest after K steps), and honest nodes verify them in turn.
The first block that no validator rejects wins (Q1, the race). Rejection by a single
validator is sufficient: anyone can replay a checked transition (fraud proof, D5).

python -m sim.simulator --config experiments/configs/kt1.json
"""
import argparse
import csv
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List

from sim import attacks
from sim.core import TASK_ID, Block, Task, block_seed, miner_id
from sim.mechanisms import run
from worker.canonical import canonical_weight_hash
from worker.data import load_dataset, test_set, val_set
from worker.model import init_state
from worker.train import accuracy, to_micro, train_steps


@dataclass
class Config:
    name: str = "run"
    dataset: str = "mnist"
    rounds: int = 10
    n_honest: int = 3
    attackers: List[Dict] = field(default_factory=list)  # [{"kind": "lazy", ...attack parameters}]
    mechanism: str = "B"
    mech_params: Dict = field(default_factory=dict)
    K: int = 200
    k: int = 20
    batch_size: int = 64
    lr: float = 0.05
    task_seed: int = 42
    seed: int = 0
    miner_devices: List[str] = field(default_factory=lambda: ["cpu"])
    validator_devices: List[str] = field(default_factory=lambda: ["cpu"])
    max_validators: int = 0  # 0 means all honest nodes verify
    commit_reveal: bool = True  # for C: the plagiarist cannot see others' results before its own commitment
    verify_all: bool = True  # also verify blocks after the winner (detection statistics only)
    baseline: bool = True
    out_dir: str = "experiments/runs"


def arrival(kind: str, cfg: Config, params: Dict) -> float:
    """Nominal block arrival time in training steps: whoever works less submits earlier."""
    m = cfg.K // cfg.k
    return {
        "honest": cfg.K,
        "lazy": 0,
        "plagiarism": cfg.K + 1,  # right after the first honest block
        "partial": params.get("frac", 0.1) * cfg.K,
        "poison": cfg.K,
        "spoof_tiny": m,
        "spoof_tolerance": 2 * cfg.K,
        "one_jump": cfg.K,
        "metric_lie": 0,
    }[kind]


def simulate(cfg: Config) -> Dict:
    ds = load_dataset(cfg.dataset)
    rng = random.Random(cfg.seed)
    nodes = [{"id": i, "kind": "honest", "params": {}} for i in range(cfg.n_honest)]
    for j, a in enumerate(cfg.attackers):
        params = {k: v for k, v in a.items() if k not in ("kind", "speed")}
        nodes.append({"id": cfg.n_honest + j, "kind": a["kind"], "params": params,
                      "speed": a.get("speed", 1.0)})
    for n in nodes:
        n.setdefault("speed", 1.0)
        n["miner"] = miner_id(n["id"])
        n["device"] = cfg.miner_devices[n["id"] % len(cfg.miner_devices)]
    validators = [n for n in nodes if n["kind"] == "honest"]

    def task_for(device: str) -> Task:
        return Task(ds, cfg.K, cfg.k, cfg.batch_size, cfg.lr, device)

    tip, prev_hash = init_state(cfg.task_seed), bytes(32)
    round_rows, cand_rows = [], []
    totals = {"miner_steps": 0.0, "verify_steps": 0.0, "useful_steps": 0.0}
    acc0 = accuracy(tip, *test_set(ds))
    t0 = time.perf_counter()

    for r in range(cfg.rounds):
        # Faster hardware (speed > 1) submits the same work earlier.
        order = sorted(nodes, key=lambda n: (arrival(n["kind"], cfg, n["params"]) / n["speed"], rng.random()))
        # All miners work in parallel; their work is spent even if the block loses.
        totals["miner_steps"] += sum(arrival(n["kind"], cfg, n["params"]) if n["kind"] != "plagiarism" else 0
                                     for n in nodes)
        cache: Dict[int, Block] = {}
        replication = "C" in cfg.mechanism.split("+")
        tip_metric = to_micro(accuracy(tip, *val_set(ds)))
        params = {**cfg.mech_params, "A": {"prev_metric_micro": tip_metric, **cfg.mech_params.get("A", {})}}

        def block_of(n) -> Block:
            if n["id"] not in cache:
                # For C everyone runs the same task with the task seed (D8), otherwise the seed is per miner (D2).
                seed = block_seed(prev_hash, TASK_ID if replication else n["miner"], r)
                task = task_for(n["device"])
                kind, victim = n["kind"], None
                if kind == "plagiarism" and replication and cfg.commit_reveal:
                    kind = "lazy"  # nothing to copy before the reveal
                elif kind == "plagiarism":
                    first_honest = next(x for x in order if x["kind"] == "honest")
                    victim = block_of(first_honest)
                blk = attacks.make(kind, task, tip, n["miner"], seed, victim=victim, **n["params"])
                blk.kind = n["kind"]
                cache[n["id"]] = blk
            return cache[n["id"]]

        winner = None
        for n in order:
            if winner and not cfg.verify_all:
                break
            blk = block_of(n)
            verdicts = []
            pool = [v for v in validators if v["id"] != n["id"]]
            if cfg.max_validators and len(pool) > cfg.max_validators:
                pool = rng.sample(pool, cfg.max_validators)
            for v in pool:
                vtask = task_for(cfg.validator_devices[v["id"] % len(cfg.validator_devices)])
                vrng = random.Random(f"{cfg.seed}/{r}/{v['id']}/{n['id']}")
                verdicts.append(run(cfg.mechanism, vtask, blk, tip, vrng, params))
            rejected = any(x.rejected for x in verdicts)
            vsteps = sum(x.cost_steps for x in verdicts)
            if not winner:
                # After the winner, blocks at this height are stale: nobody in the network would verify them.
                totals["verify_steps"] += vsteps
            cand_rows.append({"round": r, "node": n["id"], "kind": n["kind"], "rejected": int(rejected),
                              "after_winner": int(winner is not None),
                              "validators": len(verdicts), "verify_steps": vsteps,
                              "verify_s": round(sum(x.seconds for x in verdicts), 3)})
            if not rejected and not winner:
                winner = (n, blk)

        if winner:
            n, blk = winner
            tip = blk.w_end
            if n["kind"] == "honest":
                totals["useful_steps"] += cfg.K
            prev_hash = hashlib.sha256(prev_hash + canonical_weight_hash(tip) + n["miner"]).digest()
        acc = accuracy(tip, *test_set(ds))
        round_rows.append({"round": r, "winner": winner[0]["id"] if winner else -1,
                           "winner_kind": winner[0]["kind"] if winner else "none",
                           "accuracy": round(acc, 4), "elapsed_s": round(time.perf_counter() - t0, 1)})
        print(f"round {r}: winner={round_rows[-1]['winner_kind']:16s} acc={acc:.4f}", flush=True)

    baseline = centralized(cfg, ds) if cfg.baseline else []
    return {"rounds": round_rows, "candidates": cand_rows, "baseline": baseline,
            "totals": totals, "acc0": acc0}


def centralized(cfg: Config, ds) -> List[float]:
    """Baseline: the same training in one place, R·K steps with one seed, accuracy after every K."""
    state, out = init_state(cfg.task_seed), []
    for r in range(cfg.rounds):
        state, _ = train_steps(state, ds, cfg.task_seed, r * cfg.K, cfg.K, cfg.batch_size, cfg.lr)
        out.append(round(accuracy(state, *test_set(ds)), 4))
    return out


def report(cfg: Config, res: Dict) -> Dict:
    by_kind: Dict[str, List[int]] = {}
    for c in res["candidates"]:
        by_kind.setdefault(c["kind"], []).append(c["rejected"])
    wins: Dict[str, int] = {}
    for x in res["rounds"]:
        wins[x["winner_kind"]] = wins.get(x["winner_kind"], 0) + 1
    t = res["totals"]
    spent = t["miner_steps"] + t["verify_steps"]
    return {
        "rejected_share": {k: round(sum(v) / len(v), 3) for k, v in by_kind.items()},
        "submissions": {k: len(v) for k, v in by_kind.items()},
        "wins": wins,
        "final_accuracy": res["rounds"][-1]["accuracy"],
        "baseline_final_accuracy": res["baseline"][-1] if res["baseline"] else None,
        "useful_work_share": round(t["useful_steps"] / spent, 3) if spent else 0,
        "verify_overhead": round(t["verify_steps"] / t["miner_steps"], 3) if t["miner_steps"] else 0,
    }


def save(cfg: Config, res: Dict, rep: Dict) -> str:
    out = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(out, exist_ok=True)
    for name in ("rounds", "candidates"):
        rows = res[name]
        if rows:
            with open(os.path.join(out, f"{name}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"config": cfg.__dict__, "report": rep, "baseline": res["baseline"],
                   "acc0": res["acc0"]}, f, indent=2, ensure_ascii=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = Config(**json.load(f))
    res = simulate(cfg)
    rep = report(cfg, res)
    out = save(cfg, res, rep)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
