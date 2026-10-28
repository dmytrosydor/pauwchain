"""Worker invoked by the node: python worker/worker.py <task.json> (docs/contract.md, section 3)."""
import json
import os
import random
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.canonical import derive_seed
from worker.data import load_dataset, val_set
from worker.model import init_state
from worker.store import import_file, load_state, save_state
from worker.train import Cancelled, accuracy, ckpt_root, hash_states, to_micro, train_block
from worker.verify import verify_a, verify_b


def block_seed(task: dict) -> int:
    return derive_seed(
        bytes.fromhex(task["prev_block_hash"]),
        bytes.fromhex(task["miner_id"]),
        task["round"],
    )


def op_genesis(task: dict, stop=None) -> dict:
    return {"w_start": save_state(task["store_dir"], init_state(task["task_seed"]))}


def op_train(task: dict, stop=None) -> dict:
    ds = load_dataset(task["dataset"])
    device = task.get("device", "cpu")
    w_start = load_state(task["store_dir"], task["w_start"])
    ckpts = train_block(w_start, ds, block_seed(task), task["K"], task["k"],
                        task["batch_size"], task["lr"], device, stop=stop)
    hashes = hash_states(ckpts)
    for c in ckpts:
        save_state(task["store_dir"], c)
    metric = accuracy(ckpts[-1], *val_set(ds), device)
    return {
        "w_end": hashes[-1].hex(),
        "ckpt_hashes": [h.hex() for h in hashes],
        "ckpt_root": ckpt_root(hashes).hex(),
        "metric_micro": to_micro(metric),
        "steps": task["K"],
    }


def op_verify_b(task: dict, stop=None) -> dict:
    ds = load_dataset(task["dataset"])
    return verify_b(
        load_state(task["store_dir"], task["ckpt_from"]),
        load_state(task["store_dir"], task["ckpt_to"]),
        ds, block_seed(task), task["step_from"], task["k"],
        task["batch_size"], task["lr"], task["threshold_rule"],
        task.get("device", "cpu"),
    )


def op_verify_a(task: dict, stop=None) -> dict:
    ds = load_dataset(task["dataset"])
    rng = random.Random(task.get("validator_seed"))  # no seed: private OS randomness
    return verify_a(
        load_state(task["store_dir"], task["w_end"]),
        ds, task["metric_micro"], task["subset"], task["eps"], rng,
        task.get("device", "cpu"),
    )


def op_import(task: dict, stop=None) -> dict:
    return {"valid": import_file(task["store_dir"], task["path"], task["hash"])}


OPS = {"genesis": op_genesis, "train": op_train, "verify_b": op_verify_b, "verify_a": op_verify_a,
       "import": op_import}


def run_task(task: dict, stop=None) -> dict:
    """Runs one operation; errors are returned as {"ok": false, "error": ...}, not raised.
    Shared by the one-shot run (main) and the persistent worker (worker/server.py)."""
    t0 = time.perf_counter()
    try:
        result = {"ok": True, **OPS[task["op"]](task, stop=stop)}
    except Cancelled:
        result = {"ok": False, "cancelled": True, "error": "cancelled"}
    except Exception as e:  # noqa: BLE001 — any worker error must reach the node as a result
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    result["seconds"] = round(time.perf_counter() - t0, 3)
    return result


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: worker.py <task.json>", file=sys.stderr)
        return 2
    with open(sys.argv[1]) as f:
        task = json.load(f)
    result = run_task(task)

    out = task.get("out")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
