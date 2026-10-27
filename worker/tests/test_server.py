"""Постійний воркер через Unix socket: ті самі результати, що й разовий запуск, і скасування задачі."""
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time

import pytest

from worker.worker import run_task

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def request(path: str, task: dict, close_after: float = None):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(path)
    data = json.dumps(task).encode()
    s.sendall(struct.pack(">I", len(data)) + data)
    if close_after is not None:
        time.sleep(close_after)
        s.close()
        return None
    n = struct.unpack(">I", s.recv(4, socket.MSG_WAITALL))[0]
    buf = b""
    while len(buf) < n:
        buf += s.recv(n - len(buf))
    s.close()
    return json.loads(buf)


@pytest.fixture()
def server():
    # Короткий шлях: sun_path на macOS обмежено ~104 байтами.
    d = tempfile.mkdtemp(prefix="pw", dir="/tmp")
    path = os.path.join(d, "w.sock")
    proc = subprocess.Popen([sys.executable, "worker/server.py", "--socket", path, "--preload", "synthetic"],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert "ready" in proc.stdout.readline()
    yield path, d
    proc.terminate()
    proc.wait(timeout=10)


def test_same_result_as_one_shot(server):
    path, d = server
    store = os.path.join(d, "store")
    w0 = request(path, {"op": "genesis", "task_seed": 7, "store_dir": store})["w_start"]
    task = {"op": "train", "task_id": "11" * 32, "round": 1, "miner_id": "aa" * 32,
            "prev_block_hash": "00" * 32, "w_start": w0, "K": 20, "k": 10, "batch_size": 32,
            "lr": 0.05, "dataset": "synthetic", "store_dir": store}
    via_server = request(path, task)
    direct = run_task(task)
    assert via_server["ok"] and via_server["ckpt_hashes"] == direct["ckpt_hashes"]

    vb = {**task, "op": "verify_b", "ckpt_from": direct["ckpt_hashes"][0], "ckpt_to": direct["ckpt_hashes"][1],
          "step_from": 0, "threshold_rule": {"kind": "relative", "alpha": 0.05}}
    assert request(path, vb)["verdict"] == "accept"


def test_concurrent_requests(server):
    path, d = server
    store = os.path.join(d, "store")
    import threading
    out = []
    ts = [threading.Thread(target=lambda s=s: out.append(request(path, {"op": "genesis", "task_seed": s,
                                                                         "store_dir": store})))
          for s in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len({r["w_start"] for r in out}) == 4


def test_closing_connection_cancels_training(server):
    path, d = server
    store = os.path.join(d, "store")
    w0 = request(path, {"op": "genesis", "task_seed": 7, "store_dir": store})["w_start"]
    long_task = {"op": "train", "task_id": "11" * 32, "round": 1, "miner_id": "aa" * 32,
                 "prev_block_hash": "00" * 32, "w_start": w0, "K": 200000, "k": 1000, "batch_size": 32,
                 "lr": 0.05, "dataset": "synthetic", "store_dir": store}
    request(path, long_task, close_after=0.5)
    # Задача на 200 000 кроків має зупинитися за кілька кроків після закриття з'єднання.
    deadline = time.perf_counter() + 5
    while request(path, {"op": "ping"})["active"] and time.perf_counter() < deadline:
        time.sleep(0.05)
    assert request(path, {"op": "ping"})["active"] == 0
