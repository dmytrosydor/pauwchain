"""Постійний воркер: ті самі операції, що й worker.py, але через Unix socket без перезапуску Python.

python worker/server.py --socket run/node-1/verify.sock [--preload mnist]

Протокол (docs/contract.md, розділ 3.5): клієнт відкриває з'єднання, надсилає один кадр
(4 байти довжини big-endian + JSON завдання) і отримує один кадр з результатом.
Щоб скасувати задачу, клієнт закриває з'єднання: навчання зупиняється на найближчому кроці.
Кожне з'єднання обслуговується окремим потоком. Сервер завершується сам, коли зникає батьківський процес.
"""
import argparse
import json
import os
import select
import socket
import struct
import sys
import threading
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.data import load_dataset
from worker.worker import run_task

MAX_FRAME = 8 << 20
_active = 0  # скільки задач виконується зараз; видно через {"op": "ping"}
_active_lock = threading.Lock()


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf


def read_frame(conn: socket.socket) -> dict:
    (n,) = struct.unpack(">I", recv_exact(conn, 4))
    if n > MAX_FRAME:
        raise ValueError(f"frame too large: {n}")
    return json.loads(recv_exact(conn, n))


def write_frame(conn: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode()
    conn.sendall(struct.pack(">I", len(data)) + data)


def watch_for_close(conn: socket.socket, stop: threading.Event, done: threading.Event) -> None:
    """Поки задача виконується, клієнт нічого не надсилає; EOF на сокеті означає скасування."""
    while not done.is_set():
        readable, _, _ = select.select([conn], [], [], 0.05)
        if readable:
            try:
                closed = conn.recv(1, socket.MSG_PEEK) == b""
            except OSError:
                closed = True
            if closed:
                stop.set()
            return


def handle(conn: socket.socket) -> None:
    global _active
    with conn:
        try:
            task = read_frame(conn)
        except (ConnectionError, ValueError, OSError):
            return
        if task.get("op") == "ping":
            with _active_lock:
                write_frame(conn, {"ok": True, "active": _active, "pid": os.getpid()})
            return
        stop, done = threading.Event(), threading.Event()
        threading.Thread(target=watch_for_close, args=(conn, stop, done), daemon=True).start()
        with _active_lock:
            _active += 1
        try:
            result = run_task(task, stop=stop.is_set)
        finally:
            with _active_lock:
                _active -= 1
        done.set()
        if not stop.is_set():
            try:
                write_frame(conn, result)
            except OSError:
                pass


def exit_with_parent(parent: int) -> None:
    while True:
        if os.getppid() != parent:
            os._exit(0)
        time.sleep(1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--preload", default="", help="набір даних, який завантажити заздалегідь")
    args = ap.parse_args()

    if args.preload:
        load_dataset(args.preload)
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    srv.listen(16)
    threading.Thread(target=exit_with_parent, args=(os.getppid(),), daemon=True).start()
    print(f"worker server ready on {args.socket}", flush=True)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
