"""Client for the node HTTP API (node/src/api.hpp): JSON requests and the Server-Sent Events stream.

Standard library only: the monitor needs nothing besides Textual.
"""
import http.client
import json
from typing import Callable, Iterable, Iterator, Optional


class NodeClient:
    def __init__(self, port: int, host: str = "127.0.0.1", timeout: float = 3.0):
        self.host, self.port, self.timeout = host, port, timeout

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            payload = json.dumps(body) if body is not None else None
            conn.request(method, path, body=payload, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = json.loads(resp.read() or b"{}")
            if resp.status >= 400:
                raise RuntimeError(f"{method} {path}: {resp.status} {data.get('error', '')}")
            return data
        finally:
            conn.close()

    def status(self) -> dict:
        return self._request("GET", "/status")

    def chain(self, from_height: int = 0) -> dict:
        return self._request("GET", f"/chain?from={from_height}")

    def admin(self, **request) -> dict:
        return self._request("POST", "/admin", request)

    def events(self, since: int = 0, stop: Callable[[], bool] = lambda: False) -> Iterator[dict]:
        """Events with seq > since, while the connection is alive and stop() has not returned True."""
        conn = http.client.HTTPConnection(self.host, self.port, timeout=30)
        try:
            conn.request("GET", f"/events?since={since}")
            resp = conn.getresponse()
            for ev in parse_sse(_lines(resp)):
                if stop():
                    return
                yield ev
        finally:
            conn.close()


def _lines(resp) -> Iterator[str]:
    while True:
        raw = resp.readline()
        if not raw:
            return
        yield raw.decode("utf-8").rstrip("\r\n")


def parse_sse(lines: Iterable[str]) -> Iterator[dict]:
    """Parses an SSE stream: frames are separated by a blank line, the payload is in data: lines.
    Comments (": keepalive") are skipped."""
    data: list = []
    for line in lines:
        if line == "":
            if data:
                yield json.loads("\n".join(data))
                data = []
        elif line.startswith(":"):
            continue
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield json.loads("\n".join(data))
