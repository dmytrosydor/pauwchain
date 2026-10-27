"""The UI without a real network: the client is replaced by a stub."""
import asyncio
import threading

from textual.widgets import DataTable, RichLog

import tui.app as app_mod


class FakeClient:
    def __init__(self, port: int, **_):
        self.port = port
        self.lazy = port == 9204
        self.admin_calls = []

    def status(self):
        return {"id": str(self.port - 9200), "miner_id": f"{self.port:064x}", "lazy": self.lazy,
                "paused": False, "height": 3, "tip": "ab" * 32, "mined": 1, "accepted": 2,
                "rejected": 0, "lost_races": 0, "peers": [{"peer": 1, "node_id": "2"}]}

    def chain(self, from_height=0):
        blocks = [{"hash": f"{h:064x}", "header": {"height": h, "miner_id": f"{9201:064x}",
                                                   "metric_micro": 900000 + h * 10000}} for h in range(4)]
        return {"height": 3, "blocks": blocks}

    def admin(self, **req):
        self.admin_calls.append(req)
        if "mode" in req:
            self.lazy = req["mode"] == "lazy"
        return self.status()

    def events(self, since=0, stop=lambda: False):
        if since == 0:
            yield {"seq": 1, "ts": 0, "type": "verdict",
                   "data": {"accepted": False, "height": 3, "miner": f"{9204:064x}", "reason": "B failed",
                            "checks": [{"mech": "B", "j": 2, "distance": 0.4, "threshold": 0.02,
                                        "verdict": "reject"}]}}
        threading.Event().wait(0.2)


def test_tables_log_and_toggle(monkeypatch):
    monkeypatch.setattr(app_mod, "NodeClient", FakeClient)

    async def scenario():
        app = app_mod.Pouwtop([9201, 9204])
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(1.0)
            nodes = app.query_one("#nodes", DataTable)
            assert [str(c) for c in nodes.get_row_at(0)][:4] == ["1", "9201", "honest", "3"]
            assert str(nodes.get_row_at(1)[2]) == "LAZY"
            assert app.query_one("#blocks", DataTable).row_count == 4
            assert any("rejected" in line.text for line in app.query_one("#log", RichLog).lines)
            await pilot.press("down", "l")
            await pilot.pause(0.5)
            assert app.clients[9204].admin_calls == [{"mode": "honest"}]
            app.stopping.set()

    asyncio.run(scenario())
