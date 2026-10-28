"""The UI without a real network: the client is replaced by a stub."""
import asyncio
import threading
import time

from rich.text import Text
from textual.widgets import DataTable, RichLog, Static

import tui.app as app_mod
from tui.view import card, dedupe_key, describe, headline

ACCS = [84.6, 94.3, 95.8, 96.4, 96.9, 97.1, 97.3, 97.2, 97.6, 97.7, 97.9, 97.8, 98.0, 98.1]


STRANGER = "8adeb7" + "1f" * 29  # a wallet that is not one of the nodes


def miner(port: int) -> str:
    return f"{port:064x}"


def transfer(n: int) -> dict:
    return {"id": f"{n:064x}", "chain": "c0" * 32, "from": miner(9201), "to": STRANGER, "amount": 10, "fee": 1,
            "nonce": n, "sig": "5a" * 64}


class FakeClient:
    def __init__(self, port: int, **_):
        self.port = port
        self.lazy = port == 9204
        self.admin_calls = []

    def status(self):
        return {"id": str(self.port - 9200), "miner_id": miner(self.port), "lazy": self.lazy,
                "paused": False, "height": len(ACCS), "tip": "ab" * 32, "mined": 9 if self.lazy else 4,
                "accepted": 8, "rejected": 0 if self.lazy else 9, "lost_races": 0,
                "peers": [{"peer": 1, "node_id": "2"}],
                "balance": 0 if self.lazy else 150 + (self.port - 9201) * 50, "supply": 50 * len(ACCS) + 2,
                "mempool": 1 if self.lazy else 2, "task_id": "7e" * 32}

    def chain(self, from_height=0):
        now = int(time.time())
        blocks = [{"hash": "00" * 32, "header": {"height": 0, "miner_id": "00" * 32, "metric_micro": 0}}]
        for h, acc in enumerate(ACCS, 1):
            blocks.append({"hash": f"{h:064x}", "header": {
                "height": h, "miner_id": miner(9201 + h % 3), "metric_micro": int(acc * 1e4),
                "timestamp": now - (len(ACCS) - h) * 7}, "txs": [transfer(h)] * (h % 4 == 0) * (h // 4)})
        return {"height": len(ACCS), "blocks": blocks}

    def admin(self, **req):
        self.admin_calls.append(req)
        if "mode" in req:
            self.lazy = req["mode"] == "lazy"
        return self.status()

    def events(self, since=0, stop=lambda: False):
        if since == 0:
            # Every node reports the same two blocks: the feed must show each once.
            yield {"seq": 1, "ts": 0, "type": "tip",
                   "data": {"height": 14, "hash": "cd" * 32, "miner": miner(9202), "metric_micro": 981000,
                            "txs": 2, "reward": 52}}
            yield {"seq": 2, "ts": 0, "type": "verdict",
                   "data": {"accepted": False, "height": 14, "hash": "ef" * 32, "miner": miner(9204),
                            "reason": "B: transition 2 failed",
                            "checks": [{"mech": "B", "j": 2, "distance": 0.2, "threshold": 0.01,
                                        "verdict": "reject"}]}}
            # ... and the same transfer entering its waiting list.
            t = transfer(99)
            yield {"seq": 3, "ts": 0, "type": "tx",
                   "data": {k: t[k] for k in ("id", "from", "to", "amount", "fee")}}
        threading.Event().wait(0.2)


def run_app(scenario, size=(120, 40)):
    async def main():
        app = app_mod.Pouwtop([9201, 9202, 9203, 9204])
        async with app.run_test(size=size) as pilot:
            await pilot.pause(1.0)
            await scenario(app, pilot)
            app.stopping.set()

    asyncio.run(main())


def test_screen_explains_itself(monkeypatch):
    monkeypatch.setattr(app_mod, "NodeClient", FakeClient)

    async def scenario(app, pilot):
        head = str(app.query_one("#headline", Static).render())
        assert "network is working" in head and "98.1 %" in head and "node 4 is cheating" in head
        assert "caught and thrown out: 1" in head
        assert "Coins in existence: 702" in head and "2 transfers waiting" in head
        assert "CHEATER" in str(app.query_one("#card-9204", Static).render())
        assert "honest" in str(app.query_one("#card-9201", Static).render())
        assert "coins        200" in str(app.query_one("#card-9202", Static).render())
        blocks = app.query_one("#blocks", DataTable)
        assert blocks.row_count == len(ACCS) + 1
        assert blocks.get_row_at(2)[3] == "3"  # block #12 holds 3 transfers
        assert blocks.get_row_at(1)[3] == ""
        lines = [line.text for line in app.query_one("#feed", RichLog).lines]
        assert sum("Caught node 4 cheating" in t for t in lines) == 1
        assert sum("Block #14 added" in t for t in lines) == 1
        assert any("node 2 earned 52 coins · 2 transfers confirmed" in t for t in lines)
        assert sum("node 1 sent 10 coins (+1 fee) to pw8adeb7…" in t for t in lines) == 1
        assert any("20×" in t for t in lines)

    run_app(scenario)


def test_keys_switch_selected_node(monkeypatch):
    monkeypatch.setattr(app_mod, "NodeClient", FakeClient)

    async def scenario(app, pilot):
        await pilot.press("left", "c")  # wraps around to the last node, the cheater
        await pilot.pause(0.5)
        assert app.clients[9204].admin_calls == [{"mode": "honest"}]
        await pilot.press("right", "p")
        await pilot.pause(0.5)
        assert app.clients[9201].admin_calls == [{"paused": True}]
        await pilot.press("question_mark")
        await pilot.pause(0.2)
        assert app.screen.__class__.__name__ == "Help"
        await pilot.press("x")
        await pilot.pause(0.2)
        assert app.screen.__class__.__name__ != "Help"

    run_app(scenario)


def test_screenshot(monkeypatch, tmp_path):
    """Renders the screen to an SVG (handy for README) and checks nothing crashes at a narrow width."""
    monkeypatch.setattr(app_mod, "NodeClient", FakeClient)

    async def scenario(app, pilot):
        app.save_screenshot(str(tmp_path / "pouwtop.svg"))

    run_app(scenario)
    run_app(lambda app, pilot: asyncio.sleep(0), size=(90, 30))
    assert (tmp_path / "pouwtop.svg").stat().st_size > 1000


def plain(markup: str) -> str:
    return Text.from_markup(markup).plain


def test_coin_lines_in_plain_words():
    names = {miner(9201): "1", miner(9202): "2"}
    tip = {"seq": 1, "ts": 0, "type": "tip",
           "data": {"height": 5, "hash": "aa" * 32, "miner": miner(9202), "metric_micro": 970000,
                    "txs": 1, "reward": 51}}
    assert plain(describe(tip, names)) == ("■ Block #5 added to the chain — trained by node 2, "
                                           "model accuracy 97.0 %, node 2 earned 51 coins · 1 transfer confirmed")
    tip["data"].update(txs=0, reward=50)
    assert plain(describe(tip, names)).endswith("node 2 earned 50 coins")
    tx = {"seq": 2, "ts": 0, "type": "tx",
          "data": {"id": "bb" * 32, "from": STRANGER, "to": miner(9201), "amount": 1, "fee": 0}}
    assert plain(describe(tx, names)) == "→ pw8adeb7… sent 1 coin to node 1 (waiting for the next block)"
    assert dedupe_key(tx) == ("tx", "bb" * 32)
    assert dedupe_key({"type": "tx", "data": {}}) is None


def test_coins_in_headline_and_card():
    st = FakeClient(9201).status()
    assert "coins        150" in card(st, 4).plain
    head = headline({9201: st, 9204: FakeClient(9204).status()}, None, 0).plain
    assert "Coins in existence: 702  ·  2 transfers waiting for a block" in head  # the cheater's view is ignored
    st["mempool"] = 0
    assert "no transfers waiting" in headline({9201: st}, None, 0).plain
    del st["supply"], st["balance"]  # an older node: no coin lines at all
    assert "Coins" not in headline({9201: st}, None, 0).plain and "coins" not in card(st, 4).plain
