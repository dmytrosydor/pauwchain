"""pouwtop: watch a pouwchain network train a model and catch cheaters, live.

Keys: ←/→ choose a node, c makes it cheat (or honest again), p pauses it, ? explains the screen, q quits.
"""
import threading
import time
from typing import Dict, List, Optional

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, RichLog, Static

from tui.client import NodeClient
from tui.view import ago, card, chart, dedupe_key, describe, headline, who

HELP = """\
[b]What am I looking at?[/b]

A small network of computers ([b]nodes[/b]) is [b]training an AI model together[/b] — it learns to read
handwritten digits. Instead of wasting power on puzzles like Bitcoin, each computer earns
the right to add a block by doing a piece of real training.

[b]The catch:[/b] anyone could lie and say "I trained" without doing it. So every block is
[b]checked by the others[/b]: they secretly pick a couple of pieces of the work and redo them.
If the numbers don't match, the block is thrown out.

[b]On the screen[/b]
  [b]top[/b]          is the network healthy, how good is the model, how many cheats were caught
  [b]nodes[/b]        each one's role: [green]honest[/green] trains, [red]CHEATER[/red] sends fake blocks
  [b]accuracy[/b]     how the model got better, block by block
  [b]blocks[/b]       the shared chain: who trained each block and the accuracy after it
  [b]feed[/b]         what just happened, in plain words

[b]Try it[/b]: pick an honest node with ←/→ and press [b]c[/b] — watch the others catch it.
Press [b]c[/b] again to make it honest.

[dim]Press any key to close.[/dim]"""


class Help(ModalScreen):
    CSS = """
    Help { align: center middle; }
    #help { width: 96; height: auto; padding: 1 3; border: round $accent; background: $surface; }
    """

    def compose(self) -> ComposeResult:
        yield Static(Text.from_markup(HELP), id="help")

    def on_key(self) -> None:
        self.dismiss()


class Pouwtop(App):
    TITLE = "pouwtop"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { background: $background; }
    #headline { height: 5; padding: 0 2; border: round $success; }
    #headline.bad { border: round $error; }
    #headline.busy { border: round $warning; }
    .section { color: $text-muted; padding: 0 1; height: 1; }
    #cards { height: 7; }
    .card { width: 1fr; height: 7; padding: 0 1; border: round $panel-lighten-2; }
    .card.selected { border: thick $accent; }
    .card.cheater { border: round $error; }
    .card.cheater.selected { border: thick $error; }
    #middle { height: 1fr; min-height: 10; }
    #chart-box { width: 1fr; border: round $panel-lighten-2; padding: 0 1; }
    #blocks { width: 1fr; border: round $panel-lighten-2; }
    #feed { height: 1fr; min-height: 6; border: round $panel-lighten-2; padding: 0 1; }
    """
    BINDINGS = [
        Binding("left", "select(-1)", "choose node", key_display="←→", priority=True),
        Binding("right", "select(1)", "choose node", show=False, priority=True),
        Binding("c", "toggle_cheat", "make it cheat / honest"),
        Binding("p", "toggle_pause", "pause it"),
        Binding("question_mark", "help", "what is this?", key_display="?"),
        Binding("q", "quit", "quit"),
    ]

    def __init__(self, ports: List[int]):
        super().__init__()
        self.clients = {p: NodeClient(p) for p in ports}
        self.ports = list(ports)
        self.statuses: Dict[int, Optional[dict]] = {p: None for p in ports}
        self.names: Dict[str, str] = {}  # miner_id → node id
        self.selected = 0
        self.chain: List[dict] = []
        self.seen: set = set()  # feed lines already shown (every node reports the same block)
        self.caught: set = set()  # hashes of fake blocks thrown out
        self.stopping = threading.Event()

    def compose(self) -> ComposeResult:
        yield Static(id="headline")
        yield Static("NODES — the computers in the network.  [b]←/→[/] select one, [b]c[/] make it cheat",
                     classes="section")
        with Horizontal(id="cards"):
            for i, port in enumerate(self.ports):
                c = Static(id=f"card-{port}", classes="card" + (" selected" if i == 0 else ""))
                c.border_title = f"port {port}"
                yield c
        with Horizontal(id="middle"):
            with Vertical(id="chart-box"):
                yield Static(id="chart")
            yield DataTable(id="blocks", cursor_type="none", zebra_stripes=True)
        yield RichLog(id="feed", markup=True, wrap=True, max_lines=500)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chart-box").border_title = "model accuracy, block by block"
        blocks = self.query_one("#blocks", DataTable)
        blocks.border_title = "the chain (newest first)"
        blocks.can_focus = False
        for col, width in (("block", 7), ("trained by", 12), ("accuracy", 10), ("when", 11)):
            blocks.add_column(col, key=col, width=width)
        feed = self.query_one("#feed", RichLog)
        feed.border_title = "what's happening"
        feed.can_focus = False
        self.following = False
        self.set_interval(1.0, self.poll_status)
        self.poll_status()
        self.render_all()

    def on_unmount(self) -> None:
        self.stopping.set()

    # --- data ---------------------------------------------------------------------------------

    @work(thread=True)
    def poll_status(self) -> None:
        for port, client in self.clients.items():
            try:
                st = client.status()
            except (OSError, RuntimeError):
                st = None
            self.call_from_thread(self.set_status, port, st)
        # Events are followed only once the node names are known, so the feed says "node 4", not a hash.
        self.call_from_thread(self.start_following)

    def start_following(self) -> None:
        if not self.following:
            self.following = True
            for port in self.ports:
                self.follow(port)

    @work(thread=True)
    def follow(self, port: int) -> None:
        """Event stream of a node; reconnects on failure and resumes from the last seq."""
        client, last = self.clients[port], 0
        while not self.stopping.is_set():
            try:
                for ev in client.events(since=last, stop=self.stopping.is_set):
                    last = ev["seq"]
                    self.call_from_thread(self.on_node_event, port, ev)
            except (OSError, ValueError):
                pass
            self.stopping.wait(1.0)

    @work(thread=True, exclusive=True, group="chain")
    def load_chain(self) -> None:
        """The chain as an honest node sees it (a cheater's view is the same, but honest is the reference)."""
        honest = [p for p in self.ports if self.statuses.get(p) and not self.statuses[p]["lazy"]]
        for port in honest or self.ports:
            try:
                chain = self.clients[port].chain()
            except (OSError, RuntimeError):
                continue
            self.call_from_thread(self.set_chain, chain["blocks"])
            return

    def set_status(self, port: int, st: Optional[dict]) -> None:
        if st:
            self.names[st["miner_id"]] = st["id"]
        old = self.statuses.get(port)
        self.statuses[port] = st
        if st and (not old or old["height"] != st["height"] or not self.chain):
            self.load_chain()
        self.render_all()

    def set_chain(self, blocks: List[dict]) -> None:
        self.chain = blocks
        self.render_all()

    def on_node_event(self, port: int, ev: dict) -> None:
        d = ev.get("data", {})
        if ev["type"] == "verdict" and not d.get("accepted") and "hash" in d:
            self.caught.add(d["hash"])
        key = dedupe_key(ev)
        if key is not None:
            if key in self.seen:
                return
            self.seen.add(key)
        line = describe(ev, self.names)
        if line is None:
            return
        st = self.statuses.get(port)
        if ev["type"] == "mode" and st:
            line = f"[b]Node {st['id']}[/] {line}"
        stamp = time.strftime("%H:%M:%S", time.localtime(ev["ts"] / 1000))
        self.query_one("#feed", RichLog).write(Text.from_markup(f"[dim]{stamp}[/]  {line}"))

    # --- display ------------------------------------------------------------------------------

    def render_all(self) -> None:
        tip = self.chain[-1]["header"] if self.chain else None
        head = self.query_one("#headline", Static)
        head.update(headline(self.statuses, tip, len(self.caught)))
        up = [s for s in self.statuses.values() if s]
        head.set_class(not up, "bad")
        head.set_class(bool(up) and len({s["tip"] for s in up if not s["lazy"]}) > 1, "busy")

        per_miner: Dict[str, int] = {}
        for b in self.chain[1:]:
            per_miner[b["header"]["miner_id"]] = per_miner.get(b["header"]["miner_id"], 0) + 1
        for i, port in enumerate(self.ports):
            st = self.statuses.get(port)
            c = self.query_one(f"#card-{port}", Static)
            c.update(card(st, per_miner.get(st["miner_id"], 0) if st else 0))
            if st:
                c.border_title = f"node {st['id']}"
            c.set_class(i == self.selected, "selected")
            c.set_class(bool(st and st["lazy"]), "cheater")

        box = self.query_one("#chart-box")
        accs = [b["header"]["metric_micro"] / 1e4 for b in self.chain[1:]]
        width = max(10, box.size.width - 2)
        height = max(3, box.size.height - 3)
        self.query_one("#chart", Static).update(chart(accs, width, height))

        table = self.query_one("#blocks", DataTable)
        table.clear()
        now = time.time()
        for b in reversed(self.chain[-100:]):
            h = b["header"]
            if h["height"] == 0:
                table.add_row("#0", Text("start", style="dim"), "—", "")
                continue
            table.add_row(f"#{h['height']}", who(h["miner_id"], self.names), f"{h['metric_micro'] / 1e4:.1f} %",
                          Text(ago(h["timestamp"], now), style="dim") if h.get("timestamp") else "")

    # --- actions ------------------------------------------------------------------------------

    def action_select(self, step: int) -> None:
        self.selected = (self.selected + step) % len(self.ports)
        self.render_all()

    @work(thread=True)
    def admin(self, port: int, **request) -> None:
        try:
            self.clients[port].admin(**request)
        except (OSError, RuntimeError) as e:
            self.call_from_thread(self.query_one("#feed", RichLog).write, f"[red]could not reach the node: {e}[/]")
        self.poll_status()

    def action_toggle_cheat(self) -> None:
        port = self.ports[self.selected]
        st = self.statuses.get(port)
        if st:
            self.admin(port, mode="honest" if st["lazy"] else "lazy")

    def action_toggle_pause(self) -> None:
        port = self.ports[self.selected]
        st = self.statuses.get(port)
        if st:
            self.admin(port, paused=not st.get("paused", False))

    def action_help(self) -> None:
        self.push_screen(Help())

