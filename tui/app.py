"""pouwtop: вузли, ланцюг, перевірки й точність моделі наживо.

Клавіші: ↑/↓ — вибрати вузол, l — перемкнути лінивий режим (атака 1), p — пауза майнінгу,
c — очистити журнал, q — вихід.
"""
import threading
import time
from typing import Dict, List

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, RichLog, Sparkline, Static

from tui.client import NodeClient
from tui.view import describe, short, who

QUIET = {"tip"}  # події, які оновлюють панелі, але не пишуться в журнал
COLUMNS = [("вузол", 6), ("API", 6), ("режим", 12), ("висота", 7), ("вершина", 10), ("знайшов", 8),
           ("прийняв", 8), ("відхилив", 9), ("програв", 8), ("піри", 5)]


class Pouwtop(App):
    TITLE = "pouwtop"
    SUB_TITLE = "мережа, що навчає модель"
    CSS = """
    #nodes { height: 8; }
    #middle { height: 1fr; }
    #blocks-box { width: 3fr; }
    #acc-box { width: 2fr; padding: 0 1; }
    #acc { height: 1fr; }
    #log { height: 1fr; border-top: solid $primary; }
    .title { color: $text-muted; }
    """
    BINDINGS = [
        ("l", "toggle_lazy", "лінивий режим"),
        ("p", "toggle_pause", "пауза"),
        ("c", "clear_log", "очистити журнал"),
        ("q", "quit", "вихід"),
    ]

    def __init__(self, ports: List[int]):
        super().__init__()
        self.clients = {p: NodeClient(p) for p in ports}
        self.statuses: Dict[int, dict] = {}
        self.names: Dict[str, str] = {}  # miner_id → id вузла
        self.stopping = threading.Event()

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="nodes", cursor_type="row", zebra_stripes=True)
        with Horizontal(id="middle"):
            with Vertical(id="blocks-box"):
                yield Static("Ланцюг вибраного вузла", classes="title")
                yield DataTable(id="blocks", zebra_stripes=True)
            with Vertical(id="acc-box"):
                yield Static("Точність моделі по блоках", id="acc-label", classes="title")
                yield Sparkline([], id="acc")
        yield RichLog(id="log", markup=True, wrap=True, max_lines=2000)
        yield Footer()

    def on_mount(self) -> None:
        nodes = self.query_one("#nodes", DataTable)
        for col, width in COLUMNS:
            nodes.add_column(col, key=col, width=width)
        blocks = self.query_one("#blocks", DataTable)
        for col, width in (("висота", 7), ("хеш", 10), ("майнер", 12), ("точність", 10)):
            blocks.add_column(col, key=col, width=width)
        for port in self.clients:
            nodes.add_row(*["…"] * 10, key=str(port))
            self.follow(port)
        self.set_interval(1.0, self.poll_status)
        self.poll_status()

    def on_unmount(self) -> None:
        self.stopping.set()

    # --- дані -------------------------------------------------------------------------------

    @work(thread=True)
    def poll_status(self) -> None:
        for port, client in self.clients.items():
            try:
                st = client.status()
            except (OSError, RuntimeError):
                st = None
            self.call_from_thread(self.show_status, port, st)

    @work(thread=True)
    def follow(self, port: int) -> None:
        """Потік подій вузла; при обриві перепідключається й продовжує з останнього seq."""
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
    def load_chain(self, port: int) -> None:
        try:
            chain = self.clients[port].chain()
        except (OSError, RuntimeError):
            return
        self.call_from_thread(self.show_chain, port, chain)

    # --- відображення -----------------------------------------------------------------------

    def selected_port(self) -> int:
        table = self.query_one("#nodes", DataTable)
        ports = list(self.clients)
        return ports[min(table.cursor_row, len(ports) - 1)] if ports else 0

    def show_status(self, port: int, st) -> None:
        table = self.query_one("#nodes", DataTable)
        if st is None:
            row = ["?", str(port), Text("недоступний", style="red")] + ["-"] * 7
        else:
            self.names[st["miner_id"]] = st["id"]
            mode = Text("ЛІНИВИЙ", style="bold yellow") if st["lazy"] else Text("сумлінний")
            if st.get("paused"):
                mode.append(" ⏸")
            # Пара вузлів з'єднана в обидва боки, тож рахуємо різні node_id, а не з'єднання.
            peers = len({p["node_id"] for p in st["peers"]})
            row = [st["id"], str(port), mode, str(st["height"]), short(st["tip"]), str(st["mined"]),
                   str(st["accepted"]), str(st["rejected"]), str(st["lost_races"]), str(peers)]
            first = port not in self.statuses
            self.statuses[port] = st
            if first and port == self.selected_port():
                self.load_chain(port)
        for (col, _), value in zip(COLUMNS, row):
            table.update_cell(str(port), col, value)

    def show_chain(self, port: int, chain: dict) -> None:
        if port != self.selected_port():
            return
        table = self.query_one("#blocks", DataTable)
        table.clear()
        blocks = chain["blocks"]
        for b in reversed(blocks[-200:]):
            h = b["header"]
            acc = f"{h['metric_micro'] / 1e4:.2f} %" if h["height"] else "—"
            miner = who(h["miner_id"], self.names) if h["height"] else "генезис"
            table.add_row(str(h["height"]), short(b["hash"]), miner, acc)
        accs = [b["header"]["metric_micro"] / 1e4 for b in blocks if b["header"]["height"]]
        self.query_one("#acc", Sparkline).data = accs or [0]
        label = f"Точність моделі по блоках: {accs[-1]:.2f} %" if accs else "Точність моделі по блоках"
        self.query_one("#acc-label", Static).update(label)

    def on_node_event(self, port: int, ev: dict) -> None:
        st = self.statuses.get(port)
        node = st["id"] if st else str(port)
        if ev["type"] not in QUIET:
            stamp = time.strftime("%H:%M:%S", time.localtime(ev["ts"] / 1000))
            self.query_one("#log", RichLog).write(f"[dim]{stamp}[/] [b]вузол {node}[/]  {describe(ev, self.names)}")
        if ev["type"] == "tip" and port == self.selected_port():
            self.load_chain(port)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "nodes" and event.row_key is not None:
            self.load_chain(int(event.row_key.value))

    # --- дії --------------------------------------------------------------------------------

    @work(thread=True)
    def admin(self, port: int, **request) -> None:
        try:
            self.clients[port].admin(**request)
        except (OSError, RuntimeError) as e:
            self.call_from_thread(self.query_one("#log", RichLog).write, f"[red]помилка: {e}[/]")
        self.poll_status()

    def action_toggle_lazy(self) -> None:
        port = self.selected_port()
        st = self.statuses.get(port)
        if st:
            self.admin(port, mode="honest" if st["lazy"] else "lazy")

    def action_toggle_pause(self) -> None:
        port = self.selected_port()
        st = self.statuses.get(port)
        if st:
            self.admin(port, paused=not st.get("paused", False))

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()
