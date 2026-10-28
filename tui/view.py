"""Plain-language rendering for pouwtop: the headline, node cards, the accuracy chart and the event feed.

Pure functions over the node API's JSON, so they can be tested without a terminal.
"""
import time
from typing import Dict, List, Optional

from rich.text import Text

BARS = " ▁▂▃▄▅▆▇█"


def short(h: str, n: int = 8) -> str:
    return h[:n]


def who(miner: str, names: Dict[str, str]) -> str:
    return f"node {names[miner]}" if miner in names else f"miner {short(miner)}"


def accuracy(header: dict) -> float:
    return header["metric_micro"] / 1e4


def ago(ts: int, now: Optional[float] = None) -> str:
    s = max(0, int((now or time.time()) - ts))
    if s < 60:
        return f"{s} s ago"
    if s < 3600:
        return f"{s // 60} min ago"
    return f"{s // 3600} h ago"


def _how_far_off(checks: List[dict]) -> Optional[float]:
    """How many times the worst recomputed transition exceeded the allowed error."""
    ratios = [c["distance"] / c["threshold"] for c in checks
              if c.get("mech") == "B" and c.get("threshold", 0) > 0 and c.get("verdict") != "accept"]
    return max(ratios) if ratios else None


def describe(ev: dict, names: Dict[str, str]) -> Optional[str]:
    """One feed line (Rich markup) for an event, or None if the event is routine and stays out of the feed."""
    t, d = ev["type"], ev.get("data", {})
    if t == "tip":
        return (f"[green]■[/] Block [b]#{d['height']}[/] added to the chain — trained by {who(d['miner'], names)}, "
                f"model accuracy [b]{d['metric_micro'] / 1e4:.1f} %[/]")
    if t == "verdict" and not d.get("accepted"):
        cheater = who(d["miner"], names)
        reason = d.get("reason", "")
        if reason.startswith("B"):
            off = _how_far_off(d.get("checks", []))
            detail = f"redoing a piece of its training gave numbers {off:.0f}× further off than allowed" if off else \
                "its training could not be reproduced"
            return f"[red]✗ Caught {cheater} cheating[/] on block #{d['height']}: {detail}. Block thrown out."
        if reason.startswith("A"):
            return f"[red]✗ Caught {cheater} lying[/] about the accuracy of block #{d['height']}. Block thrown out."
        if "parent" in reason:
            return f"[red]✗ Threw out block #{d['height']} from {cheater}[/]: it was built on a fake block."
        return f"[red]✗ Threw out block #{d['height']} from {cheater}[/]: {reason}."
    if t == "unavailable":
        return (f"[yellow]… Could not download block #{d['height']} from {who(d['miner'], names)}[/] — "
                "will try again later.")
    if t == "mode":
        if d.get("paused"):
            return "[yellow]⏸ paused[/] — stopped training, still checking others"
        if d.get("lazy"):
            return "[bold yellow]⚠ switched to CHEATING[/] — will send fake blocks without training"
        return "[green]✓ back to honest work[/]"
    return None


def dedupe_key(ev: dict) -> Optional[tuple]:
    """Every node reports the same block; the feed shows it once."""
    d = ev.get("data", {})
    if ev["type"] in ("tip", "verdict", "unavailable") and "hash" in d:
        return ev["type"], d["hash"]
    return None


def headline(statuses: Dict[int, Optional[dict]], tip: Optional[dict], caught: int) -> Text:
    """Three lines on top: is the network fine, how good is the model, how many cheats were caught."""
    up = [s for s in statuses.values() if s]
    if not up:
        return Text.from_markup(
            "[bold red]● No connection to the network.[/]\n"
            "Start it with [b]docker compose up -d[/] (or [b]HEIGHT=0 scripts/demo.sh[/]) and wait a few seconds.")
    honest = [s for s in up if not s["lazy"]]
    agree = len({s["tip"] for s in honest}) <= 1
    height = max(s["height"] for s in up)
    if agree:
        who_agrees = {0: "No honest node is", 1: "The only honest node is", 2: "Both honest nodes are"}.get(
            len(honest), f"All {len(honest)} honest nodes are")
        status = f"[bold green]● The network is working.[/] {who_agrees} on the same chain of [b]{height}[/] blocks."
    else:
        status = "[bold yellow]● Nodes are catching up with each other…[/] (normal right after a new block)"
    if len(up) < len(statuses):
        status += f"  [red]{len(statuses) - len(up)} offline.[/]"
    model = (f"The shared AI model reads handwritten digits with [b]{accuracy(tip):.1f} %[/] accuracy."
             if tip and tip["height"] else "The AI model has not been trained yet.")
    cheaters = [s["id"] for s in up if s["lazy"]]
    if cheaters:
        who_cheats = ", ".join(f"node {c}" for c in cheaters)
        guard = (f"[red][b]{who_cheats}[/] {'is' if len(cheaters) == 1 else 'are'} cheating[/] — "
                 f"fake blocks caught and thrown out: [b]{caught}[/]")
    else:
        guard = f"No one is cheating right now.  Cheating attempts caught so far: [b]{caught}[/]"
    return Text.from_markup(f"{status}\n{model}\n{guard}")


def card(st: Optional[dict], in_chain: int) -> Text:
    """What one node is and what it is doing, in four short lines."""
    if st is None:
        return Text.from_markup("[red]✗ offline[/]\n\nnot responding")
    if st.get("paused"):
        role, doing = "[yellow]⏸ paused[/]", "only checking others"
    elif st["lazy"]:
        role, doing = "[bold red]⚠ CHEATER[/]", "sending fake blocks"
    else:
        role, doing = "[bold green]✓ honest[/]", f"training block #{st['height'] + 1}"
    if st["lazy"]:
        stats = f"fakes sent   [b]{st['mined']}[/]\nin the chain [b]{in_chain}[/] blocks"
    else:
        stats = f"in the chain [b]{in_chain}[/] blocks\ncaught       [b]{st['rejected']}[/] fakes"
    return Text.from_markup(f"{role}\n{doing}\n{stats}")


def chart(values: List[float], width: int, height: int) -> Text:
    """Bar chart of accuracy per block, newest on the right, with a % scale on the left.
    Few blocks get wide bars, so the chart fills its box; many blocks keep only the newest that fit."""
    if not values:
        return Text("no blocks yet", style="dim")
    plot = max(1, width - 7)
    values = values[-plot:]
    bar = max(1, min(6, plot // len(values)))
    lo = max(0.0, (min(values) // 10) * 10 - (10 if min(values) % 10 < 2 else 0))
    hi = 100.0
    rows = []
    for r in range(height, 0, -1):
        line = ""
        for v in values:
            fill = (v - lo) / (hi - lo) * height - (r - 1)  # how much of this cell is filled, in cells
            line += BARS[max(0, min(8, round(fill * 8)))] * (bar - 1 if bar > 2 else bar) + (" " if bar > 2 else "")
        label = f"{hi:.0f}%" if r == height else (f"{lo:.0f}%" if r == 1 else "")
        rows.append(f"{label:>5}│{line}")
    span = len(values) * bar
    left, right = "oldest", "newest →"
    rows.append(f"{'':>5}└{left}{right:>{max(1, span - len(left))}}")
    return Text("\n".join(rows), style="green")
