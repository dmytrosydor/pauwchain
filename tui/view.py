"""Turns node events into log lines (Rich markup)."""
from typing import Dict


def short(h: str, n: int = 8) -> str:
    return h[:n]


def who(miner: str, names: Dict[str, str]) -> str:
    return f"node {names[miner]}" if miner in names else f"miner {short(miner)}"


def describe(ev: dict, names: Dict[str, str]) -> str:
    t, d = ev["type"], ev.get("data", {})
    if t == "verdict":
        checks = d.get("checks", [])
        b = [c for c in checks if c.get("mech") == "B"]
        failed = [c for c in b if c["verdict"] != "accept"]
        if d.get("accepted"):
            side = " (side branch)" if d.get("new_tip") is False else ""
            worst = max((c["distance"] / c["threshold"] for c in b if c["threshold"] > 0), default=0.0)
            info = f"  [dim]B {len(b)}/{len(b)}, max {worst:.0%} of threshold[/]" if b else ""
            return f"[green]✓ accepted[/] h={d['height']} from {who(d['miner'], names)}{side}{info}"
        info = "".join(f"  [dim]B j={c['j']}: {c['distance']:.2g} > {c['threshold']:.2g}[/]" for c in failed)
        return f"[bold red]✗ rejected[/] h={d['height']} from {who(d['miner'], names)}: {d.get('reason', '')}{info}"
    if t == "mined":
        if d.get("lazy"):
            return f"[yellow]submitted a lazy block[/] h={d['height']} (no training)"
        return f"[cyan]mined a block[/] h={d['height']}, accuracy {d['metric_micro'] / 1e4:.2f} %"
    if t == "lost_race":
        return f"[dim]lost the race at h={d['height']}[/]"
    if t == "mining_aborted":
        return f"[dim]stopped training h={d['height']}: new tip[/]"
    if t == "mode":
        mode = "LAZY" if d.get("lazy") else "honest"
        return f"[magenta]mode: {mode}{', paused' if d.get('paused') else ''}[/]"
    if t == "peer":
        return f"[dim]node {d.get('node_id', '?')} connected (height {d.get('height')})[/]"
    if t == "genesis":
        return f"[dim]genesis {short(d['hash'])}[/]"
    return f"[dim]{t}[/]"
