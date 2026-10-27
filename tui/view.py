"""Перетворення подій вузла на рядки журналу (Rich markup)."""
from typing import Dict


def short(h: str, n: int = 8) -> str:
    return h[:n]


def who(miner: str, names: Dict[str, str]) -> str:
    return f"вузол {names[miner]}" if miner in names else f"майнер {short(miner)}"


def describe(ev: dict, names: Dict[str, str]) -> str:
    t, d = ev["type"], ev.get("data", {})
    if t == "verdict":
        checks = d.get("checks", [])
        b = [c for c in checks if c.get("mech") == "B"]
        failed = [c for c in b if c["verdict"] != "accept"]
        if d.get("accepted"):
            side = " (бічна гілка)" if d.get("new_tip") is False else ""
            worst = max((c["distance"] / c["threshold"] for c in b if c["threshold"] > 0), default=0.0)
            info = f"  [dim]B {len(b)}/{len(b)}, макс. {worst:.0%} порогу[/]" if b else ""
            return f"[green]✓ прийняв[/] h={d['height']} від {who(d['miner'], names)}{side}{info}"
        info = "".join(f"  [dim]B j={c['j']}: {c['distance']:.2g} > {c['threshold']:.2g}[/]" for c in failed)
        return f"[bold red]✗ відхилив[/] h={d['height']} від {who(d['miner'], names)}: {d.get('reason', '')}{info}"
    if t == "mined":
        if d.get("lazy"):
            return f"[yellow]подав лінивий блок[/] h={d['height']} (без навчання)"
        return f"[cyan]знайшов блок[/] h={d['height']}, точність {d['metric_micro'] / 1e4:.2f} %"
    if t == "lost_race":
        return f"[dim]програв перегони на h={d['height']}[/]"
    if t == "mining_aborted":
        return f"[dim]перервав навчання h={d['height']}: нова вершина[/]"
    if t == "mode":
        mode = "ЛІНИВИЙ" if d.get("lazy") else "сумлінний"
        return f"[magenta]режим: {mode}{', пауза' if d.get('paused') else ''}[/]"
    if t == "peer":
        return f"[dim]підключився вузол {d.get('node_id', '?')} (висота {d.get('height')})[/]"
    if t == "genesis":
        return f"[dim]генезис {short(d['hash'])}[/]"
    return f"[dim]{t}[/]"
