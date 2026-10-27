"""Графіки для розділу 4 з CSV/JSON, які пишуть sim.simulator і sim.matrix.

python -m sim.plots accuracy experiments/runs/kt1_lazy
python -m sim.plots matrix experiments/runs/e2
python -m sim.plots e1 experiments/runs/e1
python -m sim.plots e3 experiments/runs/e3
"""
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-."]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150,
})


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _mean_by_round(rows):
    """Середня точність по раундах серед повторів (seed)."""
    by_round = defaultdict(list)
    for r in rows:
        by_round[int(r["round"])].append(float(r["accuracy"]))
    return [sum(by_round[k]) / len(by_round[k]) for k in sorted(by_round)]


def accuracy(run_dir: str) -> str:
    with open(os.path.join(run_dir, "summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(run_dir, "rounds.csv")) as f:
        rounds = list(csv.DictReader(f))
    xs = [0] + [int(r["round"]) + 1 for r in rounds]
    sim = [summary["acc0"]] + [float(r["accuracy"]) for r in rounds]
    base = [summary["acc0"]] + summary["baseline"]

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(xs, sim, color=BLUE, lw=2, marker="o", ms=4, label="мережа (PoUW)")
    if summary["baseline"]:
        ax.plot(xs, base, color=ORANGE, lw=2, ls="--", label="централізоване навчання")
        ax.annotate(f"{base[-1]:.3f}", (xs[-1], base[-1]), xytext=(6, 6), textcoords="offset points",
                    color=INK2, fontsize=9)
    ax.annotate(f"{sim[-1]:.3f}", (xs[-1], sim[-1]), xytext=(6, -12), textcoords="offset points",
                color=INK2, fontsize=9)
    ax.set_xlabel("раунд (блок)")
    ax.set_ylabel("точність на тестовому наборі")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = os.path.join(run_dir, "accuracy.png")
    fig.savefig(out)
    return out


def matrix(run_dir: str) -> str:
    with open(os.path.join(run_dir, "matrix.csv")) as f:
        rows = list(csv.DictReader(f))
    col = "attack" if "attack" in rows[0] else "kind"
    kinds = list(dict.fromkeys(r[col] for r in rows))
    mechs = list(dict.fromkeys(r["mechanism"] for r in rows))
    rej, n = defaultdict(int), defaultdict(int)
    for r in rows:
        rej[r[col], r["mechanism"]] += int(r["rejected"])
        n[r[col], r["mechanism"]] += 1
    grid = [[rej[k, m] / n[k, m] for m in mechs] for k in kinds]

    fig, ax = plt.subplots(figsize=(1.1 * len(mechs) + 2.2, 0.45 * len(kinds) + 1.2))
    ax.imshow(grid, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(mechs)), mechs)
    ax.set_yticks(range(len(kinds)), kinds)
    ax.tick_params(length=0)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.6 else INK)
    ax.set_title("частка відхилених блоків (рядок honest — FRR)", color=INK, fontsize=10, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    out = os.path.join(run_dir, "matrix.png")
    fig.savefig(out)
    return out


def e1(run_dir: str) -> str:
    """Розподіл відносної відстані B за типом переходу; вертикальна лінія — поріг α = 0,3."""
    with open(os.path.join(run_dir, "transitions.csv")) as f:
        rows = list(csv.DictReader(f))
    ks = sorted({int(r["k"]) for r in rows})
    kinds = ["honest", "lazy", "partial", "spoof_tiny", "spoof_tolerance", "plagiarism"]
    floor = 1e-7
    fig, axes = plt.subplots(1, len(ks), figsize=(3.3 * len(ks), 3.4), sharey=True)
    for ax, k in zip(axes, ks):
        for i, kind in enumerate(kinds):
            for dev, dy, mk in (("cpu", -0.15, "o"), ("mps", 0.15, "^")):
                xs = [max(float(r["ratio"]), floor) for r in rows
                      if int(r["k"]) == k and r["kind"] == kind and r["device"] == dev]
                ax.scatter(xs, [i + dy] * len(xs), s=10, color=BLUE if dev == "cpu" else ORANGE,
                           marker=mk, alpha=0.6, linewidths=0, label=dev if i == 0 else None)
        ax.axvline(0.3, color=INK, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(floor / 2, 10)
        ax.set_title(f"k = {k}", color=INK, fontsize=10)
        ax.set_xlabel("‖W_перерах − W_заявл‖ / ‖ΔW‖")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(kinds)), kinds)
    axes[0].legend(title="валідатор", frameon=False, loc="upper left", fontsize=8, title_fontsize=8)
    fig.text(0.01, 0.01, f"значення 0 (побітовий збіг) показано як {floor:g}", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(run_dir, "ratio_by_kind.png")
    fig.savefig(out)
    return out


def e3(run_dir: str) -> str:
    """Точність після останнього раунду і частка виграних атакувальниками раундів від частки атакувальників."""
    with open(os.path.join(run_dir, "final.csv")) as f:
        rows = list(csv.DictReader(f))
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    outs = []
    for metric, label, fname in (("final_accuracy", "точність після 10 раундів", "e3_accuracy.png"),
                                 ("attacker_win_share", "частка раундів, виграних атакувальниками",
                                  "e3_wins.png")):
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for i, sc in enumerate(scenarios):
            by_frac = defaultdict(list)
            for r in rows:
                if r["scenario"] == sc:
                    by_frac[float(r["fraction"])].append(float(r[metric]))
            xs = sorted(by_frac)
            ys = [sum(by_frac[x]) / len(by_frac[x]) for x in xs]
            err = [_std(by_frac[x]) for x in xs]
            ax.errorbar([x * 100 for x in xs], ys, yerr=err if any(err) else None, color=SERIES[i % 7], lw=2,
                        ls=LINESTYLES[i % 7], marker=MARKERS[i % 7], ms=5, capsize=3, label=sc)
        ax.set_xlabel("частка зловмисних вузлів, %")
        ax.set_ylabel(label)
        ax.set_ylim(-0.03, 1.03)
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        out = os.path.join(run_dir, fname)
        fig.savefig(out)
        outs.append(out)
    return ", ".join(outs)


def pareto(run_dir: str) -> str:
    """Механізм як точка: середня частка виявлених атак проти вартості перевірки (частка від K)."""
    with open(os.path.join(run_dir, "matrix.csv")) as f:
        rows = list(csv.DictReader(f))
    mechs = list(dict.fromkeys(r["mechanism"] for r in rows))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, m in enumerate(mechs):
        mr = [r for r in rows if r["mechanism"] == m]
        att = [int(r["rejected"]) for r in mr if r["kind"] != "honest"]
        frr = [int(r["rejected"]) for r in mr if r["kind"] == "honest"]
        cost = sum(float(r["verify_steps"]) / float(r["K"]) for r in mr) / len(mr)
        det = sum(att) / len(att)
        ax.scatter([cost], [det], s=60, color=SERIES[i % 6], marker=MARKERS[i % 6], zorder=3)
        ax.annotate(f"{m}  (FRR {sum(frr) / len(frr):.2f})", (cost, det), xytext=(7, [-12, 8, -12, 8, -12, -16][i % 6]),
                    textcoords="offset points", color=INK, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("вартість перевірки, частка від K (лог. шкала)")
    ax.set_ylabel("середня частка виявлених атак")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out = os.path.join(run_dir, "pareto.png")
    fig.savefig(out)
    return out


def e5(run_dir: str) -> str:
    """Точність спільної моделі за раундами для кожного α і механізму; пунктир — мережа без атакувальників."""
    with open(os.path.join(run_dir, "rounds.csv")) as f:
        rows = list(csv.DictReader(f))
    groups = list(dict.fromkeys((r["mechanism"], r["alpha"]) for r in rows))
    mechs = [m for m in dict.fromkeys(g[0] for g in groups) if m != "none"]
    fig, axes = plt.subplots(1, len(mechs), figsize=(4.2 * len(mechs), 3.6), sharey=True)
    axes = axes if len(mechs) > 1 else [axes]
    base = _mean_by_round([r for r in rows if r["mechanism"] == "none"])
    for ax, m in zip(axes, mechs):
        if base:
            ax.plot(range(1, len(base) + 1), base, color=INK2, lw=1.5, ls="--", label="без атак")
        for i, (gm, a) in enumerate(g for g in groups if g[0] == m):
            ys = _mean_by_round([r for r in rows if r["mechanism"] == gm and r["alpha"] == a])
            ax.plot(range(1, len(ys) + 1), ys, color=SERIES[i % 5], lw=2, marker=MARKERS[i % 5], ms=3,
                    label=f"α = {float(a):g}")
        ax.set_title(f"механізм {m}", color=INK, fontsize=10)
        ax.set_xlabel("раунд")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("точність (тестова половина)")
    fig.tight_layout()
    out = os.path.join(run_dir, "e5_accuracy.png")
    fig.savefig(out)
    return out


if __name__ == "__main__":
    kind, path = sys.argv[1], sys.argv[2]
    print({"accuracy": accuracy, "matrix": matrix, "e1": e1, "e3": e3, "pareto": pareto, "e5": e5}[kind](path))
