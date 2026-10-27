"""Графіки для розділу 4 з CSV/JSON, які пишуть sim.simulator і sim.matrix.

python -m sim.plots accuracy experiments/runs/kt1_lazy
python -m sim.plots matrix experiments/runs/e2
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
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150,
})


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
    kinds = list(dict.fromkeys(r["kind"] for r in rows))
    mechs = list(dict.fromkeys(r["mechanism"] for r in rows))
    rej, n = defaultdict(int), defaultdict(int)
    for r in rows:
        rej[r["kind"], r["mechanism"]] += int(r["rejected"])
        n[r["kind"], r["mechanism"]] += 1
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


if __name__ == "__main__":
    kind, path = sys.argv[1], sys.argv[2]
    print({"accuracy": accuracy, "matrix": matrix}[kind](path))
