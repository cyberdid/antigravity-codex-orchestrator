"""Figure 0 of docs/paper/manuscript.md: information flow of the contract-test mechanism.

Usage: python3 scripts/analysis/paper_diagram.py -> docs/paper/figures/fig1_architecture.{pdf,png}
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs/paper/figures"
INK, MUTED, ACCENT, FILL = "#0b0b0b", "#52514e", "#eb6834", "#f3f2ef"
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "savefig.facecolor": "white"})


def box(ax, x, y, w, h, title, body="", accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc="white" if not accent else "#fdf0ea", ec=ACCENT if accent else INK, lw=1))
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=7.6, weight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=6.6, color=MUTED, linespacing=1.25)


def arrow(ax, a, b, label="", color=INK, lw=1.1, ls="-", label_xy=None, ha="center", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        x, y = label_xy or ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 1.4)
        ax.text(x, y, label, ha=ha, va="bottom", fontsize=7, color=color if color != INK else MUTED, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 52)
    ax.axis("off")

    # Trust regions: the dashed line is the only boundary any data crosses.
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 84.5, 51, boxstyle="round,pad=0,rounding_size=1.5", fc=FILL, ec="none"))
    ax.text(2, 50, "local machine (source code never leaves)", fontsize=7.2, color=MUTED, va="top")
    ax.plot([88, 88], [1, 51], color=ACCENT, lw=1, ls=(0, (4, 3)))
    ax.text(91, 50, "cloud", fontsize=7.2, color=MUTED, va="top")

    # Components on a 3-column x 2-row grid (+ cloud column).
    box(ax, 2, 28, 22, 16, "Task workspace", "buggy module\nwith docstrings,\nvisible tests")
    box(ax, 33, 28, 22, 16, "Interface extractor", "signatures, docstrings,\ntype fields (AST);\nno function bodies")
    box(ax, 93, 24, 25, 20, "Contract-test writer", "stronger model writes\none unittest module\nper task from the\ninterface alone", accent=True)
    box(ax, 63, 6, 20, 16, "Static validator", "allowed imports\nand calls only;\nTestCase present")
    box(ax, 33, 6, 22, 16, "Local agent", "gemma4:12b edits code\nuntil visible and\ncontract tests pass")
    box(ax, 2, 6, 22, 16, "Independent check", "visible tests +\nhidden holdout,\nshown to neither model")

    # Flows (orange = the only traffic across the boundary).
    arrow(ax, (24, 36), (33, 36), "reads", label_xy=(28.5, 37))
    arrow(ax, (55, 36), (93, 36), "interface only", color=ACCENT, lw=1.6, label_xy=(74, 37.2))
    arrow(ax, (100, 24), (83, 15), "test module", color=ACCENT, lw=1.6, label_xy=(97, 16.5), ha="left")
    arrow(ax, (63, 14), (55, 14), "installs", label_xy=(59, 15))
    arrow(ax, (40, 22), (17, 28), "edits", rad=0.0, label_xy=(26.5, 25.6))
    arrow(ax, (33, 11), (24, 11), "run ends", label_xy=(28.5, 12))
    ax.text(13, 2.2, "verified  /  silent false success", ha="center", fontsize=6.8, color=INK)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_architecture.{ext}", dpi=300, bbox_inches="tight")
    print("written", OUT / "fig1_architecture.png")


if __name__ == "__main__":
    main()
