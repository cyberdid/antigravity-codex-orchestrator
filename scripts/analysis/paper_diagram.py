"""Figure 1: interface-only test or checklist review and separate scoring.

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
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=9.0, weight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=8.0, color=MUTED, linespacing=1.25)


def arrow(ax, a, b, label="", color=INK, lw=1.1, ls="-", label_xy=None, ha="center", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        x, y = label_xy or ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 1.4)
        ax.text(x, y, label, ha=ha, va="bottom", fontsize=8.0, color=color if color != INK else MUTED, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 57)
    ax.axis("off")

    # Trust regions: the dashed line is the only boundary any data crosses.
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 85, 56, boxstyle="round,pad=0,rounding_size=1.5", fc=FILL, ec="none"))
    ax.text(2, 55, "LOCAL: implementation kept on device", fontsize=8.4, color=MUTED, va="top")
    ax.plot([87, 87], [1, 56], color=ACCENT, lw=1, ls=(0, (4, 3)))
    ax.text(91, 55, "CLOUD", fontsize=8.4, color=MUTED, va="top")

    # Components on a 3-column x 2-row grid (+ cloud column).
    box(ax, 2, 34, 25, 16, "Task workspace", "buggy code, docstrings\nand visible tests")
    box(ax, 33, 34, 40, 16, "Interface extractor", "signatures, docstrings, type fields;\nprivacy check; no function bodies")
    box(ax, 91, 32, 27, 18, "Cloud reviewer", "writes contract tests\nOR a rule checklist", accent=True)
    box(ax, 62, 6, 23, 18, "Review intake", "static test validation\nOR append checklist")
    box(ax, 32, 6, 25, 18, "Local builder", "Gemma edits code;\nruns available checks")
    box(ax, 2, 6, 25, 18, "Post-run scoring", "original visible tests\n+ hidden evaluation")

    # Flows (orange = the only traffic across the boundary).
    arrow(ax, (27, 42), (33, 42))
    arrow(ax, (73, 42), (91, 42), "interface", color=ACCENT, lw=1.6, label_xy=(82, 43.5))
    arrow(ax, (104, 32), (77, 24), color=ACCENT, lw=1.6)
    ax.text(107, 25, "tests /\nchecklist", ha="center", va="top", fontsize=8, color=ACCENT)
    arrow(ax, (62, 15), (57, 15))
    arrow(ax, (40, 24), (17, 34), "edits", label_xy=(25, 28))
    arrow(ax, (32, 15), (27, 15))
    ax.text(44, 2, "Hidden evaluation is never shown to either model.", ha="center", fontsize=8.2, color=INK)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_architecture.{ext}", dpi=300, bbox_inches="tight")
    print("written", OUT / "fig1_architecture.png")


if __name__ == "__main__":
    main()
