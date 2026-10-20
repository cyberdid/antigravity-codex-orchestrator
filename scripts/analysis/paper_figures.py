"""Figures for docs/paper/manuscript.md, computed from the benchmark files.

Usage: python scripts/analysis/paper_figures.py  -> docs/paper/figures/fig{2,3,4}_*.{pdf,png}
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/paper/figures"
BASE, SPEC = "#2a78d6", "#eb6834"          # validated categorical slots 1-2 (light)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white", "figure.facecolor": "white", "pdf.fonttype": 42})


def cells(name):
    out = {}
    for pol, v in json.loads((ROOT / "benchmarks" / name).read_text())["policies"].items():
        for sr in v["seed_runs"]:
            for t in sr["tasks"]:
                arm = "S" if pol.endswith("SPECTESTS") else "B"
                out[arm, t["task_id"].split("_")[0], sr["seed"]] = t
    return out


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1(r4, r5):
    tasks = [f"H{i}" for i in range(1, 20)]
    data = {**r5, **r4}
    rate = lambda arm, task: sum(data[arm, task, s]["pipeline_verified"] for s in (303, 404, 505)) / 3
    fig, ax = plt.subplots(figsize=(3.4, 4.6))
    for y, task in enumerate(reversed(tasks)):
        b, s = rate("B", task), rate("S", task)
        if b != s:
            ax.plot([b, s], [y, y], color=MUTED, lw=1, zorder=1)
        ax.scatter(b, y + (0.17 if b == s else 0), s=36, facecolor="white", edgecolor=BASE, lw=1.6, marker="o", zorder=3)
        ax.scatter(s, y - (0.17 if b == s else 0), s=30, color=SPEC, marker="s", zorder=3, edgecolor="white", lw=0.8)
    ax.axhline(5.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(1.04, 2.5, "fresh tasks\n(R4)", fontsize=7.5, color=MUTED, va="center", clip_on=False)
    ax.text(1.04, 12, "earlier tasks\n(R5)", fontsize=7.5, color=MUTED, va="center", clip_on=False)
    ax.set_yticks(range(19), list(reversed(tasks)))
    ax.set_xticks([0, 1 / 3, 2 / 3, 1], ["0", "1/3", "2/3", "1"])
    ax.set_xlim(-0.08, 1.08)
    ax.set_xlabel("verified completion (3 seeds per task)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=36, facecolor="white", edgecolor=BASE, lw=1.6, marker="o", label="baseline")
    ax.scatter([], [], s=30, color=SPEC, marker="s", label="contract tests (SPECTESTS)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.5, handletextpad=0.3)
    save(fig, "fig2_per_task")


def fig2():
    r = json.loads((ROOT / "benchmarks/oracle_quality.json").read_text())
    v2 = ("spec_contract_v2", "spec_contract_confirm_all", "spec_contract_confirm_fresh")
    hk = vk = ck = cn = 0
    for c in r["mutation"].values():
        for m in c["mutants"]:
            if m["holdout"]:
                hk += 1
                vk += m["visible"]
                for tag, killed in m["contract"].items():
                    if tag.startswith(v2):
                        cn += 1
                        ck += killed
    fig, ax = plt.subplots(figsize=(3.4, 1.5))
    rows = [("task's visible tests", vk / hk, f"{vk}/{hk}"), ("contract tests (v2)", ck / cn, f"{ck}/{cn}")]
    for y, (label, value, frac) in enumerate(reversed(rows)):
        ax.barh(y, value, height=0.55, color=BASE if "visible" in label else SPEC,
                hatch=None if "visible" in label else "////", edgecolor="white", lw=0)
        ax.text(value + 0.02, y, f"{value:.1%}  ({frac})", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(2), [r[0] for r in reversed(rows)])
    ax.set_xlim(0, 1.35)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_xlabel("holdout-killed mutants also killed")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "fig4_mutation")


def fig3(r4, r5):
    def silent(c, arm):
        rows = [t for (a, _, _), t in c.items() if a == arm]
        return sum(t["worker_reported_success"] and not t["pipeline_verified"] for t in rows), len(rows)
    groups = [("R4 fresh\n(6 tasks)", r4), ("R5 earlier\n(13 tasks)", r5), ("pooled\n(19 tasks)", {**r4, **r5})]
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    w = 0.36
    for x, (label, c) in enumerate(groups):
        for dx, arm, color, hatch in ((-w / 2, "B", BASE, None), (w / 2, "S", SPEC, "////")):
            k, n = silent(c, arm)
            ax.bar(x + dx, k / n, width=w - 0.03, color=color, hatch=hatch, edgecolor="white", lw=0)
            ax.text(x + dx, k / n + 0.02, f"{k}/{n}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(3), [g[0] for g in groups])
    ax.set_ylim(0, 0.95)
    ax.set_yticks([0, 0.25, 0.5, 0.75], ["0%", "25%", "50%", "75%"])
    ax.set_ylabel("silent false successes")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=BASE, label="baseline"),
                       Patch(facecolor=SPEC, hatch="////", edgecolor="white", label="contract tests")],
              frameon=False, fontsize=7.5, loc="upper right")
    save(fig, "fig3_silent")


if __name__ == "__main__":
    r4, r5 = cells("spec_contract_confirm_fresh.json"), cells("spec_contract_confirm_all.json")
    fig1(r4, r5)
    fig2()
    fig3(r4, r5)
    print("written:", sorted(p.name for p in OUT.iterdir()))
