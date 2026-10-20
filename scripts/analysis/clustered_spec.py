"""Task-level (clustered) analysis of baseline vs SPECTESTS benchmark files.

Seeds of one task are not independent, so the unit of inference is the task:
per-task difference in verified rate, task-level sign test, exact sign-flip
permutation test and a cluster bootstrap CI.
Usage: python scripts/analysis/clustered_spec.py FILE [FILE ...]
"""
import json
import math
import random
import sys


def load(path):
    cell = {}
    for pol, v in json.load(open(path))["policies"].items():
        for sr in v["seed_runs"]:
            for t in sr["tasks"]:
                cell[("S" if pol.endswith("SPECTESTS") else "B", t["task_id"].split("_")[0], sr["seed"])] = t["pipeline_verified"]
    return cell


def analyse(cell, label):
    tasks = sorted({k[1] for k in cell}, key=lambda x: int(x[1:]))
    diffs = {}
    for task in tasks:
        seeds = sorted(s for (arm, t, s) in cell if arm == "B" and t == task)
        diffs[task] = (sum(cell["S", task, s] for s in seeds) - sum(cell["B", task, s] for s in seeds)) / len(seeds)
    vals = list(diffs.values())
    w, l = sum(v > 0 for v in vals), sum(v < 0 for v in vals)
    sign_p = min(1.0, 2 * sum(math.comb(w + l, i) for i in range(min(w, l) + 1)) / 2 ** (w + l)) if w + l else 1.0
    nz, obs = [v for v in vals if v], sum(vals)
    hits = sum(abs(sum(v if m >> i & 1 else -v for i, v in enumerate(nz))) >= abs(obs) - 1e-12 for m in range(2 ** len(nz)))
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(20000))
    print(f"{label}: {len(tasks)} tasks, per-task difference {{{', '.join(f'{t}: {v:+.2f}' for t, v in diffs.items())}}}")
    print(f"  task-level sign test {w} better / {l} worse: p = {sign_p:.4f}; sign-flip permutation p = {hits / 2 ** len(nz):.4f}; "
          f"mean difference {obs / len(vals):+.3f}, cluster bootstrap 95% CI [{boots[500]:+.3f}, {boots[19499]:+.3f}]")


if __name__ == "__main__":
    merged = {}
    for path in sys.argv[1:]:
        cell = load(path)
        analyse(cell, path)
        merged.update(cell)
    if len(sys.argv) > 2:
        analyse(merged, "pooled")
