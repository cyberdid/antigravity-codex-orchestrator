"""Pre-registered analysis of Paper I campaigns A and B (docs/paper/PREREGISTRATION_CAMPAIGN_{A,B}.md).

Usage: python scripts/analysis/campaign_a.py RESULTS.json [ABLATION.json]
Unit of inference: task. Per-task verified rate = verified runs / seeds.
"""
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE, TESTS, LIST = "P1_STAGE_ADAPTIVE", "P1_STAGE_ADAPTIVE_SPECTESTS", "P1_STAGE_ADAPTIVE_CHECKLIST"


def load(path):
    runs = defaultdict(dict)  # (policy, task) -> {seed: task record}
    for pol, v in json.loads(Path(path).read_text())["policies"].items():
        for sr in v["seed_runs"]:
            for t in sr["tasks"]:
                runs[pol, t["task_id"]][sr["seed"]] = t
    return runs


def rate(runs, pol, task, key="pipeline_verified"):
    seeds = runs[pol, task]
    return sum(bool(t[key]) for t in seeds.values()) / len(seeds)


def silent(t):
    return bool(t["worker_reported_success"]) and not t["pipeline_verified"]


def sign_p(w, l):
    n = w + l
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n) if n else 1.0


def permutation_p(diffs):
    nz, obs = [d for d in diffs if d], sum(diffs)
    if len(nz) > 20:  # Monte Carlo beyond 2^20 sign patterns
        rng = random.Random(0)
        hits = sum(abs(sum(d if rng.random() < .5 else -d for d in nz)) >= abs(obs) - 1e-12 for _ in range(200000))
        return hits / 200000
    hits = sum(abs(sum(d if m >> i & 1 else -d for i, d in enumerate(nz))) >= abs(obs) - 1e-12 for m in range(2 ** len(nz)))
    return hits / 2 ** len(nz)


def bootstrap_ci(diffs, reps=20000):
    rng = random.Random(0)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(reps))
    return means[int(.025 * reps)], means[int(.975 * reps) - 1]


def compare(runs, tasks, a, b, label):
    diffs = [rate(runs, b, t) - rate(runs, a, t) for t in tasks]
    w, l = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    lo, hi = bootstrap_ci(diffs)
    print(f"{label}: tasks better/worse/equal {w}/{l}/{len(diffs) - w - l}; sign test p = {sign_p(w, l):.4f}; "
          f"permutation p = {permutation_p(diffs):.4f}; mean diff {sum(diffs) / len(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return w, l


def main(path, ablation=None):
    sys.path.insert(0, str(ROOT / "scripts" / "paper_tasks"))
    runs = load(path)
    if any(task.startswith("N") for _, task in runs):
        # Read the frozen construction record instead of re-executing buggy modules
        # (N7 can hang). This analysis needs category and hidden-failing-rule counts,
        # all of which were recorded before the repair campaign.
        specs = json.loads((ROOT / "scripts/paper_tasks/manifest.json").read_text())
        construction = json.loads((ROOT / "scripts/paper_tasks/natural_tasks.json").read_text())
        manifest = {}
        for record in construction["records"]:
            if "task_id" not in record:
                continue
            chosen = next(a for a in record["attempts"] if a["seed"] == record["generation_seed"])
            failing = set(chosen["failing_rules"])
            manifest[record["task_id"]] = {
                "category": specs[record["spec"]]["category"],
                "hidden_failing": len(failing - set(record["visible_rules"])),
            }
    else:
        manifest = json.loads((ROOT / "scripts/paper_tasks/manifest.json").read_text())
    tasks = sorted(manifest, key=lambda t: int(t.split("_")[0][1:]))
    print(f"tasks {len(tasks)}; arms {sorted({p for p, _ in runs})}")
    print("\nArm | verified runs | silent false successes | wall s (runs) | generation wall s (one-time) | cloud tokens")
    for pol in (BASE, TESTS, LIST):
        recs = [t for task in tasks for t in runs[pol, task].values()]
        if not recs:
            continue
        gen = {r["task_id"]: (r["metadata"].get("spec_tests_wall_s") or r["metadata"].get("spec_checklist_wall_s") or 0) for r in recs}
        tok = {r["task_id"]: (r["metadata"].get("spec_tests_tokens") or r["metadata"].get("spec_checklist_tokens") or 0) for r in recs}
        print(f"{pol} | {sum(r['pipeline_verified'] for r in recs)}/{len(recs)} | {sum(silent(r) for r in recs)}/{len(recs)} | "
              f"{sum(r['wall_clock_s'] for r in recs):.0f} | {sum(gen.values()):.0f} | {sum(tok.values())}")
    print("\nPRIMARY (H1): SPECTESTS vs baseline, task level")
    compare(runs, tasks, BASE, TESTS, "  verified")
    print("Secondary:")
    if any(runs[LIST, t] for t in tasks):
        compare(runs, tasks, LIST, TESTS, "  H3 SPECTESTS vs CHECKLIST")
        compare(runs, tasks, BASE, LIST, "  CHECKLIST vs baseline")
    d_sil = [sum(silent(r) for r in runs[TESTS, t].values()) / len(runs[TESTS, t]) -
             sum(silent(r) for r in runs[BASE, t].values()) / len(runs[BASE, t]) for t in tasks]
    w, l = sum(d < 0 for d in d_sil), sum(d > 0 for d in d_sil)
    print(f"  H2 silent false successes, tasks lower/higher with SPECTESTS {w}/{l}; sign test p = {sign_p(w, l):.4f}")
    for cat in ("single", "multi-file", "underspecified"):
        sub = [t for t in tasks if manifest[t]["category"] == cat]
        if sub:  # Campaign B has no under-specified tasks
            compare(runs, sub, BASE, TESTS, f"  category {cat} ({len(sub)} tasks)")
    harm = [t for t in tasks if manifest[t]["hidden_failing"] == 0 and rate(runs, TESTS, t) < rate(runs, BASE, t)]
    print(f"  harm on tasks without hidden failing rules: {harm or 'none'}")
    print("\nPer task (baseline / SPECTESTS / CHECKLIST verified rate; hidden failing rules):")
    for t in tasks:
        cells = " / ".join(f"{rate(runs, p, t):.2f}" if runs[p, t] else "-" for p in (BASE, TESTS, LIST))
        print(f"  {t:26s} {cells}   hidden_failing={manifest[t]['hidden_failing']} [{manifest[t]['category']}]")
    if ablation:
        abl = load(ablation)
        print("\nAblation (exploratory, seed 303): verified by disclosure level")
        for pol in ("P1_STAGE_ADAPTIVE_SPECTESTS_SIG", "P1_STAGE_ADAPTIVE_SPECTESTS_DOC", TESTS):
            recs = [t for task in tasks for t in abl[pol, task].values()] or [t for task in tasks for t in runs[pol, task].values() if t["seed"] == 303]
            if recs:
                print(f"  {pol}: {sum(r['pipeline_verified'] for r in recs)}/{len(recs)}, silent {sum(silent(r) for r in recs)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
