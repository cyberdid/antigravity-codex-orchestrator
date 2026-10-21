"""Pooled evidence across all runs with the corrected measurement (post hoc; Paper I §6.10).

Earlier tasks (R4 fresh H14-H19, R5 H1-H13), Campaign A (P1-P25) and Campaign B (natural bugs),
baseline vs the cloud arm(s):
run-level paired counts and task-level sign tests for silent false successes and failures.
Usage: python scripts/analysis/pooled.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (303, 404, 505)


def sign_p(w, l):
    n = w + l
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n) if n else 1.0


def cells(name, arms, tag):
    out = {}
    for pol, v in json.loads((ROOT / "benchmarks" / name).read_text())["policies"].items():
        for sr in v["seed_runs"]:
            for t in sr["tasks"]:
                if pol in arms:
                    out[(arms[pol], tag + t["task_id"].split("_")[0], sr["seed"])] = t
    return out


def silent(t):
    return bool(t["worker_reported_success"]) and not t["pipeline_verified"]


def failed(t):
    return not t["pipeline_verified"]


def run_level(c, arm, bad):
    pairs = [(t, s) for (a, t, s) in c if a == arm and ("B", t, s) in c]
    fixes = sum(bad(c["B", t, s]) and not bad(c[arm, t, s]) for t, s in pairs)
    breaks = sum(bad(c[arm, t, s]) and not bad(c["B", t, s]) for t, s in pairs)
    return len(pairs), fixes, breaks, sign_p(fixes, breaks)


def task_level(c, arm, bad):
    tasks = sorted({t for (a, t, _) in c if a == arm})
    better = sum(sum(bad(c[arm, t, s]) for s in SEEDS) < sum(bad(c["B", t, s]) for s in SEEDS) for t in tasks)
    worse = sum(sum(bad(c[arm, t, s]) for s in SEEDS) > sum(bad(c["B", t, s]) for s in SEEDS) for t in tasks)
    return len(tasks), better, worse, sign_p(better, worse)


def totals(c, arm, key):
    rows = [v for (a, _, _), v in c.items() if a == arm]
    return sum(key(v) for v in rows), len(rows)


def main():
    a_arms = {"P1_STAGE_ADAPTIVE": "B", "P1_STAGE_ADAPTIVE_SPECTESTS": "S", "P1_STAGE_ADAPTIVE_CHECKLIST": "C"}
    e_arms = {"P1_STAGE_ADAPTIVE": "B", "P1_STAGE_ADAPTIVE_SPECTESTS": "S"}
    campaign_a = cells("campaign_a.json", a_arms, "A")
    earlier = {**cells("spec_contract_confirm_fresh.json", e_arms, "E"), **cells("spec_contract_confirm_all.json", e_arms, "E")}
    print("Campaign A, run level (pairs, runs the arm fixes, runs it breaks, sign-test p):")
    for arm in ("S", "C"):
        print(f"  {arm} silent false successes {run_level(campaign_a, arm, silent)}; failures {run_level(campaign_a, arm, failed)}")
    campaign_b = cells("campaign_b.json", a_arms, "N")
    pooled = {k: v for k, v in (campaign_a | campaign_b).items() if k[0] in ("B", "S")} | earlier
    print("Pooled earlier + Campaign A + Campaign B, contract tests vs baseline (post hoc):")
    print(f"  runs: silent {totals(pooled, 'B', silent)} -> {totals(pooled, 'S', silent)}; "
          f"verified {totals(pooled, 'B', lambda t: t['pipeline_verified'])} -> {totals(pooled, 'S', lambda t: t['pipeline_verified'])}")
    print(f"  task level (tasks, improved, worsened, p): silent {task_level(pooled, 'S', silent)}; failures {task_level(pooled, 'S', failed)}")


if __name__ == "__main__":
    main()
