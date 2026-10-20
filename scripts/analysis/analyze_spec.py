"""Summarise a SPEC/SPECTESTS benchmark JSON per policy, task and seed (paired sign test)."""
import json
import math
import sys
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def sign_test(w, l):
    n = w + l
    if not n:
        return 1.0
    return min(1.0, sum(math.comb(n, k) for k in range(0, min(w, l) + 1)) * 2 / 2 ** n)


def main(path):
    d = json.load(open(path))
    print({k: d.get(k) for k in ("git_commit", "git_dirty", "attempts", "cloud_backend", "timestamp_utc", "resumed_from")})
    policies = list(d["policies"])
    cell = {}
    for pol, v in d["policies"].items():
        for sr in v["seed_runs"]:
            for t in sr["tasks"]:
                cell[(pol, t["task_id"].split("_")[0], sr["seed"])] = t
    tasks = sorted({k[1] for k in cell}, key=lambda x: int(x[1:]))
    print("\nPolicy | verified | silent false successes | wall s | worker tokens | cloud tokens")
    for pol in policies:
        rows = [t for (p, _, _), t in cell.items() if p == pol]
        ver = sum(t["pipeline_verified"] for t in rows)
        silent = sum(t["worker_reported_success"] and not t["pipeline_verified"] for t in rows)
        wall = sum(t["wall_clock_s"] for t in rows)
        tok = sum(t["metadata"]["input_tokens"] + t["metadata"]["output_tokens"] for t in rows)
        spec_tok = {t["task_id"]: t["metadata"].get("spec_tests_tokens", 0) for t in rows}
        cloud = sum(spec_tok.values()) + sum(t["metadata"].get("cloud_assist_tokens", 0) for t in rows)
        lo, hi = wilson(ver, len(rows))
        print(f"{pol} | {ver}/{len(rows)} (95% CI {lo:.2f}-{hi:.2f}) | {silent} | {wall:.0f} | {tok} | {cloud}")
    print("\nPer task (✓ verified, v visible only, ✗ neither):")
    for task in tasks:
        parts = []
        for pol in policies:
            seeds = {s: t for (p, tk, s), t in cell.items() if p == pol and tk == task}
            marks = "".join("✓" if seeds[s]["pipeline_verified"] else ("v" if seeds[s]["metadata"]["visible_tests_passed"] else "✗")
                            for s in sorted(seeds))
            parts.append(f"{sum(t['pipeline_verified'] for t in seeds.values())}/{len(seeds)} {marks}")
        print(task + " | " + " | ".join(parts))
    base = policies[0]
    for pol in policies[1:]:
        w = l = ties = 0
        for (p, task, s), t in cell.items():
            if p != pol or (base, task, s) not in cell:
                continue
            b = cell[(base, task, s)]["pipeline_verified"]
            if t["pipeline_verified"] and not b:
                w += 1
            elif b and not t["pipeline_verified"]:
                l += 1
            else:
                ties += 1
        print(f"\n{pol} vs {base}: wins {w}, losses {l}, ties {ties}, sign-test p={sign_test(w, l):.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
