"""Post-hoc publication checks from saved results; no model calls or code execution.

Reports campaign cost including one generation per task, static-rejection frequency,
and pooled sensitivity to grouping A/B instances by their originating specification.
Usage: python3 -B scripts/analysis/publication.py > benchmarks/publication_analysis.json
"""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "P1_STAGE_ADAPTIVE"
TESTS = BASE + "_SPECTESTS"
CHECKLIST = BASE + "_CHECKLIST"
FILES = ("spec_contract_confirm_fresh", "spec_contract_confirm_all", "campaign_a", "campaign_b")


def sign_p(wins, losses):
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n) if n else 1.0


def rows(data, arm):
    return [r for seed in data["policies"][arm]["seed_runs"] for r in seed["tasks"]]


def silent(row):
    return bool(row["worker_reported_success"]) and not row["pipeline_verified"]


def campaign_cost(data):
    result = {}
    baseline = sum(r["wall_clock_s"] for r in rows(data, BASE))
    for arm in (BASE, TESTS, CHECKLIST):
        records = rows(data, arm)
        generation = {}
        for r in records:
            meta = r["metadata"]
            seconds = meta.get("spec_tests_wall_s") or meta.get("spec_checklist_wall_s") or 0
            task = r["task_id"]
            if task in generation and generation[task] != seconds:
                raise ValueError(f"Inconsistent cached generation time for {task}: {arm}")
            generation[task] = seconds
        wall = sum(r["wall_clock_s"] for r in records)
        total = wall + sum(generation.values())
        result[arm] = {"runs": len(records), "run_wall_s": wall,
                       "one_time_generation_s": sum(generation.values()), "campaign_total_s": total,
                       "campaign_overhead_pct": 100 * (total / baseline - 1),
                       "generation_reused_across_seeds": True}
    return result


def grouped_sensitivity(datasets):
    groups = defaultdict(lambda: {BASE: [], TESTS: []})
    instances = set()
    for name, data in datasets.items():
        for arm in (BASE, TESTS):
            for row in rows(data, arm):
                identifier = row["task_id"].split("_")[0]
                earlier = name.startswith("spec_contract")
                group = identifier if earlier else "P" + identifier[1:]
                groups[group][arm].append(row)
                instances.add(("earlier" if earlier else name, identifier))
    result = {"task_instances": len(instances), "originating_specification_groups": len(groups),
              "method": "Equal weight per originating specification; within-group rate pools all its runs. "
                        "A Pk and B Nk share a group. H tasks retain separate groups. Exact two-sided sign test.",
              "interpretation": "Exploratory sensitivity, not a pre-registered test or a guarantee of group independence."}
    for label, outcome, direction in (("verified", lambda r: bool(r["pipeline_verified"]), 1),
                                       ("silent", silent, -1)):
        differences = {}
        for group, arms in groups.items():
            if len(arms[BASE]) != len(arms[TESTS]):
                raise ValueError(f"Unmatched group {group}")
            rates = {a: sum(outcome(r) for r in records) / len(records) for a, records in arms.items()}
            differences[group] = direction * (rates[TESTS] - rates[BASE])
        better = sum(d > 0 for d in differences.values())
        worse = sum(d < 0 for d in differences.values())
        result[label] = {"improved": better, "worsened": worse, "tied": len(groups) - better - worse,
                         "two_sided_sign_p": sign_p(better, worse)}
    return result


def main():
    datasets = {name: json.loads((ROOT / "benchmarks" / (name + ".json")).read_text()) for name in FILES}
    grading = json.loads((ROOT / "benchmarks/campaign_a_test_grading.json").read_text())
    report = {"analysis_status": "post hoc, added during revision on 2026-10-21",
              "inputs_sha256": {name + ".json": hashlib.sha256((ROOT / "benchmarks" / (name + ".json")).read_bytes()).hexdigest()
                                for name in FILES},
              "campaign_a_cost": campaign_cost(datasets["campaign_a"]),
              "campaign_b_cost": campaign_cost(datasets["campaign_b"]),
              "campaign_a_contract_installation": {"attempted_task_modules": 25, "installed_task_modules": len(grading),
                                                   "tasks_without_installed_module": 25 - len(grading),
                                                   "task_rejection_fraction": (25 - len(grading)) / 25,
                                                   "note": "P18 rejected on both generation attempts; frequency is per task, not per reply. "
                                                           "A rejected reply is retained as an intention-to-treat outcome."},
              "originating_specification_sensitivity": grouped_sensitivity(datasets)}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
