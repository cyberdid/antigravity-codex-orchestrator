#!/usr/bin/env python3
"""Harness² adaptation experiment: H0 vs H2 (standard) vs H3 (verifier-gated).

Adaptation split H1–H4: each arm walks the tasks in order; on every task one
Propose–Probe–Compose step (k=1, INV + DOM candidates, one composition) runs from
the arm's current harness, and the arm's promotion rule picks the next harness.
Held-out split H5–H10: the frozen final harness of each arm (and the default one,
H0) is evaluated once per seed. Worker policy is fixed for every rollout:
gemma4:12b, think=false, num_ctx 16384, num_predict 2048, P1_STAGE stage hints,
adaptive generation limit, single attempt, 300 s deadline.

The harness editor is the cloud architect (agy backend by default). It sees the
agent's own trajectory, visible test output and final diff of the synthetic
benchmark tasks; hidden holdout results are only used by the H3 gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark_gemma_orion_phase2b as bench  # noqa: E402

from antigravity_hybrid_orchestrator.harness.evidence import Rollout  # noqa: E402
from antigravity_hybrid_orchestrator.harness.proposer import HarnessProposer  # noqa: E402
from antigravity_hybrid_orchestrator.harness.recursion import (  # noqa: E402
    TaskStep, gated_choice, lesson, run_task_step, standard_choice, utility,
)
from antigravity_hybrid_orchestrator.harness.tree import DEFAULT_TREE, HarnessTree  # noqa: E402

POLICY = "P1_STAGE_ADAPTIVE"
ADAPTATION = ["H1", "H2", "H3", "H4"]
HELD_OUT = ["H5", "H6", "H7", "H8", "H9", "H10"]


def _task(prefix: str):
    return next(t for t in bench.TASKS if t[0].startswith(prefix + "_"))


def _apply_policy_env() -> None:
    for key, value in bench.POLICIES[POLICY].items():
        if key.startswith("OLLAMA_"):
            os.environ[key] = str(value)


class Runner:
    def __init__(self, out_dir: Path, seed: int, log) -> None:
        self.out_dir = out_dir
        self.seed = seed
        self.log = log
        self.cache: Dict[str, Rollout] = {}
        self.records: List[Dict[str, Any]] = []

    def rollout(self, prefix: str, tree: HarnessTree, label: str, seed: int | None = None) -> Rollout:
        seed = self.seed if seed is None else seed
        key = f"{prefix}:{seed}:{tree.fingerprint()}"
        if key in self.cache:
            self.log(f"    [{label}] {prefix} harness {tree.fingerprint()[:10]} -> cached")
            return self.cache[key]
        os.environ["OLLAMA_SEED"] = str(seed)
        task_id, setup_fn, holdout_fn = _task(prefix)
        record = bench.run_single_task(POLICY, seed, task_id, setup_fn, holdout_fn, harness=tree)
        r = Rollout.from_benchmark(label, tree.fingerprint(), record)
        self.cache[key] = r
        self.records.append({"label": label, "seed": seed, "harness": tree.fingerprint(),
                             "record": {k: v for k, v in record.items() if k != "metadata"},
                             "metadata": {k: record["metadata"].get(k) for k in (
                                 "visible_tests_passed", "holdout_tests_passed", "input_tokens",
                                 "output_tokens", "truncated_generations", "budget_escalations",
                                 "discarded_truncated_tool_calls", "turn_trajectory", "workspace_diff",
                                 "visible_test_output")}})
        self.log(f"    [{label}] {prefix} harness {tree.fingerprint()[:10]} -> gate={r.gate_accepted} "
                 f"visible={r.visible_passed} holdout={r.holdout_passed} wall={r.wall_s:.0f}s "
                 f"trunc={r.truncations} U={utility(r):.1f}")
        return r


def _step_record(step: TaskStep) -> Dict[str, Any]:
    return {
        "task_id": step.task_id,
        "base_harness": step.base_tree.fingerprint(),
        "base": step.base.summary(),
        "proposals": {
            label: {"dim": p.dim, "kept_base": p.kept_base, "fell_back": p.fell_back, "error": p.error,
                    "notes": p.notes, "harness": p.tree.fingerprint(), "tree": p.tree.to_dict(),
                    "input_tokens": p.input_tokens, "output_tokens": p.output_tokens, "calls": p.calls}
            for label, p in step.proposals.items()
        },
        "rollouts": {label: r.summary() for label, r in step.rollouts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=303, help="Seed for adaptation rollouts")
    parser.add_argument("--eval-seeds", default="303,404", help="Seeds for held-out evaluation")
    parser.add_argument("--cloud-backend", choices=["agy", "haiku", "luna"], default="agy")
    parser.add_argument("--output", default=str(REPO_ROOT / "benchmarks" / "harness2_experiment.json"))
    args = parser.parse_args()

    out_path = Path(args.output)
    out_dir = out_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = (out_dir / "progress.log").open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    from antigravity_hybrid_orchestrator.architect.client import CloudArchitectClient

    cloud = CloudArchitectClient(backend=args.cloud_backend)
    cloud_calls: List[Dict[str, Any]] = []
    proposer = HarnessProposer(cloud.complete, audit=cloud_calls.append)
    _apply_policy_env()
    runner = Runner(out_dir, args.seed, log)

    git = lambda *a: subprocess.check_output(["git", *a], cwd=str(REPO_ROOT), text=True).strip()  # noqa: E731
    results: Dict[str, Any] = {
        "experiment": "Harness² adaptation: H0 vs H2 (standard) vs H3 (verifier-gated)",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "worker_policy": POLICY, "adaptation_seed": args.seed, "cloud_backend": args.cloud_backend,
        "adaptation_tasks": ADAPTATION, "held_out_tasks": HELD_OUT,
        "arms": {}, "steps": [], "held_out": {},
    }

    def save() -> None:
        results["cloud_calls"] = [{k: v for k, v in c.items() if k != "reply"} for c in cloud_calls]
        results["cloud_tokens"] = sum(c.get("input_tokens", 0) + c.get("output_tokens", 0) for c in cloud_calls)
        results["rollouts"] = runner.records
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "cloud_replies.json").write_text(json.dumps(cloud_calls, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")

    arms = {"H2": {"tree": DEFAULT_TREE, "library": [], "decisions": []},
            "H3": {"tree": DEFAULT_TREE, "library": [], "decisions": []}}
    step_cache: Dict[str, TaskStep] = {}
    for prefix in ADAPTATION:
        for arm_name, arm in arms.items():
            key = f"{prefix}:{arm['tree'].fingerprint()}:{len(arm['library'])}"
            log(f"== adaptation {prefix} | arm {arm_name} | base {arm['tree'].fingerprint()[:10]}")
            step = step_cache.get(key)
            if step is None:
                step = run_task_step(prefix, arm["tree"], proposer=proposer,
                                     rollout=lambda tree, label, p=prefix: runner.rollout(p, tree, label),
                                     library=arm["library"])
                step_cache[key] = step
                results["steps"].append(dict(_step_record(step), arm_first_used=arm_name))
            choose = standard_choice if arm_name == "H2" else gated_choice
            new_tree, decision = choose(step)
            label = decision.split()[1] if decision.startswith("ADOPT") else None
            arm["library"].append(lesson(prefix, decision, step.proposals.get(label) if label else None))
            arm["decisions"].append({"task": prefix, "decision": decision, "harness": new_tree.fingerprint()})
            arm["tree"] = new_tree
            log(f"   {arm_name}: {decision}")
            save()

    for arm_name, arm in arms.items():
        arm["tree"].write(out_dir / f"final_{arm_name}")
        results["arms"][arm_name] = {"final_harness": arm["tree"].fingerprint(), "tree": arm["tree"].to_dict(),
                                     "decisions": arm["decisions"], "library": arm["library"]}
    results["arms"]["H0"] = {"final_harness": DEFAULT_TREE.fingerprint(), "tree": DEFAULT_TREE.to_dict()}
    save()

    eval_seeds = [int(s) for s in args.eval_seeds.split(",") if s.strip()]
    finals = {"H0": DEFAULT_TREE, "H2": arms["H2"]["tree"], "H3": arms["H3"]["tree"]}
    for seed in eval_seeds:
        for arm_name, tree in finals.items():
            for prefix in HELD_OUT:
                log(f"== held-out {prefix} | arm {arm_name} | seed {seed}")
                r = runner.rollout(prefix, tree, f"eval-{arm_name}", seed=seed)
                results["held_out"].setdefault(arm_name, []).append(dict(r.summary(), seed=seed, utility=utility(r)))
                save()

    summary = {}
    for arm_name, rows in results["held_out"].items():
        summary[arm_name] = {
            "verified": sum(r["gate_accepted"] for r in rows), "runs": len(rows),
            "wall_s": round(sum(r["wall_s"] for r in rows), 1), "tokens": sum(r["tokens"] for r in rows),
            "truncations": sum(r["truncations"] for r in rows),
            "mean_utility": round(sum(r["utility"] for r in rows) / max(1, len(rows)), 1),
        }
    results["held_out_summary"] = summary
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    log(f"DONE {json.dumps(summary)}")


if __name__ == "__main__":
    main()
