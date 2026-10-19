#!/usr/bin/env python3
"""Controlled GEMMA-ORION Phase 1 A/B Smoke Benchmark (G0 vs G1 vs G2).

Evaluates local `gemma4:12b` on identical R&D coding/verification tasks and
deterministic report synthesis without modifying global Ollama settings or
existing Orion-2 artifacts.

Configurations:
- G0: OLLAMA_THINK=false, OLLAMA_NUM_CTX=16384, OLLAMA_NUM_PREDICT=2048
- G1: OLLAMA_THINK=true,  OLLAMA_NUM_CTX=16384, OLLAMA_NUM_PREDICT=2048
- G2: OLLAMA_THINK=true,  OLLAMA_NUM_CTX=16384, OLLAMA_NUM_PREDICT=4096
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Dict, List

import numpy as np

from antigravity_hybrid_orchestrator.engine.loop import SelfCorrectionLoop
from antigravity_hybrid_orchestrator.integrity.evidence import validate_narrative
from antigravity_hybrid_orchestrator.integrity.report_builder import ResearchReportBuilder
from antigravity_hybrid_orchestrator.planner.dag import SubTask
from antigravity_hybrid_orchestrator.workforce.ollama_client import WorkforceClient
from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite
from antigravity_hybrid_orchestrator.workforce.worker import AutonomousWorker


CONFIGURATIONS: Dict[str, Dict[str, str]] = {
    "G0": {
        "OLLAMA_THINK": "false",
        "OLLAMA_NUM_CTX": "16384",
        "OLLAMA_NUM_PREDICT": "2048",
    },
    "G1": {
        "OLLAMA_THINK": "true",
        "OLLAMA_NUM_CTX": "16384",
        "OLLAMA_NUM_PREDICT": "2048",
    },
    "G2": {
        "OLLAMA_THINK": "true",
        "OLLAMA_NUM_CTX": "16384",
        "OLLAMA_NUM_PREDICT": "4096",
    },
}


def _setup_task_b1(workspace: Path) -> tuple[SubTask, Callable[[Path], Dict[str, Any]]]:
    """Task B1: Multi-file debugging of selective pairwise metrics with abstention (-1)."""
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "src" / "metric_eval.py").write_text(
        '''"""Selective pairwise clustering metrics."""

def selective_pairwise_metrics(labels_true, labels_pred):
    """Compute pairwise precision, recall, f1, and fdp on non-abstained points (labels_pred != -1).

    Known true entities have labels_true >= 0; true outliers have labels_true == -1.
    """
    n = len(labels_true)
    if n != len(labels_pred):
        raise ValueError("Length mismatch")

    tp = 0
    fp = 0
    fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            # BUG 1: does not check if i or j was abstained (labels_pred == -1) in pred_same
            pred_same = (labels_pred[i] == labels_pred[j])
            true_same = (labels_true[i] >= 0 and labels_true[j] >= 0 and labels_true[i] == labels_true[j])
            if pred_same and true_same:
                tp += 1
            elif pred_same and not true_same:
                fp += 1
            elif not pred_same and true_same:
                fn += 1

    # BUG 2: denominator uses tp - fp instead of tp + fp, and zero-decision precision should be 0.0
    precision = tp / (tp - fp) if (tp - fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fdp = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {
        "pairwise_precision": float(precision),
        "pairwise_recall": float(recall),
        "pairwise_f1": float(f1),
        "false_discovery_proportion": float(fdp),
    }
''',
        encoding="utf-8",
    )
    (workspace / "test_metric_eval.py").write_text(
        '''import unittest
from src.metric_eval import selective_pairwise_metrics

class TestSelectiveMetrics(unittest.TestCase):
    def test_abstained_points_are_not_merged(self):
        # Points 2 and 3 are abstained (-1); they must NOT count as predicted same cluster
        res = selective_pairwise_metrics([0, 0, -1, -1], [1, 1, -1, -1])
        self.assertAlmostEqual(res["pairwise_precision"], 1.0)
        self.assertAlmostEqual(res["pairwise_recall"], 1.0)
        self.assertAlmostEqual(res["false_discovery_proportion"], 0.0)

    def test_all_abstained_gives_zero_precision(self):
        res = selective_pairwise_metrics([0, 0, 1, 1], [-1, -1, -1, -1])
        self.assertAlmostEqual(res["pairwise_precision"], 0.0)
        self.assertAlmostEqual(res["pairwise_recall"], 0.0)
        self.assertAlmostEqual(res["pairwise_f1"], 0.0)

if __name__ == "__main__":
    unittest.main()
''',
        encoding="utf-8",
    )

    subtask = SubTask(
        id="b1_selective_metrics_fix",
        title="Fix abstention and precision formula bugs in selective_pairwise_metrics",
        description=(
            "Inspect `src/metric_eval.py` and `test_metric_eval.py`. Fix the two bugs in "
            "`selective_pairwise_metrics`: (1) two points with `labels_pred == -1` are abstained "
            "and must never be treated as `pred_same = True`, and (2) `pairwise_precision` must divide "
            "`tp` by `(tp + fp)` (returning `0.0` when `tp + fp == 0`). Run the unit test to verify."
        ),
        input_files=["src/metric_eval.py", "test_metric_eval.py"],
        outputs=["src/metric_eval.py"],
        affected_files=["src/metric_eval.py"],
        success_criteria="Unit tests in test_metric_eval.py pass and independent oracle check succeeds.",
        verification_command="python3 -m unittest test_metric_eval.py",
    )

    def independent_oracle(ws: Path) -> Dict[str, Any]:
        import importlib.util

        spec = importlib.util.spec_from_file_location("cand_metric", ws / "src" / "metric_eval.py")
        if spec is None or spec.loader is None:
            return {"status": "REJECTED", "feedback": "Could not load src/metric_eval.py"}
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Held-out challenge case not present in test_metric_eval.py
        out = mod.selective_pairwise_metrics([0, 0, 0, 1, 1, -1], [5, 5, -1, 5, -1, -1])
        expected_prec = 1.0 / 3.0
        expected_rec = 1.0 / 4.0
        expected_fdp = 2.0 / 3.0
        expected_f1 = 2 * expected_prec * expected_rec / (expected_prec + expected_rec)
        ok = (
            np.isclose(out["pairwise_precision"], expected_prec)
            and np.isclose(out["pairwise_recall"], expected_rec)
            and np.isclose(out["false_discovery_proportion"], expected_fdp)
            and np.isclose(out["pairwise_f1"], expected_f1)
        )
        return {
            "status": "ACCEPTED" if ok else "REJECTED",
            "feedback": "" if ok else f"Oracle mismatch: got {out}",
        }

    return subtask, independent_oracle


def _setup_task_b2(workspace: Path) -> tuple[SubTask, Callable[[Path], Dict[str, Any]]]:
    """Task B2: Fix ANN projection candidate reranking sort order and distance calculation."""
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "src" / "ann_index.py").write_text(
        '''"""Lightweight random-projection ANN candidate reranker."""
import numpy as np

def approximate_search(vectors: np.ndarray, queries: np.ndarray, k: int = 5, window: int = 32, seed: int = 7):
    """Return (indices, squared_distances) of shape (len(queries), k) sorted by ascending squared L2 distance."""
    rng = np.random.default_rng(seed)
    n, d = vectors.shape
    proj = rng.normal(size=(d, 4))
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)

    v_proj = vectors @ proj
    q_proj = queries @ proj
    sorted_orders = np.argsort(v_proj, axis=0)

    out_ids = np.empty((len(queries), k), dtype=np.int64)
    out_dists = np.empty((len(queries), k), dtype=np.float64)

    for qi in range(len(queries)):
        cand_set = set()
        for p in range(4):
            pos = int(np.searchsorted(v_proj[sorted_orders[:, p], p], q_proj[qi, p]))
            lo = max(0, pos - window // 2)
            hi = min(n, lo + window)
            cand_set.update(sorted_orders[lo:hi, p].tolist())
        cands = np.array(sorted(cand_set), dtype=np.int64)
        diffs = vectors[cands] - queries[qi]
        sq_dists = np.sum(diffs * diffs, axis=1)
        # BUG: sorts in descending order (-sq_dists) instead of ascending (sq_dists)
        top_local = np.argsort(-sq_dists)[:k]
        out_ids[qi] = cands[top_local]
        out_dists[qi] = sq_dists[top_local]

    return out_ids, out_dists
''',
        encoding="utf-8",
    )
    (workspace / "test_ann_index.py").write_text(
        '''import unittest
import numpy as np
from src.ann_index import approximate_search

class TestANNSearch(unittest.TestCase):
    def test_ascending_distances_and_self_lookup(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(60, 16))
        queries = vectors[:5].copy()
        ids, dists = approximate_search(vectors, queries, k=5, window=32)
        self.assertEqual(ids.shape, (5, 5))
        self.assertTrue(np.all(dists[:, 1:] >= dists[:, :-1]), "Distances must be sorted ascending")
        self.assertTrue(np.allclose(dists[:, 0], 0.0, atol=1e-12), "Exact query points must have distance 0")

if __name__ == "__main__":
    unittest.main()
''',
        encoding="utf-8",
    )

    subtask = SubTask(
        id="b2_ann_rerank_fix",
        title="Fix descending distance sort bug in approximate_search",
        description=(
            "Inspect `src/ann_index.py` and `test_ann_index.py`. Fix the bug in `approximate_search` "
            "where candidate distances are sorted in descending order instead of ascending squared L2 "
            "distance. Verify with `python3 -m unittest test_ann_index.py`."
        ),
        input_files=["src/ann_index.py", "test_ann_index.py"],
        outputs=["src/ann_index.py"],
        affected_files=["src/ann_index.py"],
        success_criteria="test_ann_index.py passes and independent RIE retrieval check accepts.",
        verification_command="python3 -m unittest test_ann_index.py",
    )

    def independent_oracle(ws: Path) -> Dict[str, Any]:
        import importlib.util

        spec = importlib.util.spec_from_file_location("cand_ann", ws / "src" / "ann_index.py")
        if spec is None or spec.loader is None:
            return {"status": "REJECTED", "feedback": "Could not load src/ann_index.py"}
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rng = np.random.default_rng(999)
        vectors = rng.normal(size=(80, 12))
        queries = vectors[10:18].copy()
        ids, dists = mod.approximate_search(vectors, queries, k=5, window=40)
        sorted_ok = bool(np.all(dists[:, 1:] >= dists[:, :-1]))
        exact_self_ok = bool(np.allclose(dists[:, 0], 0.0, atol=1e-10))
        unique_ok = all(len(set(row.tolist())) == 5 for row in ids)
        ok = sorted_ok and exact_self_ok and unique_ok
        return {
            "status": "ACCEPTED" if ok else "REJECTED",
            "feedback": "" if ok else "Independent ANN oracle failed",
        }

    return subtask, independent_oracle


def run_configuration(config_name: str, env_vars: Dict[str, str], orion2_dir: Path) -> Dict[str, Any]:
    """Runs B1, B2, and B3 under the specified configuration and collects full telemetry."""
    old_env = {k: os.environ.get(k) for k in env_vars}
    try:
        for k, v in env_vars.items():
            os.environ[k] = v

        client = WorkforceClient()
        model_digest = client.get_model_digest()
        task_records: List[Dict[str, Any]] = []

        for task_label, setup_fn in [("B1_selective_metrics", _setup_task_b1), ("B2_ann_rerank", _setup_task_b2)]:
            with tempfile.TemporaryDirectory(prefix=f"gemma-orion-{config_name}-{task_label}-") as tmp_dir:
                ws = Path(tmp_dir)
                subtask, oracle_fn = setup_fn(ws)
                suite = WorkspaceToolSuite(workspace_root=ws)
                worker = AutonomousWorker(client=client)
                loop = SelfCorrectionLoop(
                    tool_suite=suite,
                    worker=worker,
                    checkpoint_git=False,
                    acceptance_gate=lambda _st, _ws=ws, _oracle=oracle_fn: _oracle(_ws),
                )
                t0 = time.monotonic()
                res = loop.execute_subtask_with_retry(subtask)
                wall_s = round(time.monotonic() - t0, 3)
                meta = dict(res.metadata or {})
                verif = meta.get("verification", {}) or {}
                integrity = (verif.get("metadata", {}) or {}).get("integrity", {}) or {}
                rie_accepted = integrity.get("status") == "ACCEPTED"
                task_records.append({
                    "task_id": task_label,
                    "success": bool(res.success and rie_accepted),
                    "turns": res.turns,
                    "tool_calls_count": res.tool_calls_count,
                    "invalid_tool_arguments": int(meta.get("invalid_tool_arguments", 0)),
                    "context_overflows": int(meta.get("context_overflows", 0)),
                    "truncated_generations": int(meta.get("truncated_generations", 0)),
                    "input_tokens": int(meta.get("input_tokens", 0)),
                    "output_tokens": int(meta.get("output_tokens", 0)),
                    "turns_with_thinking": int(meta.get("turns_with_thinking", 0)),
                    "thinking_chars": int(meta.get("thinking_chars", 0)),
                    "wall_clock_s": wall_s,
                    "rie_status": integrity.get("status", "REJECTED" if not res.success else "ACCEPTED"),
                    "error": res.error,
                })

        # Task B3: Deterministic ResearchReportBuilder + live Gemma qualitative interpretation
        t0 = time.monotonic()
        builder = ResearchReportBuilder(
            study_dir=orion2_dir,
            profile="full",
            evidence_id="cd6dbb999a4b4a75a15cd864ade65db3",
        )
        usage_capture: Dict[str, Any] = {}
        orig_chat = client.chat

        def tracked_chat(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            out = orig_chat(*args, **kwargs)
            usage_capture.update(out)
            return out

        client.chat = tracked_chat  # type: ignore[method-assign]
        try:
            report_res = builder.build_report(client=client)
        finally:
            client.chat = orig_chat  # type: ignore[method-assign]
        b3_wall_s = round(time.monotonic() - t0, 3)
        independent_narrative_failures = validate_narrative(report_res.markdown, report_res.claims)
        b3_rie_ok = report_res.accepted and len(independent_narrative_failures) == 0

        task_records.append({
            "task_id": "B3_research_report_builder",
            "success": bool(b3_rie_ok),
            "turns": 1,
            "tool_calls_count": 0,
            "invalid_tool_arguments": 0,
            "context_overflows": 0,
            "truncated_generations": int(bool(usage_capture.get("truncated", False))),
            "input_tokens": int(usage_capture.get("input_tokens", 0)),
            "output_tokens": int(usage_capture.get("output_tokens", 0)),
            "turns_with_thinking": int(bool(usage_capture.get("has_thinking", False))),
            "thinking_chars": int(usage_capture.get("thinking_chars", 0)),
            "unverified_numbers_stripped": report_res.unverified_numbers_stripped,
            "wall_clock_s": b3_wall_s,
            "rie_status": "ACCEPTED" if b3_rie_ok else "REJECTED",
            "error": "; ".join(report_res.failures) if not b3_rie_ok else None,
        })

        succ = [r for r in task_records if r["success"]]
        total_tokens_succ = sum(r["input_tokens"] + r["output_tokens"] for r in succ)
        total_wall_s = round(sum(r["wall_clock_s"] for r in task_records), 3)

        return {
            "configuration": config_name,
            "env": dict(env_vars),
            "model": client.model,
            "model_digest": model_digest,
            "think_enabled": env_vars["OLLAMA_THINK"].lower() == "true",
            "num_ctx": int(env_vars["OLLAMA_NUM_CTX"]),
            "num_predict": int(env_vars["OLLAMA_NUM_PREDICT"]),
            "input_budget_tokens": int(env_vars["OLLAMA_NUM_CTX"]) - int(env_vars["OLLAMA_NUM_PREDICT"]),
            "tasks_completed": f"{len(succ)}/{len(task_records)}",
            "tasks_completed_count": len(succ),
            "tasks_total": len(task_records),
            "invalid_tool_calls": sum(r["invalid_tool_arguments"] for r in task_records),
            "context_overflows": sum(r["context_overflows"] for r in task_records),
            "truncated_generations": sum(r["truncated_generations"] for r in task_records),
            "total_wall_clock_s": total_wall_s,
            "mean_wall_clock_s": round(total_wall_s / len(task_records), 3),
            "tokens_per_successful_task": round(total_tokens_succ / len(succ), 1) if succ else 0.0,
            "total_input_tokens": sum(r["input_tokens"] for r in task_records),
            "total_output_tokens": sum(r["output_tokens"] for r in task_records),
            "total_thinking_chars": sum(r["thinking_chars"] for r in task_records),
            "rie_independent_pass": f"{sum(1 for r in task_records if r['rie_status'] == 'ACCEPTED')}/{len(task_records)}",
            "tasks": task_records,
        }
    finally:
        for k, old_v in old_env.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GEMMA-ORION Phase 1 G0/G1/G2 benchmark.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["G0", "G1", "G2"],
        choices=list(CONFIGURATIONS.keys()),
        help="Configurations to benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON benchmark report",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    orion2_dir = repo_root / "temp" / "project_orion" / "orion2-run"

    results = []
    for cfg in args.configs:
        summary = run_configuration(cfg, CONFIGURATIONS[cfg], orion2_dir=orion2_dir)
        results.append(summary)

    report = {
        "benchmark": "GEMMA-ORION-Phase1",
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
