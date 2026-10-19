#!/usr/bin/env python3
"""GEMMA-ORION Phase 2B Controlled Benchmark: P0 (Baseline G0) vs P1 (RecoveryController).

Evaluates `P0` vs `P1` on 3 held-out, multi-file scientific R&D tasks (`H1`, `H2`, `H3`)
that were NOT used during Phase 1 or Phase 2A diagnosis:
- H1: Benjamini-Hochberg step-up FDR controller repair (3 input files + hidden holdout test)
- H2: RIE-gated Pareto frontier efficiency judge repair (3 input files + hidden holdout test)
- H3: Grouped cross-validation leakage repair (3 input files + hidden holdout test)

All inference parameters are held strictly constant across P0 and P1:
- model = gemma4:12b
- OLLAMA_THINK = false
- OLLAMA_NUM_CTX = 16384
- OLLAMA_NUM_PREDICT = 2048
- temperature = 0.2
- identical per-subtask wall-clock deadline (300s) and workspace initial state.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from antigravity_hybrid_orchestrator.planner.dag import SubTask
from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite
from antigravity_hybrid_orchestrator.workforce.worker import AutonomousWorker
from antigravity_hybrid_orchestrator.engine.cloud_assist import CloudAssist
from antigravity_hybrid_orchestrator.engine.control import ExecutionControl
from antigravity_hybrid_orchestrator.engine.escalation import EscalationLadder
from antigravity_hybrid_orchestrator.engine.loop import SelfCorrectionLoop

POLICIES: Dict[str, Dict[str, Any]] = {
    "P0": {
        "label": "P0 (Baseline G0: think=false, predict=2048, basic recovery)",
        "OLLAMA_THINK": "false",
        "OLLAMA_NUM_CTX": "16384",
        "OLLAMA_NUM_PREDICT": "2048",
        "OLLAMA_RECOVERY_POLICY": "P0",
        "OLLAMA_ADAPTIVE_NUM_PREDICT": "0",
    },
    "P1_REACTIVE": {
        "label": "P1_REACTIVE (Reactive RecoveryController only, no stage hints)",
        "OLLAMA_THINK": "false",
        "OLLAMA_NUM_CTX": "16384",
        "OLLAMA_NUM_PREDICT": "2048",
        "OLLAMA_RECOVERY_POLICY": "P1_REACTIVE",
        "OLLAMA_ADAPTIVE_NUM_PREDICT": "0",
    },
    "P1_STAGE": {
        "label": "P1_STAGE (Proactive [STAGE STATE] tool annotations only, P0 recovery)",
        "OLLAMA_THINK": "false",
        "OLLAMA_NUM_CTX": "16384",
        "OLLAMA_NUM_PREDICT": "2048",
        "OLLAMA_RECOVERY_POLICY": "P1_STAGE",
        "OLLAMA_ADAPTIVE_NUM_PREDICT": "0",
    },
    "P1": {
        "label": "P1 (Full RecoveryController: proactive stage hints + reactive recovery)",
        "OLLAMA_THINK": "false",
        "OLLAMA_NUM_CTX": "16384",
        "OLLAMA_NUM_PREDICT": "2048",
        "OLLAMA_RECOVERY_POLICY": "P1",
        "OLLAMA_ADAPTIVE_NUM_PREDICT": "0",
    },
    # Escalation ladder (run with --attempts 2): stage hints plus one ladder step each.
    "P1_STAGE_ADAPTIVE": {
        "label": "P1_STAGE + re-request truncated tool calls with num_predict 4096",
        "OLLAMA_THINK": "false",
        "OLLAMA_NUM_CTX": "16384",
        "OLLAMA_NUM_PREDICT": "2048",
        "OLLAMA_RECOVERY_POLICY": "P1_STAGE",
        "OLLAMA_ADAPTIVE_NUM_PREDICT": "1",
    },
    "P1_STAGE_ADAPTIVE_THINK": {
        "label": "P1_STAGE_ADAPTIVE + Thinking Mode on the retry attempt",
        "OLLAMA_THINK": "false",
        "OLLAMA_NUM_CTX": "16384",
        "OLLAMA_NUM_PREDICT": "2048",
        "OLLAMA_RECOVERY_POLICY": "P1_STAGE",
        "OLLAMA_ADAPTIVE_NUM_PREDICT": "1",
        "think_on_retry": True,
    },
    "P1_STAGE_ADAPTIVE_ASSIST": {
        "label": "P1_STAGE_ADAPTIVE + cloud guidance (spec level) before the retry attempt",
        "OLLAMA_THINK": "false",
        "OLLAMA_NUM_CTX": "16384",
        "OLLAMA_NUM_PREDICT": "2048",
        "OLLAMA_RECOVERY_POLICY": "P1_STAGE",
        "OLLAMA_ADAPTIVE_NUM_PREDICT": "1",
        "cloud_assist": True,
    },
}

TASK_DEADLINE_S = 300.0


class VramMonitor:
    """Polls nvidia-smi in background to record peak GPU memory usage (MiB)."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.peak_mib: int = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self) -> None:
        self.peak_mib = self._read_vram()
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join(timeout=2.0)
        return self.peak_mib

    @staticmethod
    def _read_vram() -> int:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
                timeout=2.0,
            ).strip()
            return int(out.splitlines()[0].strip())
        except Exception:
            return 0

    def _poll(self) -> None:
        while not self._stop.is_set():
            val = self._read_vram()
            if val > self.peak_mib:
                self.peak_mib = val
            time.sleep(self.interval_s)


def _snapshot_files(ws: Path) -> Dict[str, str]:
    snap: Dict[str, str] = {}
    for p in sorted(ws.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
            rel = str(p.relative_to(ws))
            try:
                snap[rel] = p.read_text(encoding="utf-8")
            except Exception:
                pass
    return snap


def _hash_snapshot(snap: Dict[str, str]) -> str:
    h = hashlib.sha256()
    for k, v in sorted(snap.items()):
        h.update(k.encode("utf-8"))
        h.update(b"\0")
        h.update(v.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _diff_snapshots(before: Dict[str, str], after: Dict[str, str]) -> str:
    diffs: List[str] = []
    all_keys = sorted(set(before.keys()) | set(after.keys()))
    for k in all_keys:
        b_lines = before.get(k, "").splitlines(keepends=True)
        a_lines = after.get(k, "").splitlines(keepends=True)
        if b_lines != a_lines:
            diffs.extend(
                difflib.unified_diff(b_lines, a_lines, fromfile=f"a/{k}", tofile=f"b/{k}")
            )
    return "".join(diffs)


# ---------------------------------------------------------------------------
# Held-Out Task H1: Benjamini-Hochberg Step-Up FDR Controller
# ---------------------------------------------------------------------------
def setup_h1_workspace(ws: Path) -> SubTask:
    (ws / "fdr_types.py").write_text(
        '"""Data structures for multiple hypothesis testing."""\n'
        "from dataclasses import dataclass\n"
        "from typing import List\n\n"
        "@dataclass\n"
        "class FdrDecision:\n"
        "    rejected_mask: List[bool]\n"
        "    num_rejected: int\n"
        "    critical_threshold: float\n",
        encoding="utf-8",
    )
    (ws / "fdr_control.py").write_text(
        '"""Benjamini-Hochberg step-up False Discovery Rate (FDR) controller."""\n'
        "from typing import Sequence\n"
        "from fdr_types import FdrDecision\n\n"
        "def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> FdrDecision:\n"
        '    """Apply the Benjamini-Hochberg step-up procedure at level alpha.\n\n'
        "    Let m = len(p_values). Sort p-values ascending: p_(1) <= p_(2) <= ... <= p_(m)\n"
        "    with 1-based rank k in {1, ..., m}. Find the largest k such that:\n"
        "        p_(k) <= (k / m) * alpha\n"
        "    If such k_max exists, reject all hypotheses with rank 1 .. k_max\n"
        "    (even if an earlier p_(j) for j < k_max had p_(j) > (j / m) * alpha),\n"
        "    and set critical_threshold = (k_max / m) * alpha.\n"
        "    If no k satisfies the inequality (or p_values is empty), reject none and\n"
        "    set critical_threshold = 0.0.\n"
        '    """\n'
        "    m = len(p_values)\n"
        "    if m == 0:\n"
        "        return FdrDecision(rejected_mask=[], num_rejected=0, critical_threshold=0.0)\n"
        "    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])\n"
        "    rejected = [False] * m\n"
        "    crit = 0.0\n"
        "    for i, (orig_idx, p_val) in enumerate(indexed):\n"
        "        thresh = (i / m) * alpha\n"
        "        if p_val < thresh:\n"
        "            rejected[orig_idx] = True\n"
        "            crit = thresh\n"
        "    return FdrDecision(\n"
        "        rejected_mask=rejected,\n"
        "        num_rejected=sum(rejected),\n"
        "        critical_threshold=round(crit, 10),\n"
        "    )\n",
        encoding="utf-8",
    )
    (ws / "test_fdr_control.py").write_text(
        "import unittest\n"
        "from fdr_control import benjamini_hochberg\n\n"
        "class TestFdrControl(unittest.TestCase):\n"
        "    def test_step_up_monotone_rejection(self):\n"
        "        pvals = [0.20, 0.024, 0.014, 0.09]\n"
        "        res = benjamini_hochberg(pvals, alpha=0.05)\n"
        "        self.assertEqual(res.rejected_mask, [False, True, True, False])\n"
        "        self.assertEqual(res.num_rejected, 2)\n"
        "        self.assertAlmostEqual(res.critical_threshold, 0.025, places=9)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    return SubTask(
        id="H1_fdr_step_up_fix",
        title="Fix Benjamini-Hochberg FDR controller",
        description=(
            "Inspect `fdr_types.py`, `fdr_control.py`, and `test_fdr_control.py`. "
            "Fix `benjamini_hochberg` in `fdr_control.py` so it satisfies its docstring "
            "specification and passes `python3 -m unittest -q`."
        ),
        input_files=["fdr_types.py", "fdr_control.py", "test_fdr_control.py"],
        affected_files=["fdr_control.py"],
        verification_command="python3 -m unittest -q",
    )


def verify_h1_holdout(ws: Path) -> bool:
    holdout_code = (
        "import fdr_control\n"
        "# 1. Visible test case\n"
        "r1 = fdr_control.benjamini_hochberg([0.20, 0.024, 0.014, 0.09], alpha=0.05)\n"
        "assert r1.rejected_mask == [False, True, True, False] and r1.num_rejected == 2\n"
        "# 2. Exact boundary equality p_(k) == (k/m)*alpha\n"
        "r2 = fdr_control.benjamini_hochberg([0.05, 0.05, 0.05], alpha=0.05)\n"
        "assert r2.rejected_mask == [True, True, True] and abs(r2.critical_threshold - 0.05) < 1e-9\n"
        "# 3. All null\n"
        "r3 = fdr_control.benjamini_hochberg([0.4, 0.5, 0.9], alpha=0.05)\n"
        "assert r3.rejected_mask == [False, False, False] and r3.num_rejected == 0 and r3.critical_threshold == 0.0\n"
        "# 4. Empty input\n"
        "r4 = fdr_control.benjamini_hochberg([], alpha=0.05)\n"
        "assert r4.rejected_mask == [] and r4.num_rejected == 0\n"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", holdout_code],
        cwd=str(ws),
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Held-Out Task H2: RIE-Gated Pareto Frontier Efficiency Judge
# ---------------------------------------------------------------------------
def setup_h2_workspace(ws: Path) -> SubTask:
    (ws / "candidate_spec.py").write_text(
        '"""Candidate telemetry record for Harness² evaluation."""\n'
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class HarnessCandidate:\n"
        "    candidate_id: str\n"
        "    rie_accepted: bool\n"
        "    wall_clock_s: float\n"
        "    total_tokens: int\n"
        "    tool_errors: int\n",
        encoding="utf-8",
    )
    (ws / "pareto_judge.py").write_text(
        '"""RIE-gated Pareto frontier selector for candidate harnesses."""\n'
        "from typing import List, Sequence\n"
        "from candidate_spec import HarnessCandidate\n\n"
        "def dominates(a: HarnessCandidate, b: HarnessCandidate) -> bool:\n"
        '    """Return True iff candidate `a` Pareto-dominates `b` on (wall_clock_s, total_tokens, tool_errors).\n\n'
        "    Lower values are better on all three metrics. `a` dominates `b` iff `a` is\n"
        "    no worse (<=) than `b` on ALL three metrics and strictly better (<) on AT LEAST ONE.\n"
        '    """\n'
        "    return (\n"
        "        a.wall_clock_s < b.wall_clock_s\n"
        "        and a.total_tokens < b.total_tokens\n"
        "        and a.tool_errors < b.tool_errors\n"
        "    )\n\n"
        "def select_pareto_frontier(candidates: Sequence[HarnessCandidate]) -> List[str]:\n"
        '    """Filter to `rie_accepted=True` candidates, then return IDs on the non-dominated Pareto frontier,\n'
        "    sorted by (wall_clock_s, total_tokens, tool_errors, candidate_id) ascending.\n"
        '    """\n'
        "    pool = list(candidates)\n"
        "    frontier = []\n"
        "    for cand in pool:\n"
        "        if not any(dominates(other, cand) for other in pool if other.candidate_id != cand.candidate_id):\n"
        "            frontier.append(cand)\n"
        "    frontier.sort(key=lambda c: (c.wall_clock_s, c.total_tokens, c.tool_errors, c.candidate_id))\n"
        "    return [c.candidate_id for c in frontier]\n",
        encoding="utf-8",
    )
    (ws / "test_pareto_judge.py").write_text(
        "import unittest\n"
        "from candidate_spec import HarnessCandidate\n"
        "from pareto_judge import dominates, select_pareto_frontier\n\n"
        "class TestParetoJudge(unittest.TestCase):\n"
        "    def test_rie_gate_and_pareto_dominance(self):\n"
        "        cands = [\n"
        "            HarnessCandidate('H0', True, 300.0, 8000, 2),\n"
        "            HarnessCandidate('H1', True, 180.0, 6000, 0),\n"
        "            HarnessCandidate('H2', True, 240.0, 4000, 0),\n"
        "            HarnessCandidate('H2_slow', True, 260.0, 4000, 0),\n"
        "            HarnessCandidate('H3_rejected', False, 90.0, 2000, 0),\n"
        "        ]\n"
        "        self.assertTrue(dominates(cands[2], cands[3]))\n"
        "        self.assertEqual(select_pareto_frontier(cands), ['H1', 'H2'])\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    return SubTask(
        id="H2_pareto_judge_fix",
        title="Fix Pareto frontier selector",
        description=(
            "Inspect `candidate_spec.py`, `pareto_judge.py`, and `test_pareto_judge.py`. "
            "Fix `pareto_judge.py` so `dominates` and `select_pareto_frontier` satisfy their "
            "docstring specifications and pass `python3 -m unittest -q`."
        ),
        input_files=["candidate_spec.py", "pareto_judge.py", "test_pareto_judge.py"],
        affected_files=["pareto_judge.py"],
        verification_command="python3 -m unittest -q",
    )


def verify_h2_holdout(ws: Path) -> bool:
    holdout_code = (
        "from candidate_spec import HarnessCandidate\n"
        "from pareto_judge import dominates, select_pareto_frontier\n"
        "a = HarnessCandidate('A', True, 100.0, 5000, 0)\n"
        "b = HarnessCandidate('B', True, 100.0, 5000, 1)\n"
        "c = HarnessCandidate('C', False, 10.0, 100, 0)\n"
        "assert dominates(a, b) is True\n"
        "assert dominates(a, a) is False\n"
        "assert select_pareto_frontier([b, c, a]) == ['A']\n"
        "assert select_pareto_frontier([c]) == []\n"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", holdout_code],
        cwd=str(ws),
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Held-Out Task H3: Grouped Cross-Validation Leakage Repair
# ---------------------------------------------------------------------------
def setup_h3_workspace(ws: Path) -> SubTask:
    (ws / "sample_record.py").write_text(
        '"""Dataset sample record with group_id for group-disjoint splitting."""\n'
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class SampleRecord:\n"
        "    sample_id: int\n"
        "    group_id: str\n"
        "    value: float\n",
        encoding="utf-8",
    )
    (ws / "group_splitter.py").write_text(
        '"""Deterministic group-disjoint K-fold splitter."""\n'
        "from typing import List, Sequence, Tuple\n"
        "from sample_record import SampleRecord\n\n"
        "def group_kfold_splits(\n"
        "    records: Sequence[SampleRecord], n_splits: int = 3\n"
        ") -> List[Tuple[List[int], List[int]]]:\n"
        '    """Split records into `n_splits` (train_ids, test_ids) folds with ZERO group leakage.\n\n'
        "    Requirements:\n"
        "    1. Every record with the same `group_id` must be assigned to the SAME fold.\n"
        "    2. Order unique group_ids by their first appearance in `records` and assign\n"
        "       the g-th unique group (0-indexed) to fold `g % n_splits`.\n"
        "    3. For each fold k in 0..n_splits-1:\n"
        "       - `test_ids` = [r.sample_id for r in records if r.group_id assigned to fold k]\n"
        "       - `train_ids` = [r.sample_id for r in records if r.group_id NOT assigned to fold k]\n"
        "    4. Raise ValueError if `n_splits < 2` or number of unique groups < `n_splits`.\n"
        '    """\n'
        "    if n_splits < 2:\n"
        "        raise ValueError('n_splits must be >= 2')\n"
        "    folds: List[Tuple[List[int], List[int]]] = []\n"
        "    for k in range(n_splits):\n"
        "        test_ids = [r.sample_id for i, r in enumerate(records) if i % n_splits == k]\n"
        "        train_ids = [r.sample_id for i, r in enumerate(records) if i % n_splits != k]\n"
        "        folds.append((train_ids, test_ids))\n"
        "    return folds\n",
        encoding="utf-8",
    )
    (ws / "test_group_splitter.py").write_text(
        "import unittest\n"
        "from sample_record import SampleRecord\n"
        "from group_splitter import group_kfold_splits\n\n"
        "class TestGroupSplitter(unittest.TestCase):\n"
        "    def test_no_group_leakage_across_folds(self):\n"
        "        records = [\n"
        "            SampleRecord(1, 'grp_A', 0.1),\n"
        "            SampleRecord(2, 'grp_A', 0.2),\n"
        "            SampleRecord(3, 'grp_B', 0.3),\n"
        "            SampleRecord(4, 'grp_B', 0.4),\n"
        "            SampleRecord(5, 'grp_C', 0.5),\n"
        "            SampleRecord(6, 'grp_C', 0.6),\n"
        "        ]\n"
        "        id_to_grp = {r.sample_id: r.group_id for r in records}\n"
        "        splits = group_kfold_splits(records, n_splits=3)\n"
        "        self.assertEqual(len(splits), 3)\n"
        "        for train_ids, test_ids in splits:\n"
        "            train_grps = {id_to_grp[i] for i in train_ids}\n"
        "            test_grps = {id_to_grp[i] for i in test_ids}\n"
        "            self.assertTrue(train_grps.isdisjoint(test_grps))\n"
        "        self.assertEqual(splits[0][1], [1, 2])\n"
        "        self.assertEqual(splits[1][1], [3, 4])\n"
        "        self.assertEqual(splits[2][1], [5, 6])\n\n"
        "    def test_too_few_groups_raises(self):\n"
        "        records = [SampleRecord(1, 'grp_A', 0.1), SampleRecord(2, 'grp_A', 0.2)]\n"
        "        with self.assertRaises(ValueError):\n"
        "            group_kfold_splits(records, n_splits=2)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    return SubTask(
        id="H3_group_kfold_leakage_fix",
        title="Fix group-disjoint K-fold splitter",
        description=(
            "Inspect `sample_record.py`, `group_splitter.py`, and `test_group_splitter.py`. "
            "Fix `group_kfold_splits` in `group_splitter.py` so it satisfies its docstring "
            "specification and passes `python3 -m unittest -q`."
        ),
        input_files=["sample_record.py", "group_splitter.py", "test_group_splitter.py"],
        affected_files=["group_splitter.py"],
        verification_command="python3 -m unittest -q",
    )


def verify_h3_holdout(ws: Path) -> bool:
    holdout_code = (
        "from sample_record import SampleRecord\n"
        "from group_splitter import group_kfold_splits\n"
        "recs = [\n"
        "    SampleRecord(10, 'g1', 1.0),\n"
        "    SampleRecord(20, 'g2', 2.0),\n"
        "    SampleRecord(30, 'g1', 3.0),\n"
        "    SampleRecord(40, 'g3', 4.0),\n"
        "    SampleRecord(50, 'g4', 5.0),\n"
        "    SampleRecord(60, 'g2', 6.0),\n"
        "]\n"
        "splits = group_kfold_splits(recs, n_splits=2)\n"
        "assert splits[0][1] == [10, 30, 40]\n"
        "assert splits[0][0] == [20, 50, 60]\n"
        "assert splits[1][1] == [20, 50, 60]\n"
        "assert splits[1][0] == [10, 30, 40]\n"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", holdout_code],
        cwd=str(ws),
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Held-Out Task H4: Weighted mean with defects split across two modules
# (neither fix alone makes the visible test pass)
# ---------------------------------------------------------------------------
def setup_h4_workspace(ws: Path) -> SubTask:
    (ws / "weights.py").write_text(
        '"""Weight normalization helpers."""\n'
        "from typing import List, Sequence\n\n"
        "def normalize(weights: Sequence[float]) -> List[float]:\n"
        '    """Return the weights scaled so that they sum to 1.0.\n\n'
        "    Raise ValueError if `weights` is empty, any weight is negative,\n"
        "    or the weights sum to 0.\n"
        '    """\n'
        "    total = max(weights)\n"
        "    return [w / total for w in weights]\n",
        encoding="utf-8",
    )
    (ws / "summary.py").write_text(
        '"""Weighted summary statistics."""\n'
        "from typing import Sequence\n"
        "from weights import normalize\n\n"
        "def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:\n"
        '    """Return sum(v_i * w_i) where w are `weights` normalized to sum to 1.0.\n\n'
        "    Raise ValueError if `values` and `weights` have different lengths.\n"
        '    """\n'
        "    w = normalize(weights)\n"
        "    return sum(v * wi for v, wi in zip(values, w)) / len(values)\n",
        encoding="utf-8",
    )
    (ws / "test_summary.py").write_text(
        "import unittest\n"
        "from summary import weighted_mean\n\n"
        "class TestWeightedMean(unittest.TestCase):\n"
        "    def test_weighted_mean(self):\n"
        "        self.assertAlmostEqual(weighted_mean([1.0, 2.0, 3.0], [1.0, 1.0, 2.0]), 2.25)\n\n"
        "    def test_length_mismatch_raises(self):\n"
        "        with self.assertRaises(ValueError):\n"
        "            weighted_mean([1.0, 2.0], [1.0])\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    return SubTask(
        id="H4_weighted_mean_two_module_fix",
        title="Fix weighted mean across weights and summary modules",
        description=(
            "Inspect `weights.py`, `summary.py`, and `test_summary.py`. "
            "Fix the code so `normalize` and `weighted_mean` satisfy their docstring "
            "specifications and pass `python3 -m unittest -q`."
        ),
        input_files=["weights.py", "summary.py", "test_summary.py"],
        affected_files=["weights.py", "summary.py"],
        verification_command="python3 -m unittest -q",
    )


def verify_h4_holdout(ws: Path) -> bool:
    holdout_code = (
        "from weights import normalize\n"
        "from summary import weighted_mean\n"
        "assert normalize([2.0, 2.0]) == [0.5, 0.5]\n"
        "for bad in ([], [1.0, -1.0], [0.0, 0.0]):\n"
        "    try:\n"
        "        normalize(bad)\n"
        "    except ValueError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError(f'normalize({bad}) must raise ValueError')\n"
        "assert abs(weighted_mean([10.0], [3.0]) - 10.0) < 1e-9\n"
        "assert abs(weighted_mean([0.0, 4.0], [3.0, 1.0]) - 1.0) < 1e-9\n"
        "try:\n"
        "    weighted_mean([1.0, 2.0, 3.0], [1.0, 1.0])\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('length mismatch must raise ValueError')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", holdout_code],
        cwd=str(ws),
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    return proc.returncode == 0


def _run_holdout(ws: Path, holdout_code: str) -> bool:
    proc = subprocess.run(
        [sys.executable, "-B", "-c", holdout_code],
        cwd=str(ws),
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Held-Out Task H5: Interval merging with three independent defects
# (one visible test per defect, so a first partial fix still fails)
# ---------------------------------------------------------------------------
def setup_h5_workspace(ws: Path) -> SubTask:
    (ws / "intervals.py").write_text(
        '"""Closed-interval utilities."""\n'
        "from typing import List, Sequence, Tuple\n\n"
        "def merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:\n"
        '    """Merge overlapping or touching closed intervals.\n\n'
        "    `intervals` holds (start, end) pairs in any order. Return a NEW list of merged\n"
        "    (start, end) tuples sorted by start; the input must not be modified. Intervals\n"
        "    that touch (next start == current end) are merged. Raise ValueError if any\n"
        "    interval has start > end.\n"
        '    """\n'
        "    result: List[Tuple[float, float]] = []\n"
        "    for start, end in intervals:\n"
        "        if result and start < result[-1][1]:\n"
        "            result[-1] = (result[-1][0], max(result[-1][1], end))\n"
        "        else:\n"
        "            result.append((start, end))\n"
        "    return result\n",
        encoding="utf-8",
    )
    (ws / "test_intervals.py").write_text(
        "import unittest\n"
        "from intervals import merge_intervals\n\n"
        "class TestMergeIntervals(unittest.TestCase):\n"
        "    def test_unsorted_input(self):\n"
        "        self.assertEqual(merge_intervals([(5, 6), (1, 3), (2, 4)]), [(1, 4), (5, 6)])\n\n"
        "    def test_touching_intervals_merge(self):\n"
        "        self.assertEqual(merge_intervals([(1, 2), (2, 3)]), [(1, 3)])\n\n"
        "    def test_reversed_interval_raises(self):\n"
        "        with self.assertRaises(ValueError):\n"
        "            merge_intervals([(1, 2), (3, 1)])\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    return SubTask(
        id="H5_merge_intervals_multi_defect_fix",
        title="Fix merge_intervals",
        description=(
            "Inspect `intervals.py` and `test_intervals.py`. Fix `merge_intervals` in "
            "`intervals.py` so it satisfies its docstring specification and passes "
            "`python3 -m unittest -q`."
        ),
        input_files=["intervals.py", "test_intervals.py"],
        affected_files=["intervals.py"],
        verification_command="python3 -m unittest -q",
    )


def verify_h5_holdout(ws: Path) -> bool:
    return _run_holdout(ws, (
        "from intervals import merge_intervals\n"
        "assert merge_intervals([]) == []\n"
        "assert merge_intervals([(1, 10), (2, 3)]) == [(1, 10)]\n"
        "assert merge_intervals([(3, 4), (1, 2)]) == [(1, 2), (3, 4)]\n"
        "assert merge_intervals([(0.5, 1.5), (1.5, 2.0)]) == [(0.5, 2.0)]\n"
        "src = [(4, 5), (1, 2)]\n"
        "merge_intervals(src)\n"
        "assert src == [(4, 5), (1, 2)], 'input must not be modified'\n"
        "try:\n"
        "    merge_intervals([(2, 1)])\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('start > end must raise ValueError')\n"
    ))


# ---------------------------------------------------------------------------
# Held-Out Task H6: Duration parser with a unit defect and missing validation
# (the holdout also checks the unit-order rule stated only in the docstring)
# ---------------------------------------------------------------------------
def setup_h6_workspace(ws: Path) -> SubTask:
    (ws / "durations.py").write_text(
        '"""Human-readable duration parsing."""\n'
        "import re\n\n"
        "UNITS = {'h': 60, 'm': 60, 's': 1}\n\n"
        "def parse_duration(text: str) -> int:\n"
        '    """Parse a duration such as "1h30m", "45s" or "2h5s" into total seconds.\n\n'
        "    Units are h (3600 s), m (60 s) and s (1 s). Each unit appears at most once and\n"
        "    units appear in the order h, m, s. Surrounding whitespace is ignored. Raise\n"
        "    ValueError for an empty string, an unknown unit, a number without a unit, or\n"
        "    any other text that does not follow this format.\n"
        '    """\n'
        "    total = 0\n"
        "    for value, unit in re.findall(r'(\\d+)([hms])', text):\n"
        "        total += int(value) * UNITS[unit]\n"
        "    return total\n",
        encoding="utf-8",
    )
    (ws / "test_durations.py").write_text(
        "import unittest\n"
        "from durations import parse_duration\n\n"
        "class TestParseDuration(unittest.TestCase):\n"
        "    def test_hours_and_minutes(self):\n"
        "        self.assertEqual(parse_duration('1h30m'), 5400)\n\n"
        "    def test_seconds(self):\n"
        "        self.assertEqual(parse_duration('45s'), 45)\n\n"
        "    def test_invalid_text_raises(self):\n"
        "        for bad in ('', 'abc', '10x'):\n"
        "            with self.assertRaises(ValueError):\n"
        "                parse_duration(bad)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    return SubTask(
        id="H6_parse_duration_multi_defect_fix",
        title="Fix parse_duration",
        description=(
            "Inspect `durations.py` and `test_durations.py`. Fix `parse_duration` in "
            "`durations.py` so it satisfies its docstring specification and passes "
            "`python3 -m unittest -q`."
        ),
        input_files=["durations.py", "test_durations.py"],
        affected_files=["durations.py"],
        verification_command="python3 -m unittest -q",
    )


def verify_h6_holdout(ws: Path) -> bool:
    return _run_holdout(ws, (
        "from durations import parse_duration\n"
        "assert parse_duration(' 2h5s ') == 7205\n"
        "assert parse_duration('0s') == 0\n"
        "assert parse_duration('3m') == 180\n"
        "for bad in ('1m1h', '1h1h', '90', 'h', '1h 30m x'):\n"
        "    try:\n"
        "        parse_duration(bad)\n"
        "    except ValueError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError(f'{bad!r} must raise ValueError')\n"
    ))


# ---------------------------------------------------------------------------
# Held-Out Task H7: Rolling mean with three independent defects
# ---------------------------------------------------------------------------
def setup_h7_workspace(ws: Path) -> SubTask:
    (ws / "rolling.py").write_text(
        '"""Sliding-window statistics."""\n'
        "from typing import List, Sequence\n\n"
        "def rolling_mean(values: Sequence[float], window: int) -> List[float]:\n"
        '    """Return the mean of every full window of `window` consecutive values.\n\n'
        "    The result has len(values) - window + 1 entries, in order; it is empty when\n"
        "    window > len(values). Raise ValueError if window is not an int or is < 1\n"
        "    (bool is not accepted as an int).\n"
        '    """\n'
        "    out: List[float] = []\n"
        "    for i in range(len(values) - window):\n"
        "        out.append(sum(values[i:i + window]) / len(values))\n"
        "    return out\n",
        encoding="utf-8",
    )
    (ws / "test_rolling.py").write_text(
        "import unittest\n"
        "from rolling import rolling_mean\n\n"
        "class TestRollingMean(unittest.TestCase):\n"
        "    def test_includes_last_window(self):\n"
        "        self.assertEqual(len(rolling_mean([1, 2, 3, 4], 2)), 3)\n\n"
        "    def test_window_mean_values(self):\n"
        "        self.assertEqual(rolling_mean([2, 4, 6, 8], 2)[:2], [3.0, 5.0])\n\n"
        "    def test_invalid_window_raises(self):\n"
        "        with self.assertRaises(ValueError):\n"
        "            rolling_mean([1, 2, 3], 0)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    return SubTask(
        id="H7_rolling_mean_multi_defect_fix",
        title="Fix rolling_mean",
        description=(
            "Inspect `rolling.py` and `test_rolling.py`. Fix `rolling_mean` in `rolling.py` so it "
            "satisfies its docstring specification and passes `python3 -m unittest -q`."
        ),
        input_files=["rolling.py", "test_rolling.py"],
        affected_files=["rolling.py"],
        verification_command="python3 -m unittest -q",
    )


def verify_h7_holdout(ws: Path) -> bool:
    return _run_holdout(ws, (
        "from rolling import rolling_mean\n"
        "assert rolling_mean([1.0, 2.0, 3.0], 3) == [2.0]\n"
        "assert rolling_mean([1.0, 2.0], 5) == []\n"
        "assert rolling_mean([4.0, 8.0], 1) == [4.0, 8.0]\n"
        "for bad in (-1, 1.5, True, '2'):\n"
        "    try:\n"
        "        rolling_mean([1.0, 2.0, 3.0], bad)\n"
        "    except ValueError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError(f'window={bad!r} must raise ValueError')\n"
    ))


# ---------------------------------------------------------------------------
# Held-Out Task H8: Slug generation with three independent defects
# ---------------------------------------------------------------------------
def setup_h8_workspace(ws: Path) -> SubTask:
    (ws / "slugs.py").write_text(
        '"""URL slug generation."""\n'
        "import re\n\n"
        "def slugify(title: str) -> str:\n"
        '    """Return a URL slug for `title`.\n\n'
        "    The slug is lowercase. Every run of characters other than ASCII letters and\n"
        "    digits becomes a single hyphen, and the slug never starts or ends with a\n"
        "    hyphen. Raise ValueError when no ASCII letter or digit remains.\n"
        '    """\n'
        "    return re.sub(r'[^A-Za-z0-9]', '-', title).lower()\n",
        encoding="utf-8",
    )
    (ws / "test_slugs.py").write_text(
        "import unittest\n"
        "from slugs import slugify\n\n"
        "class TestSlugify(unittest.TestCase):\n"
        "    def test_runs_collapse(self):\n"
        "        self.assertEqual(slugify('a  b'), 'a-b')\n\n"
        "    def test_edges_trimmed(self):\n"
        "        self.assertEqual(slugify(' Hello '), 'hello')\n\n"
        "    def test_empty_slug_raises(self):\n"
        "        with self.assertRaises(ValueError):\n"
        "            slugify('!!!')\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    return SubTask(
        id="H8_slugify_multi_defect_fix",
        title="Fix slugify",
        description=(
            "Inspect `slugs.py` and `test_slugs.py`. Fix `slugify` in `slugs.py` so it satisfies "
            "its docstring specification and passes `python3 -m unittest -q`."
        ),
        input_files=["slugs.py", "test_slugs.py"],
        affected_files=["slugs.py"],
        verification_command="python3 -m unittest -q",
    )


def verify_h8_holdout(ws: Path) -> bool:
    return _run_holdout(ws, (
        "from slugs import slugify\n"
        "assert slugify('Hello, World!') == 'hello-world'\n"
        "assert slugify('--A--') == 'a'\n"
        "assert slugify('x1 y2') == 'x1-y2'\n"
        "assert slugify('Кафе Mocha') == 'mocha'\n"
        "for bad in ('', '   ', 'Привіт'):\n"
        "    try:\n"
        "        slugify(bad)\n"
        "    except ValueError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError(f'{bad!r} must raise ValueError')\n"
    ))


TASKS: List[Tuple[str, Callable[[Path], SubTask], Callable[[Path], bool]]] = [
    ("H1_fdr_step_up_fix", setup_h1_workspace, verify_h1_holdout),
    ("H2_pareto_judge_fix", setup_h2_workspace, verify_h2_holdout),
    ("H3_group_kfold_leakage_fix", setup_h3_workspace, verify_h3_holdout),
    ("H4_weighted_mean_two_module_fix", setup_h4_workspace, verify_h4_holdout),
    ("H5_merge_intervals_multi_defect_fix", setup_h5_workspace, verify_h5_holdout),
    ("H6_parse_duration_multi_defect_fix", setup_h6_workspace, verify_h6_holdout),
    ("H7_rolling_mean_multi_defect_fix", setup_h7_workspace, verify_h7_holdout),
    ("H8_slugify_multi_defect_fix", setup_h8_workspace, verify_h8_holdout),
]


_SUMMED_METADATA = (
    "input_tokens", "output_tokens", "prompt_eval_cached_tokens", "eval_duration_s",
    "prompt_eval_duration_s", "turns_with_thinking", "thinking_chars", "tool_calls_count",
    "read_file_calls", "repeated_file_reads", "total_repeated_file_reads", "edit_calls",
    "test_command_calls", "invalid_tool_arguments", "context_overflows", "truncated_generations",
    "budget_escalations", "discarded_truncated_tool_calls", "compacted_messages",
    "successful_verifications", "recovery_attempts", "stage_hints_emitted",
)


class _PerAttemptDeadline:
    """Gives every attempt the same budget and keeps each attempt's telemetry."""

    def __init__(self, worker: AutonomousWorker, seconds: float) -> None:
        self.worker = worker
        self.seconds = seconds
        self.attempts: List[Dict[str, Any]] = []

    def execute_subtask(self, subtask, tool_suite, feedback_context=None, deadline=None, **kwargs):
        cap = time.monotonic() + self.seconds
        res = self.worker.execute_subtask(
            subtask, tool_suite, feedback_context=feedback_context,
            deadline=min(deadline, cap) if deadline else cap, **kwargs,
        )
        self.attempts.append({
            "think": kwargs.get("think"),
            "had_cloud_guidance": "Guidance from the cloud architect" in (feedback_context or ""),
            "success": res.success,
            "error": res.error,
            "metadata": dict(res.metadata or {}),
        })
        return res


def _aggregate_attempts(attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(attempts[-1]["metadata"]) if attempts else {}
    for key in _SUMMED_METADATA:
        values = [a["metadata"].get(key) for a in attempts]
        if any(isinstance(v, (int, float)) for v in values):
            merged[key] = sum(v for v in values if isinstance(v, (int, float)))
    merged["turn_trajectory"] = [
        dict(entry, attempt=index + 1)
        for index, attempt in enumerate(attempts)
        for entry in attempt["metadata"].get("turn_trajectory", [])
    ]
    merged["attempts_detail"] = [
        {k: v for k, v in a.items() if k != "metadata"} for a in attempts
    ]
    return merged


def run_single_task(
    policy_id: str,
    seed: int,
    task_id: str,
    setup_fn: Callable[[Path], SubTask],
    holdout_fn: Callable[[Path], bool],
    attempts: int = 1,
    cloud_client: Any = None,
) -> Dict[str, Any]:
    ws = Path(tempfile.mkdtemp(prefix=f"gemma_p2b_{policy_id}_s{seed}_{task_id}_"))
    try:
        subtask = setup_fn(ws)
        before_snap = _snapshot_files(ws)
        initial_sha = _hash_snapshot(before_snap)

        tool_suite = WorkspaceToolSuite(workspace_root=ws)
        policy = POLICIES[policy_id]
        worker = AutonomousWorker(model_name="gemma4:12b", recovery_policy=policy["OLLAMA_RECOVERY_POLICY"])
        assist_events: List[Dict[str, Any]] = []

        vram = VramMonitor()
        vram.start()
        t0 = time.monotonic()
        if attempts <= 1:
            res = worker.execute_subtask(
                subtask,
                tool_suite,
                deadline=t0 + TASK_DEADLINE_S,
            )
            attempt_records: List[Dict[str, Any]] = []
        else:
            # Same ladder code path as the orchestrator: SelfCorrectionLoop + EscalationLadder.
            os.environ["RD_MAX_REPAIR_ATTEMPTS"] = str(attempts)
            os.environ["RD_SUBTASK_TIMEOUT_S"] = str(TASK_DEADLINE_S * attempts)
            control = ExecutionControl()
            assist = None
            if policy.get("cloud_assist"):
                assist = CloudAssist(
                    cloud_client, ws, max_calls=1, execution_control=control,
                    audit=lambda event, _sid, payload: assist_events.append({"event": event, **payload}),
                )
            wrapped = _PerAttemptDeadline(worker, TASK_DEADLINE_S)
            loop = SelfCorrectionLoop(
                tool_suite=tool_suite, worker=wrapped, checkpoint_git=False, execution_control=control,
                ladder=EscalationLadder(think_on_retry=bool(policy.get("think_on_retry")), cloud_assist=assist),
            )
            res = loop.execute_subtask_with_retry(subtask)
            attempt_records = wrapped.attempts
            ladder_meta = {k: res.metadata.get(k) for k in (
                "think_retry_used", "ladder_steps_used", "cloud_assist_calls", "solved_with_cloud_assist")}
            res.metadata = dict(_aggregate_attempts(attempt_records), **ladder_meta)
        wall_s = round(time.monotonic() - t0, 4)
        peak_vram = vram.stop()

        after_snap = _snapshot_files(ws)
        final_sha = _hash_snapshot(after_snap)
        diff_text = _diff_snapshots(before_snap, after_snap)

        # Independent verification: both visible unittest and hidden holdout check
        vis_proc = subprocess.run(
            [sys.executable, "-m", "unittest", "-q"],
            cwd=str(ws),
            capture_output=True,
            text=True,
            timeout=20.0,
        )
        visible_passed = vis_proc.returncode == 0
        holdout_passed = holdout_fn(ws) if visible_passed else False
        independently_verified = bool(visible_passed and holdout_passed)
        solved_with_cloud_assist = bool(independently_verified and res.metadata.get("solved_with_cloud_assist"))
        llm_native = bool(
            res.success
            and independently_verified
            and int(res.metadata.get("successful_verifications", 0)) >= 1
            and not solved_with_cloud_assist
        )

        total_tools = int(res.tool_calls_count)
        repeated_reads = int(res.metadata.get("repeated_file_reads", 0))
        invalid_args = int(res.metadata.get("invalid_tool_arguments", 0))
        productive_tools = max(0, total_tools - repeated_reads - invalid_args)
        tool_efficiency = round(productive_tools / total_tools, 4) if total_tools > 0 else 0.0

        meta = dict(res.metadata)
        meta.update({
            "seed": seed,
            "policy_id": policy_id,
            "visible_tests_passed": visible_passed,
            "holdout_tests_passed": holdout_passed,
            "pipeline_verified": independently_verified,
            "llm_native_completion": llm_native,
            "fallback_used": False,
            "tool_call_efficiency": tool_efficiency,
            "initial_workspace_sha256": initial_sha,
            "final_workspace_sha256": final_sha,
            "workspace_modified": initial_sha != final_sha,
            "workspace_diff": diff_text,
            "attempts_used": len(attempt_records) or 1,
            "solved_with_cloud_assist": solved_with_cloud_assist,
            "cloud_assist_tokens": sum(
                int(e.get("input_tokens", 0)) + int(e.get("output_tokens", 0))
                for e in assist_events if e["event"] == "cloud_assist_response"
            ),
            "cloud_assist_events": assist_events,
        })

        return {
            "task_id": task_id,
            "policy_id": policy_id,
            "seed": seed,
            "worker_reported_success": res.success,
            "independently_verified": independently_verified,
            "pipeline_verified": independently_verified,
            "llm_native_completion": llm_native,
            "fallback_used": False,
            "turns": res.turns,
            "tool_calls_count": total_tools,
            "tool_call_efficiency": tool_efficiency,
            "wall_clock_s": wall_s,
            "peak_vram_mib": peak_vram,
            "error": res.error,
            "metadata": meta,
        }
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def _mean_std(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": 0.0, "pstdev": 0.0, "sample_stdev": 0.0, "min": 0.0, "max": 0.0}
    m = statistics.mean(vals)
    ps = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    ss = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {
        "mean": round(m, 4),
        "pstdev": round(ps, 4),
        "sample_stdev": round(ss, 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GEMMA-ORION Phase 2B P0 vs P1 Benchmark")
    parser.add_argument("--policies", default="P0,P1", help="Comma-separated policies (P0,P1)")
    parser.add_argument("--seeds", default="303", help="Comma-separated integer seeds")
    parser.add_argument("--tasks", default="H1,H2,H3", help="Comma-separated task prefixes (H1,H2,H3)")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "benchmarks" / "gemma_orion_phase2b_p0_vs_p1.json"),
        help="Path to output JSON file",
    )
    parser.add_argument("--attempts", type=int, default=1,
                        help="Attempts per task through SelfCorrectionLoop (1 = single worker attempt)")
    parser.add_argument("--cloud-backend", choices=["agy", "haiku", "luna"], default="agy",
                        help="Real cloud backend for *_ASSIST policies (owner-approved for this benchmark)")
    args = parser.parse_args()

    selected_policies = [p.strip().upper() for p in args.policies.split(",") if p.strip()]
    cloud_client = None
    if any(POLICIES[p].get("cloud_assist") for p in selected_policies):
        if args.attempts < 2:
            parser.error("cloud-assist policies need --attempts 2 or more")
        from antigravity_hybrid_orchestrator.architect.client import CloudArchitectClient

        cloud_client = CloudArchitectClient(backend=args.cloud_backend)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    task_prefixes = [t.strip().upper() for t in args.tasks.split(",") if t.strip()]
    active_tasks = [
        (tid, sfn, hfn)
        for (tid, sfn, hfn) in TASKS
        if any(tid.upper().startswith(pref) for pref in task_prefixes)
    ]

    git_head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT), text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True
    ).strip()
    git_dirty = bool(git_status)
    git_diff_bytes = subprocess.check_output(
        ["git", "diff", "HEAD"], cwd=str(REPO_ROOT)
    )
    working_tree_diff_sha256 = hashlib.sha256(git_diff_bytes).hexdigest() if git_dirty else "clean"
    git_commit = f"{git_head}-dirty" if git_dirty else git_head

    # Ensure subprocesses launched via WorkspaceToolSuite.run_command("python3 -m pytest -q")
    # resolve `python3` to the active virtualenv interpreter (`sys.executable`).
    venv_bin = str(Path(sys.executable).resolve().parent)
    current_path = os.environ.get("PATH", "")
    if venv_bin not in current_path.split(os.pathsep):
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{current_path}"

    results: Dict[str, Any] = {
        "experiment": "GEMMA-ORION Phase 2B Recovery Controller Ablation (P0 vs P1_REACTIVE vs P1_STAGE vs P1)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit,
        "git_head": git_head,
        "git_dirty": git_dirty,
        "working_tree_diff_sha256": working_tree_diff_sha256,
        "seeds": seeds,
        "task_deadline_s": TASK_DEADLINE_S,
        "attempts": args.attempts,
        "cloud_backend": args.cloud_backend if cloud_client is not None else None,
        "counterbalanced_seed_order": True,
        "policies": {},
    }

    # Run counterbalanced across seeds (forward policy order on even seed index, reversed on odd seed index)
    per_policy_seed_tasks: Dict[str, Dict[int, List[Dict[str, Any]]]] = {
        pol_id: {s: [] for s in seeds} for pol_id in selected_policies
    }

    for seed_idx, seed in enumerate(seeds):
        os.environ["OLLAMA_SEED"] = str(seed)
        ordered_policies = (
            list(selected_policies)
            if seed_idx % 2 == 0
            else list(reversed(selected_policies))
        )
        for pol_id in ordered_policies:
            pol_cfg = POLICIES[pol_id]
            for k, v in pol_cfg.items():
                if k.startswith("OLLAMA_"):
                    os.environ[k] = v

            print(f"\n=== Running Seed {seed} | Policy {pol_id} ===", flush=True)
            for tid, setup_fn, holdout_fn in active_tasks:
                print(f"  -> Task {tid} ...", end="", flush=True)
                tr = run_single_task(pol_id, seed, tid, setup_fn, holdout_fn,
                                     attempts=args.attempts, cloud_client=cloud_client)
                per_policy_seed_tasks[pol_id][seed].append(tr)
                print(
                    f" verified={tr['pipeline_verified']} native={tr['llm_native_completion']} "
                    f"turns={tr['turns']} tools={tr['tool_calls_count']} "
                    f"trunc={tr['metadata']['truncated_generations']} "
                    f"rec_attempts={tr['metadata']['recovery_attempts']} "
                    f"stage_hints={tr['metadata'].get('stage_hints_emitted', 0)} "
                    f"wall={tr['wall_clock_s']}s",
                    flush=True,
                )

    for pol_id in selected_policies:
        pol_cfg = POLICIES[pol_id]
        seed_runs: List[Dict[str, Any]] = []
        for seed in seeds:
            tasks_out = per_policy_seed_tasks[pol_id][seed]
            verified_cnt = sum(1 for t in tasks_out if t["pipeline_verified"])
            native_cnt = sum(1 for t in tasks_out if t["llm_native_completion"])
            in_tok = sum(int(t["metadata"]["input_tokens"]) for t in tasks_out)
            out_tok = sum(int(t["metadata"]["output_tokens"]) for t in tasks_out)
            tot_tok = in_tok + out_tok
            wall_tot = round(sum(float(t["wall_clock_s"]) for t in tasks_out), 4)
            trunc_tot = sum(int(t["metadata"]["truncated_generations"]) for t in tasks_out)
            rec_tot = sum(int(t["metadata"]["recovery_attempts"]) for t in tasks_out)
            stage_tot = sum(int(t["metadata"].get("stage_hints_emitted", 0)) for t in tasks_out)
            rep_reads = sum(int(t["metadata"]["repeated_file_reads"]) for t in tasks_out)

            seed_runs.append({
                "seed": seed,
                "tasks": tasks_out,
                "aggregate": {
                    "tasks_total": len(tasks_out),
                    "pipeline_verified_count": verified_cnt,
                    "pipeline_verified_rate": round(verified_cnt / max(1, len(tasks_out)), 4),
                    "llm_native_completion_count": native_cnt,
                    "llm_native_completion_rate": round(native_cnt / max(1, len(tasks_out)), 4),
                    "fallback_used_count": 0,
                    "total_input_tokens": in_tok,
                    "total_output_tokens": out_tok,
                    "total_tokens": tot_tok,
                    "total_wall_clock_s": wall_tot,
                    "truncated_generations": trunc_tot,
                    "recovery_attempts": rec_tot,
                    "stage_hints_emitted": stage_tot,
                    "repeated_file_reads": rep_reads,
                    "peak_vram_mib": max((int(t["peak_vram_mib"]) for t in tasks_out), default=0),
                },
            })

        results["policies"][pol_id] = {
            "label": pol_cfg["label"],
            "env": {k: v for k, v in pol_cfg.items() if k.startswith("OLLAMA_")},
            "seed_runs": seed_runs,
            "summary_across_seeds": {
                "pipeline_verified_rate": _mean_std([r["aggregate"]["pipeline_verified_rate"] for r in seed_runs]),
                "llm_native_completion_rate": _mean_std([r["aggregate"]["llm_native_completion_rate"] for r in seed_runs]),
                "total_wall_clock_s": _mean_std([r["aggregate"]["total_wall_clock_s"] for r in seed_runs]),
                "total_tokens": _mean_std([float(r["aggregate"]["total_tokens"]) for r in seed_runs]),
                "truncated_generations": _mean_std([float(r["aggregate"]["truncated_generations"]) for r in seed_runs]),
                "recovery_attempts": _mean_std([float(r["aggregate"]["recovery_attempts"]) for r in seed_runs]),
                "stage_hints_emitted": _mean_std([float(r["aggregate"]["stage_hints_emitted"]) for r in seed_runs]),
                "repeated_file_reads": _mean_std([float(r["aggregate"]["repeated_file_reads"]) for r in seed_runs]),
            },
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nSaved Phase 2B benchmark results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
