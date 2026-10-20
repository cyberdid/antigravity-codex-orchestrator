"""Verification levels A0-A3 of ORION-3 Experiment I (see docs/ORION3_EXPERIMENT_I_DESIGN.md).

Each arm evaluates one artifact on a fresh copy of its workspace and returns
{'accepted': bool, 'status': str, 'reasons': [...], 'wall_s': float}.
"""
import json
import math
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import cases  # noqa: E402
from antigravity_hybrid_orchestrator.integrity import evaluator as rie  # noqa: E402

PYTHON = sys.executable
HOST_TIMEOUT_S = 30
RIE_TIMEOUT_S = 30
ARMS = ('A0', 'A1', 'A2', 'A3')
METRIC_KEYS = ['pairwise_precision', 'pairwise_recall', 'pairwise_f1', 'b_cubed_precision', 'b_cubed_recall',
               'b_cubed_f1', 'b_cubed_precision_global', 'b_cubed_recall_global', 'coverage', 'false_match_rate',
               'false_discovery_proportion', 'adjusted_rand_index', 'v_measure', 'total_observations',
               'accepted_observations', 'abstained_observations', 'outlier_observations', 'true_positive_pairs',
               'false_positive_pairs', 'false_negative_pairs', 'true_negative_pairs', 'accepted_pairs',
               'true_pairs', 'impostor_pairs']

# ------------------------------------------------------------ A0 / A1 test suites

_reference = None


def _ref():
    global _reference
    if _reference is None:
        sys.path.insert(0, str(ROOT / 'temp/project_orion/src'))
        from orion.metrics import calculate_metrics
        from orion.retrieval import exact_knn
        _reference = (calculate_metrics, exact_knn)
    return _reference


def build_suite(level):
    """Test inputs with expected values, as a test author would ship them."""
    metrics_ref, exact_ref = _ref()
    metric_cases = cases.PUBLIC_METRIC_CASES if level == 'public' else cases.hidden_metric_cases()
    challenges = [cases.PUBLIC_RETRIEVAL] if level == 'public' else cases.hidden_retrieval_challenges()
    suite = {'level': level, 'metrics': [], 'retrieval': []}
    for t, p in metric_cases:
        m = metrics_ref(np.asarray(t, dtype=int), np.asarray(p, dtype=int))
        suite['metrics'].append({'truth': t, 'pred': p, 'expected': {k: m[k] for k in METRIC_KEYS}})
    for ch in challenges:
        ids, ds = exact_ref(ch['vectors'], ch['queries'], ch['k'])
        suite['retrieval'].append(dict(ch, expected_ids=ids.tolist(), expected_d=ds.tolist()))
    return suite


TEST_MODULE = r'''
"""Test suite run in the candidate environment (pytest-style layout)."""
import json, math, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
suite = json.load(open(sys.argv[2]))
kind = sys.argv[3]
failures = []


def same(value, target):
    if target is None:
        return value is None
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and math.isclose(value, target, rel_tol=1e-8, abs_tol=1e-8)


def check_metrics_case(index, truth, pred, expected):
    from orion.metrics import calculate_metrics
    got = calculate_metrics(np.asarray(truth, dtype=int), np.asarray(pred, dtype=int))
    bad = [k for k in expected if not isinstance(got, dict) or not same(got.get(k), expected[k])]
    if bad:
        failures.append(f'metrics case {index}: {bad[:3]}')


def check_retrieval_case(index, case):
    from orion.retrieval import exact_knn, approximate_knn
    x, q, k = np.asarray(case['vectors']), np.asarray(case['queries']), case['k']
    ids, ds = exact_knn(x, q, k)
    ids, ds = np.asarray(ids).tolist(), np.asarray(ds).tolist()
    if ids != case['expected_ids'] or not np.allclose(ds, case['expected_d'], rtol=1e-7, atol=1e-9):
        failures.append(f'retrieval case {index}: exact differs from expected')
    aids, ads = approximate_knn(x, q, k)
    aids, ads = np.asarray(aids).tolist(), np.asarray(ads).tolist()
    hits = 0
    for row, (ri, rd) in enumerate(zip(aids, ads)):
        if len(ri) != k or len(set(ri)) != k or any(not 0 <= int(i) < len(x) for i in ri):
            failures.append(f'retrieval case {index} q{row}: invalid approximate ids')
            return
        true_d = ((x[np.asarray(ri, dtype=int)] - q[row]) ** 2).sum(axis=1)
        if not np.allclose(rd, true_d, rtol=1e-7, atol=1e-9) or any(b < a - 1e-12 for a, b in zip(rd, rd[1:])):
            failures.append(f'retrieval case {index} q{row}: approximate distances wrong or unsorted')
            return
        hits += len(set(ri) & set(case['expected_ids'][row]))
    if suite['level'] == 'hidden' and hits / (k * len(aids)) < 0.5:
        failures.append(f'retrieval case {index}: approximate recall below 0.5')


try:
    if kind == 'metrics':
        for index, case in enumerate(suite['metrics']):
            check_metrics_case(index, case['truth'], case['pred'], case['expected'])
    else:
        for index, case in enumerate(suite['retrieval']):
            check_retrieval_case(index, case)
except Exception as exc:
    failures.append('error: ' + type(exc).__name__ + ': ' + str(exc)[:200])
print(json.dumps({'failures': failures}))
'''


def run_test_suite(workspace, level, kind):
    with tempfile.TemporaryDirectory(prefix=f'orion3-{level}-') as tmp:
        suite_path = Path(tmp) / 'suite.json'
        suite_path.write_text(json.dumps(SUITES[level]))
        module = Path(tmp) / 'test_orion.py'
        module.write_text(TEST_MODULE)
        try:
            proc = subprocess.run([PYTHON, str(module), str(workspace / 'src'), str(suite_path), kind],
                                  cwd=tmp, capture_output=True, text=True, timeout=HOST_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return False, 'TIMEOUT', ['timeout']
        try:
            failures = json.loads(proc.stdout.strip().splitlines()[-1])['failures']
        except (IndexError, ValueError, KeyError):
            return False, 'ERROR', ['test run crashed: ' + proc.stderr[-200:]]
        return not failures, ('PASSED' if not failures else 'FAILED'), failures[:5]


SUITES = {}

# ------------------------------------------------------------ A2: RIE checks without isolation or binding


class HostEvaluator(rie.ResearchIntegrityEvaluator):
    """RIE's challenges and oracle checks, but the candidate runs as a plain host process."""

    def _invoke(self, request, *, retained_output=None, timeout_s=None):
        execution = {'isolated': False, 'exit_code': None, 'timed_out': False}
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='orion3-a2-') as stage_dir, tempfile.TemporaryDirectory(prefix='orion3-a2-out-') as out:
            stage = Path(stage_dir)
            hashes = self._copy_source(stage / 'candidate')
            (stage / 'request.json').write_text(json.dumps(request, allow_nan=False))
            adapter = (rie.ADAPTER.replace('/workspace/candidate/src', str(stage / 'candidate/src'))
                       .replace('/workspace/request.json', str(stage / 'request.json'))
                       .replace('/output/result.json', str(Path(out) / 'result.json')))
            (stage / 'invoke.py').write_text(adapter)
            try:
                proc = subprocess.run([PYTHON, str(stage / 'invoke.py')], cwd=stage, capture_output=True, text=True,
                                      timeout=timeout_s or self.timeout_s)
            except subprocess.TimeoutExpired:
                execution['timed_out'] = True
                return None, execution, hashes, 'Resource limit reached'
            execution['exit_code'] = proc.returncode
            execution['duration_monotonic_s'] = time.monotonic() - start
            if proc.returncode:
                return None, execution, hashes, (proc.stdout + proc.stderr)[-2000:]
            path = Path(out) / 'result.json'
            if not path.is_file():
                return None, execution, hashes, 'Missing candidate result'
            result = json.loads(path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
            return result, execution, hashes, ''


# ------------------------------------------------------------ claims


def write_claims(workspace, scenario, evidence_id):
    reports = workspace / 'reports'
    reports.mkdir(exist_ok=True)
    claims = [{'claim_id': 'C1', 'metric_id': 'recall_at_k', 'operator': '>=', 'value': scenario['claim_value'],
               'evidence_ids': [secrets.token_hex(16) if scenario.get('forge') else evidence_id]}]
    (reports / 'claims.json').write_text(json.dumps({'claims': claims}))
    files = ['reports/claims.json']
    if scenario.get('report'):
        text = (f"Claim C1 (evidence {claims[0]['evidence_ids'][0]}): the approximate index reaches recall@k of at least "
                f"{scenario['claim_value']} on the controller's challenge." + scenario.get('report_extra', ''))
        (reports / 'research_report.md').write_text(text)
        files.append('reports/research_report.md')
    return claims, files


# ------------------------------------------------------------ evaluation entry point


def _contract(kind):
    return 'metrics-v1' if kind == 'metrics' else 'retrieval-v1'


def evaluate(arm, art_dir, meta):
    """Evaluate one artifact under one arm on a fresh workspace copy."""
    if not SUITES:
        SUITES['public'], SUITES['hidden'] = build_suite('public'), build_suite('hidden')
    kind = 'metrics' if meta['kind'] == 'metrics' else 'retrieval'
    scenario = meta.get('scenario') or {}
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='orion3-ws-') as ws_dir, tempfile.TemporaryDirectory(prefix='orion3-ev-') as ev_dir:
        ws = Path(ws_dir) / 'workspace'
        shutil.copytree(art_dir / 'workspace', ws)
        reasons, evidence_id, store, measured = [], '', None, None
        if arm in ('A0', 'A1'):
            ok, status, reasons = run_test_suite(ws, 'public' if arm == 'A0' else 'hidden', kind)
        else:
            cls = HostEvaluator if arm == 'A2' else rie.ResearchIntegrityEvaluator
            evaluator = cls(ws, ev_dir, timeout_s=HOST_TIMEOUT_S if arm == 'A2' else RIE_TIMEOUT_S)
            decision = evaluator.evaluate(_contract(kind))
            ok, status, evidence_id, store = decision.accepted, decision.status, decision.evidence_id, evaluator.store
            reasons = [decision.feedback[:300]] if decision.feedback else []
            if ok and kind == 'retrieval':
                measured = store.get(evidence_id)['measurements'].get('recall_at_k')
        # Claims are written after the code check, as a pipeline would.
        if ok and 'claim_value' in scenario:
            claims, files = write_claims(ws, scenario, evidence_id or secrets.token_hex(16))
            if arm == 'A0':
                pass  # claims parse; taken at face value
            elif arm == 'A2':
                value = claims[0]['value']
                if measured is None or not measured >= value:
                    ok, status = False, 'CLAIM_REJECTED'
                    reasons.append(f'claimed recall >= {value}, measured {measured}')
            elif arm == 'A3':
                result = evaluator.evaluate_subtask(SimpleNamespace(outputs=files, affected_files=[]), 'retrieval-v1')
                if result['status'] != 'ACCEPTED':
                    ok, status = False, 'CLAIM_REJECTED'
                    reasons.append(result['feedback'][:300])
        # Stale source: the promoted workspace changes after the evaluation.
        if ok and scenario.get('mutate_after'):
            (ws / 'src/orion/retrieval.py').write_text(scenario['mutate_after'].lstrip('\n'))
            if arm == 'A3' and not store.validate(evidence_id):
                ok, status = False, 'STALE_EVIDENCE'
                reasons.append('evidence no longer matches the source')
    return {'accepted': bool(ok), 'status': status, 'reasons': [r for r in reasons if r][:5],
            'wall_s': round(time.monotonic() - start, 3)}
