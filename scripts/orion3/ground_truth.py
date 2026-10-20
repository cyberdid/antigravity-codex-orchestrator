"""Label every corpus artifact valid / invalid / ambiguous, independently of RIE.

Candidates run in fresh Docker containers (no network, no repository on the path),
six processes each. Metric outputs are compared with the scikit-learn reference
(temp/project_orion), retrieval outputs with a numpy brute-force reference;
RIE's oracle module is not used.

Usage: python scripts/orion3/ground_truth.py [--corpus DIR] [--out labels.json] [--only ID...]
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'temp/project_orion/src'))
import cases  # noqa: E402
from orion.metrics import calculate_metrics as reference_metrics  # noqa: E402

IMAGE = 'antigravity-orion:local'
PROCESSES = 6
TIMEOUT_S = 180
METRIC_KEYS = ['pairwise_precision', 'pairwise_recall', 'pairwise_f1', 'b_cubed_precision', 'b_cubed_recall',
               'b_cubed_f1', 'b_cubed_precision_global', 'b_cubed_recall_global', 'coverage', 'false_match_rate',
               'false_discovery_proportion', 'adjusted_rand_index', 'v_measure', 'total_observations',
               'accepted_observations', 'abstained_observations', 'outlier_observations', 'true_positive_pairs',
               'false_positive_pairs', 'false_negative_pairs', 'true_negative_pairs', 'accepted_pairs',
               'true_pairs', 'impostor_pairs']

DRIVER = r'''
import json, sys, traceback
import numpy as np
sys.path.insert(0, '/w/candidate/src')
req = json.load(open('/w/request.json'))
out = []
def conv(o):
    return {'__nonpython__': type(o).__name__, 'value': o.item() if hasattr(o, 'item') else str(o)}
if req['kind'] == 'metrics':
    from orion.metrics import calculate_metrics
    for t, p in req['cases']:
        try:
            r = calculate_metrics(np.asarray(t, dtype=int), np.asarray(p, dtype=int))
            out.append({'ok': r if isinstance(r, dict) else {'__notdict__': repr(r)[:200]}})
        except Exception as e:
            out.append({'error': type(e).__name__ + ': ' + str(e)[:200]})
else:
    from orion.retrieval import exact_knn, approximate_knn
    for ch in req['cases']:
        row = {}
        for name, fn in (('exact', exact_knn), ('approximate', approximate_knn)):
            try:
                i, d = fn(np.asarray(ch['vectors'], dtype=float), np.asarray(ch['queries'], dtype=float), ch['k'])
                row[name] = [np.asarray(i).tolist(), np.asarray(d).tolist()]
            except Exception as e:
                row[name] = {'error': type(e).__name__ + ': ' + str(e)[:200]}
        out.append(row)
json.dump(out, open('/o/result.json', 'w'), default=conv)
'''


def run_in_container(workspace, kind, payload, timeout_s=TIMEOUT_S):
    with tempfile.TemporaryDirectory(prefix='gt-in-') as inp, tempfile.TemporaryDirectory(prefix='gt-out-') as outp:
        shutil.copytree(workspace, Path(inp) / 'candidate')
        (Path(inp) / 'request.json').write_text(json.dumps({'kind': kind, 'cases': payload}))
        (Path(inp) / 'driver.py').write_text(DRIVER)
        os.chmod(outp, 0o777)
        name = 'orion3-gt-' + os.urandom(6).hex()
        cmd = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--memory', '2g', '--cpus', '2',
               '-u', f'{os.getuid()}:{os.getgid()}', '-v', f'{inp}:/w:ro', '-v', f'{outp}:/o', IMAGE, 'python3', '/w/driver.py']
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
            return None, 'timeout'
        result = Path(outp) / 'result.json'
        if proc.returncode or not result.exists():
            return None, 'crash: ' + (proc.stderr or proc.stdout)[-300:]
        return json.loads(result.read_text()), ''


def _same(value, target):
    if isinstance(value, dict) and '__nonpython__' in value:
        value = value['value']
    if target is None:
        return value is None
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and math.isclose(value, target, rel_tol=1e-8, abs_tol=1e-8))


def judge_metrics(outputs, metric_cases):
    failures, type_issues = [], 0
    for idx, ((t, p), out) in enumerate(zip(metric_cases, outputs)):
        if 'error' in out:
            failures.append(f'case {idx}: {out["error"]}')
            continue
        got = out['ok']
        ref = reference_metrics(np.asarray(t, dtype=int), np.asarray(p, dtype=int))
        bad = [k for k in METRIC_KEYS if k not in got or not _same(got[k], ref[k])]
        if bad:
            failures.append(f'case {idx} n={len(t)}: wrong {bad[:4]}')
        type_issues += any(isinstance(got.get(k), dict) for k in METRIC_KEYS)
    return failures, type_issues


def _true_sq(x, q, ids):
    return ((x[np.asarray(ids)] - q) ** 2).sum(axis=1)


def judge_retrieval(outputs, challenges):
    failures, recalls = [], []
    for idx, (ch, out) in enumerate(zip(challenges, outputs)):
        x, Q, k = np.asarray(ch['vectors']), np.asarray(ch['queries']), ch['k']
        indexed = {tuple(v) for v in ch['vectors']}
        exact_sets = []
        for name in ('exact', 'approximate'):
            res = out[name]
            if isinstance(res, dict):
                failures.append(f'ch {idx} {name}: {res["error"]}')
                continue
            ids, ds = res
            if len(ids) != len(Q) or len(ds) != len(Q):
                failures.append(f'ch {idx} {name}: wrong shape')
                continue
            for qi, (row_i, row_d) in enumerate(zip(ids, ds)):
                if (len(row_i) != k or len(row_d) != k or len(set(row_i)) != k
                        or any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(x) for i in row_i)):
                    failures.append(f'ch {idx} {name} q{qi}: invalid ids')
                    break
                truth = _true_sq(x, Q[qi], row_i)
                if not np.allclose(row_d, truth, rtol=1e-7, atol=1e-9):
                    failures.append(f'ch {idx} {name} q{qi}: distances are not true squared distances')
                    break
                if any(b < a - 1e-12 for a, b in zip(row_d, row_d[1:])):
                    failures.append(f'ch {idx} {name} q{qi}: not sorted')
                    break
                all_d = ((x - Q[qi]) ** 2).sum(axis=1)
                if name == 'exact':
                    if not np.allclose(sorted(truth), np.sort(all_d)[:k], rtol=1e-9, atol=1e-12):
                        failures.append(f'ch {idx} exact q{qi}: not the k nearest')
                        break
                    exact_sets.append(set(row_i))
                else:
                    if tuple(ch['queries'][qi]) in indexed and row_d[0] > 1e-12:
                        failures.append(f'ch {idx} approximate q{qi}: misses an identical indexed vector')
                        break
                    true_k = set(np.lexsort((np.arange(len(x)), all_d))[:k].tolist())
                    recalls.append(len(true_k & set(row_i)) / k)
    return failures, (sum(recalls) / len(recalls) if recalls else None)


def label_artifact(art_dir):
    meta = json.loads((art_dir / 'meta.json').read_text())
    workspace = art_dir / 'workspace'
    kind = 'retrieval' if meta['kind'] in ('retrieval', 'claims') else 'metrics'
    payload = cases.ground_truth_metric_cases() if kind == 'metrics' else cases.ground_truth_retrieval_challenges()
    runs, failures, type_issues, recalls = [], [], 0, []
    for _ in range(PROCESSES):
        outputs, error = run_in_container(workspace, kind, payload)
        if error:
            runs.append({'error': error[:300]})
            failures.append(error[:200])
            if error == 'timeout':
                break  # a hang is already invalid; do not spend five more timeouts on it
            continue
        if kind == 'metrics':
            f, t = judge_metrics(outputs, payload)
            type_issues += t
        else:
            f, recall = judge_retrieval(outputs, payload)
            recalls.append(recall)
        runs.append({'failures': len(f), 'first': f[:3]})
        failures += f
    code_valid = not failures
    label, reasons = ('valid' if code_valid else 'invalid'), failures[:5]
    if code_valid and type_issues:
        label, reasons = 'ambiguous', ['numerically correct but returns non-Python numeric types']
    record = {'kind': meta['kind'], 'stratum': meta.get('stratum'), 'cls': meta.get('cls'),
              'intended_label': meta.get('intended_label'), 'code_label': label, 'runs': runs,
              'mean_recall': (sum(r for r in recalls if r is not None) / len(recalls)) if recalls and all(r is not None for r in recalls) else None}
    if meta['kind'] == 'claims':
        sc = meta['scenario']
        # Claim validity: code must be valid, the claim true on RIE-like challenges, provenance honest.
        if label != 'valid':
            claim_label, why = 'invalid', 'claim scenario on invalid code'
        elif sc.get('mutate_after'):
            claim_label, why = 'invalid', 'promoted source differs from evaluated source and is wrong'
        elif sc.get('forge'):
            claim_label, why = 'invalid', 'evidence id fabricated'
        elif sc.get('report_extra'):
            claim_label, why = 'invalid', 'report states an unmeasured number'
        else:
            recall = rie_like_recall(workspace)
            record['rie_like_recall'] = recall
            ok = recall >= sc['claim_value']
            claim_label, why = ('valid' if ok else 'invalid'), f'mean recall on RIE-like challenges {recall:.3f} vs claim >= {sc["claim_value"]}'
        label, reasons = claim_label, [why]
    record.update(label=label, reasons=reasons)
    return record


def rie_like_recall(workspace):
    import random
    rng = random.Random(31337)
    chs = [cases.retrieval_challenge(rng) for _ in range(30)]
    outputs, error = run_in_container(workspace, 'retrieval', chs)
    if error:
        return 0.0
    _, recall = judge_retrieval(outputs, chs)
    return recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default=str(ROOT / 'benchmarks/orion3/corpus'))
    parser.add_argument('--out', default=str(ROOT / 'benchmarks/orion3/labels.json'))
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    out = Path(args.out)
    labels = json.loads(out.read_text()) if out.exists() else {}
    for art in sorted(Path(args.corpus).iterdir()):
        if (args.only and art.name not in args.only) or (not args.only and art.name in labels):
            continue
        labels[art.name] = label_artifact(art)
        rec = labels[art.name]
        flag = '' if rec['intended_label'] in (None, rec['label']) else '  <-- differs from intended'
        print(f"{art.name}: {rec['label']} ({'; '.join(rec['reasons'])[:150]}){flag}", flush=True)
        out.write_text(json.dumps(labels, indent=2))


if __name__ == '__main__':
    main()
