"""Controller-owned numerical verification of Orion study files.

This module never imports candidate code. Profiles are operator contracts;
manifest hashes identify bytes but are not evidence of authenticity. Execution
and source binding belong to the isolated evaluator wrapping these functions.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile
import numpy as np
from .oracle import reference_metrics

METHODS = ('threshold_graph', 'agglomerative', 'rr_graph', 'risk_aware')
TOGGLES = {'full_system': None, 'without_reciprocal_consistency': 'use_reciprocal_consistency',
           'without_local_calibration': 'use_local_calibration',
           'without_cohesion_constraint': 'use_cohesion_constraint',
           'without_abstention': 'use_abstention',
           'without_adaptive_neighborhood': 'use_adaptive_neighborhood'}
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _profile(profile):
    if profile == 'smoke':
        return {'profile': 'smoke', 'development_seeds': [1], 'evaluation_seeds': [55],
                'generator': {'num_entities': 4, 'observations_per_entity': 12, 'embedding_dim': 32,
                              'variance': .35, 'noise_level': .1, 'inter_entity_similarity': .4, 'outliers_ratio': .1},
                'scenarios': [{'domain_shift': 0.}, {'domain_shift': .35}], 'threshold_grid': [.4, .65, .85],
                'sensitivity_grid': {'merge_threshold': [.4, .8], 'domain_shift': [0., .35]}}
    if profile != 'full':
        raise ValueError('Unknown operator study profile')
    return {'profile': 'full', 'development_seeds': [1, 42, 1024], 'evaluation_seeds': [55, 99, 2048],
            'generator': {'num_entities': 10, 'observations_per_entity': 30, 'embedding_dim': 192,
                          'variance': .5, 'noise_level': .15, 'inter_entity_similarity': .55,
                          'difficulty': 'hard', 'unequal_populations': True},
            'scenarios': [{'outliers_ratio': rate, 'domain_shift': shift}
                          for rate in [.01, .05, .1, .2, .5] for shift in [0., .35]],
            'threshold_grid': [.25, .4, .6, .8], 'maximum_development_fdp': .01,
            'sensitivity_grid': {'cluster_population': [5, 10, 20], 'noise': [.01, .15, .3],
                                 'dimensionality': [64, 128, 192], 'overlap': [.2, .55, .8],
                                 'domain_shift': [0., .25, .5], 'outlier_rate': [.01, .1, .3],
                                 'neighborhood_size': [5, 20, 50], 'merge_threshold': [.25, .5, .8]}}


def _file(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('invalid artifact path')
    rel = PurePosixPath(relative)
    if rel.is_absolute() or any(part in ('..', '.') for part in rel.parts):
        raise ValueError('artifact path escapes study directory')
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('symlink artifact is forbidden')
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError('missing or escaped artifact: ' + relative)
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError('artifact exceeds file resource policy')
    return path


def _digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(root, relative):
    path = _file(root, relative)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError('oversized JSON artifact')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('duplicate JSON object key')
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _npz(root, relative):
    path = _file(root, relative)
    with zipfile.ZipFile(path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > 32 * 1024 * 1024:
            raise ValueError('expanded NumPy archive exceeds resource policy')
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _same(actual, expected):
    if expected is None: return actual is None
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8)
    return actual == expected


def _params(method, threshold):
    if method == 'threshold_graph': return {'threshold': threshold}
    if method == 'agglomerative': return {'distance_threshold': math.sqrt(2 * (1 - threshold))}
    if method == 'rr_graph': return {'k': 20, 'threshold': threshold}
    return {'max_neighbors': 20, 'expansion_factor': 1.2, 'reciprocal_threshold': threshold,
            'calibration_factor': 1.1, 'abstention_threshold': threshold,
            'cohesion_threshold': max(0., threshold - .1),
            'use_adaptive_neighborhood': True, 'use_reciprocal_consistency': True,
            'use_local_calibration': True, 'use_cohesion_constraint': True, 'use_abstention': True}


def _supplemental_metrics(truth, pred, oracle):
    true = [('outlier', i) if x == -1 else ('known', x) for i, x in enumerate(truth)]
    partition = [('unresolved', i) if x == -1 else ('assigned', x) for i, x in enumerate(pred)]
    accepted = [i for i, p in enumerate(pred) if p != -1]
    tc = Counter(true); pc = Counter(pred[i] for i in accepted)
    cells = Counter((true[i], pred[i]) for i in accepted)
    by_pred = defaultdict(list); by_true = defaultdict(set)
    for (t, p), count in cells.items(): by_pred[p].append(count)
    for t, p in zip(true, partition): by_true[t].add(p)
    ac = Counter(true[i] for i in accepted); gc = Counter(partition)
    divide = lambda a, b: a / b if b else None
    selective = reference_metrics([truth[i] for i in accepted], [pred[i] for i in accepted])
    return {'recall_all_known_pairs': oracle['pairwise_recall'],
            'pairwise_recall_accepted': selective['pairwise_recall'],
            'observation_assignment_coverage': oracle['coverage'],
            'abstention_rate': divide(len(pred) - len(accepted), len(pred)),
            'assigned_non_singleton_fraction': divide(sum(v for v in pc.values() if v > 1), len(pred)),
            'point_contamination': divide(sum(sum(v) - max(v) for v in by_pred.values()), len(accepted)),
            'cluster_contamination': divide(sum(len(v) > 1 for v in by_pred.values()), len(pc)),
            'cluster_level_contamination': divide(sum(len(v) > 1 for v in by_pred.values()), len(pc)),
            'fragmentation': divide(sum(len(v) > 1 for v in by_true.values()), len(tc)),
            'safe_identity_recovery': divide(sum(len(v) == 1 and ac[t] == tc[t] and gc[next(iter(v))] == tc[t] for t, v in by_true.items()), len(tc)),
            'adjusted_rand_index_accepted': selective['adjusted_rand_index'],
            'v_measure_accepted': selective['v_measure']}


def _dataset_hash(arrays):
    h = hashlib.sha256()
    for name in ('embeddings', 'labels_true', 'domains', 'outlier_mask'):
        arr = arrays[name]; h.update(str(arr.shape).encode()); h.update(arr.tobytes())
    return h.hexdigest()


def _exact_neighbors(vectors, queries, k):
    """Independent NumPy squared-L2 scan, bounded to 16 by 8192 distances."""
    ids = np.empty((len(queries), k), dtype=np.int64)
    distances = np.empty((len(queries), k))
    for start in range(0, len(queries), 16):
        q = np.asarray(queries[start:start + 16], dtype=float)
        best_i = np.empty((len(q), 0), dtype=np.int64); best_d = np.empty((len(q), 0))
        for offset in range(0, len(vectors), 8192):
            x = np.asarray(vectors[offset:offset + 8192], dtype=float)
            # einsum keeps this implementation independent of SciPy/candidate retrieval.
            ds = np.sum(q*q, axis=1)[:, None] + np.sum(x*x, axis=1)[None, :] - 2*np.einsum('ij,kj->ik', q, x, optimize=False)
            ds = np.maximum(ds, 0.)
            candidate_d = np.concatenate((best_d, ds), axis=1)
            candidate_i = np.concatenate((best_i, np.broadcast_to(np.arange(offset, offset + len(x)), ds.shape)), axis=1)
            order = np.lexsort((candidate_i, candidate_d), axis=1)[:, :k]
            best_i = np.take_along_axis(candidate_i, order, axis=1)
            best_d = np.take_along_axis(candidate_d, order, axis=1)
        ids[start:start + len(q)] = best_i; distances[start:start + len(q)] = best_d
    return ids, distances


class _Validator:
    def __init__(self, root, profile):
        self.root = Path(root).resolve(strict=True)
        self.spec = _profile(profile); self.profile = profile
        self.failures = []; self.checks = []; self.measurements = {}; self.claim_links = []
        self.predictions = {}; self.dataset_keys = {}; self.frozen = None

    def require(self, condition, message):
        if not condition: self.failures.append(message)
        return bool(condition)

    def artifact_digest(self, relative, expected):
        return self.require(isinstance(expected, str) and bool(_SHA.fullmatch(expected)) and _digest(_file(self.root, relative)) == expected,
                            'artifact digest mismatch: ' + relative)

    def manifest(self):
        doc = _json(self.root, 'manifest.json')
        files = doc.get('files')
        if not self.require(isinstance(files, dict), 'manifest files mapping missing'): return
        actual = set()
        for path in self.root.rglob('*'):
            if path.is_symlink(): raise ValueError('symlink in study manifest tree')
            if path.is_file() and path.name != 'manifest.json': actual.add(str(path.relative_to(self.root)))
        if len(actual) > 4096: raise ValueError('too many study files')
        self.require(set(files) == actual, 'manifest file inventory differs from actual artifacts')
        for relative, digest in files.items(): self.artifact_digest(relative, digest)
        source = doc.get('source_sha256')
        self.require(isinstance(source, dict) and bool(source) and all(isinstance(v, str) and _SHA.fullmatch(v) for v in source.values()), 'manifest source digests missing')
        self.require('dirty_worktree' in doc and 'code_revision' in doc, 'manifest source provenance missing')

    def record(self, row, gen, identity):
        relative = row.get('predictions_file')
        self.artifact_digest(relative, row.get('predictions_sha256'))
        arrays = _npz(self.root, relative)
        required = {'embeddings', 'labels_true', 'labels_pred', 'domains', 'outlier_mask'}
        if not self.require(set(arrays) == required, 'prediction archive schema: ' + str(relative)): return None
        n = gen.get('num_entities', 10) * gen.get('observations_per_entity', 50)
        d = gen.get('embedding_dim', 192)
        x, t, p, domains, outliers = (arrays[key] for key in ('embeddings', 'labels_true', 'labels_pred', 'domains', 'outlier_mask'))
        shape_ok = x.shape == (n, d) and all(a.shape == (n,) for a in (t, p, domains, outliers))
        if not self.require(shape_ok, 'prediction array shape: ' + str(relative)): return None
        labels_ok = all(np.issubdtype(a.dtype, np.integer) for a in (t, p, domains)) and outliers.dtype == np.bool_
        if not self.require(labels_ok, 'prediction label/mask dtype: ' + str(relative)): return None
        self.require(np.all(t >= -1) and np.all(t < gen.get('num_entities', 10)) and np.all(p >= -1), 'prediction labels outside contract: ' + str(relative))
        self.require(np.isfinite(x).all() and np.allclose(np.linalg.norm(x, axis=1), 1., atol=1e-10), 'dataset vectors not finite unit vectors: ' + str(relative))
        self.require(set(domains.tolist()) == {0, 1}, 'dataset domains missing or invalid: ' + str(relative))
        self.require(np.array_equal(outliers, t == -1) and int(outliers.sum()) == int(n * gen.get('outliers_ratio', .05)), 'dataset outlier semantics: ' + str(relative))
        digest = _dataset_hash(arrays)
        self.require(row.get('dataset_sha256') == digest, 'dataset digest mismatch: ' + str(relative))
        if identity in self.dataset_keys:
            self.require(self.dataset_keys[identity] == digest, 'dataset pairing mismatch: ' + str(identity))
        else: self.dataset_keys[identity] = digest
        expected = reference_metrics(t.tolist(), p.tolist())
        expected.update(_supplemental_metrics(t.tolist(), p.tolist(), expected))
        metrics = row.get('metrics', {})
        self.require(isinstance(metrics, dict), 'metrics object missing: ' + str(relative))
        if isinstance(metrics, dict):
            # Core oracle fields are mandatory; supplemental legacy fields, if
            # reported, must also agree. Candidate-only unknown metrics fail.
            for key, value in expected.items():
                if key in reference_metrics([], []) or key in metrics:
                    self.require(key in metrics and _same(metrics.get(key), value), 'metric ' + key + ' mismatch: ' + str(relative))
            self.require(not (set(metrics) - set(expected)), 'unknown unverified metric: ' + str(relative))
        strata = row.get('stratified_metrics', {})
        for key, mask in [('domain_0', domains == 0), ('domain_1', domains == 1), ('outliers', outliers), ('known', ~outliers)]:
            if key in strata:
                oracle = reference_metrics(t[mask].tolist(), p[mask].tolist())
                for name, value in oracle.items():
                    self.require(_same(strata[key].get(name), value), 'stratified metric mismatch: ' + str(relative))
        self.require(row.get('status') == 'success', 'unsuccessful study row')
        self.require(isinstance(row.get('runtime_sec'), (float, int)) and not isinstance(row.get('runtime_sec'), bool) and math.isfinite(row['runtime_sec']) and row['runtime_sec'] >= 0, 'invalid descriptive runtime')
        self.predictions[relative] = (digest, p)
        return expected

    def clustering(self):
        spec = self.spec
        config = _json(self.root, 'config.json')
        self.require(config == spec, 'config differs from operator profile or has overlapping seeds')
        selected = _json(self.root, 'selected_config.json')
        self.artifact_digest('config.json', selected.get('config_sha256'))
        self.frozen = _digest(_file(self.root, 'selected_config.json'))
        self.require(selected.get('development_seeds') == spec['development_seeds'] and selected.get('evaluation_seeds') == spec['evaluation_seeds'], 'selection seed split mismatch')
        declared = selected.get('methods', {})
        if not isinstance(declared, dict): raise ValueError('selection methods must be an object')
        self.require(set(declared) == set(METHODS), 'selection method inventory mismatch')
        groups = {name: _json(self.root, name + '/results.json').get('results') for name in ('development', 'evaluation', 'ablation', 'sensitivity')}
        counts = {'development': 24 if self.profile == 'smoke' else 480, 'evaluation': 8 if self.profile == 'smoke' else 120,
                  'ablation': 12 if self.profile == 'smoke' else 180, 'sensitivity': 4 if self.profile == 'smoke' else 72}
        for name, rows in groups.items():
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows): raise ValueError('study rows must be arrays of objects')
            if len(rows) > counts[name]: raise ValueError('study cohort count exceeds operator resource policy')
            self.require(len(rows) == counts[name], name + ' count differs from operator profile')
            self.measurements[name + '_count'] = len(rows)
        composite = _json(self.root, 'results.json')
        for section, key in [('evaluation', 'results'), ('ablation', 'ablations'), ('sensitivity', 'sensitivity')]:
            self.require(composite.get(key) == groups[section], 'duplicate result envelope mismatch: ' + section)
        self.require(composite.get('selected_config_sha256') == self.frozen, 'frozen selection digest mismatch in result envelope')
        dev_groups = defaultdict(list); seen = set()
        for row in groups['development']:
            method, seed, scenario, candidate = (row.get(k) for k in ('method', 'seed', 'scenario_id', 'candidate'))
            key = (method, seed, scenario, candidate)
            if not self.require(method in METHODS and seed in spec['development_seeds'] and isinstance(scenario, int) and 0 <= scenario < len(spec['scenarios']) and isinstance(candidate, int) and 0 <= candidate < len(spec['threshold_grid']) and key not in seen, 'development cohort key invalid or duplicated'): continue
            seen.add(key); scenario_config = spec['scenarios'][scenario]
            self.require(row.get('scenario') == scenario_config and row.get('split') == 'development', 'development cohort metadata mismatch')
            self.require(row.get('parameters') == _params(method, spec['threshold_grid'][candidate]), 'development parameters differ from declared grid')
            gen = {**spec['generator'], **scenario_config, 'seed': seed}
            oracle = self.record(row, gen, json.dumps(gen, sort_keys=True))
            if oracle is not None: dev_groups[method, candidate].append(oracle)
        # Recompute selection using only independently checked development scores.
        for method in METHODS:
            ranks = []
            summaries = []
            for candidate, threshold in enumerate(spec['threshold_grid']):
                rows = dev_groups[method, candidate]
                valid = len(rows) == len(spec['development_seeds']) * len(spec['scenarios']) and all(r['false_discovery_proportion'] is not None for r in rows)
                f1 = float(np.mean([r['pairwise_f1'] or 0 for r in rows])) if rows else 0.
                fdp = float(np.mean([r['false_discovery_proportion'] for r in rows])) if valid else None
                eligible = valid and fdp <= .01
                rank = (int(eligible), f1 if eligible else -(fdp if valid else 2), f1, -candidate)
                ranks.append(rank); summaries.append({'parameters': _params(method, threshold), 'mean_pairwise_f1': f1, 'mean_fdp': fdp, 'eligible': eligible})
            chosen = max(range(len(ranks)), key=lambda i: ranks[i])
            self.require(declared.get(method) == _params(method, spec['threshold_grid'][chosen]), 'selection does not maximize trusted development objective: ' + method)
            reported = selected.get('scores', {}).get(method, [])
            self.require(len(reported) == len(summaries), 'selection candidate summaries missing: ' + method)
            for report, summary in zip(reported, summaries):
                self.require(all(key in report and _same(report[key], value) for key, value in summary.items()), 'selection candidate summary mismatch: ' + method)
        for section in ('evaluation', 'ablation', 'sensitivity'):
            seen = set()
            for row in groups[section]:
                method, seed = row.get('method'), row.get('seed')
                if not self.require(method in METHODS and seed in spec['evaluation_seeds'], section + ' held-out key invalid'): continue
                gen = {**spec['generator'], 'seed': seed}; expected_params = declared.get(method, {})
                if section != 'sensitivity':
                    scenario = row.get('scenario_id')
                    if not self.require(isinstance(scenario, int) and 0 <= scenario < len(spec['scenarios']), 'evaluation scenario invalid'): continue
                    gen.update(spec['scenarios'][scenario])
                    self.require(row.get('scenario') == spec['scenarios'][scenario] and row.get('split') == 'evaluation', 'evaluation cohort metadata mismatch')
                    key = (method, seed, scenario)
                    if section == 'ablation':
                        variant = row.get('variant'); key += (variant,)
                        if not self.require(method == 'risk_aware' and variant in TOGGLES, 'ablation variant invalid'): continue
                        expected_params = dict(declared.get('risk_aware', {}))
                        if TOGGLES[variant]: expected_params[TOGGLES[variant]] = False
                else:
                    parameter, value = row.get('parameter'), row.get('value'); key = (seed, parameter, value)
                    if not self.require(method == 'risk_aware' and parameter in spec['sensitivity_grid'] and value in spec['sensitivity_grid'][parameter], 'sensitivity factor invalid'): continue
                    fields = {'cluster_population': 'num_entities', 'noise': 'noise_level', 'dimensionality': 'embedding_dim', 'overlap': 'inter_entity_similarity', 'domain_shift': 'domain_shift', 'outlier_rate': 'outliers_ratio'}
                    expected_params = dict(declared.get('risk_aware', {}))
                    if parameter in fields: gen[fields[parameter]] = value
                    elif parameter == 'neighborhood_size': expected_params['max_neighbors'] = value
                    else: expected_params.update(abstention_threshold=value, reciprocal_threshold=value)
                    self.require(row.get('generator_parameters') == gen and row.get('split') == 'evaluation_sensitivity', 'sensitivity generator metadata mismatch')
                self.require(key not in seen, section + ' duplicate cohort key'); seen.add(key)
                self.require(row.get('parameters') == expected_params, section + ' parameters differ from frozen selection/one component toggle')
                self.require(row.get('selected_config_sha256') == self.frozen, section + ' stale frozen selection digest')
                self.record(row, gen, json.dumps(gen, sort_keys=True))
        # A full-system ablation must reproduce the held-out risk predictions.
        eval_risk = {(r['seed'], r['scenario_id']): r for r in groups['evaluation'] if r.get('method') == 'risk_aware'}
        for row in groups['ablation']:
            if row.get('variant') == 'full_system':
                base = eval_risk.get((row['seed'], row['scenario_id']))
                if base and row['predictions_file'] in self.predictions and base['predictions_file'] in self.predictions:
                    a, b = self.predictions[row['predictions_file']], self.predictions[base['predictions_file']]
                    self.require(a[0] == b[0] and np.array_equal(a[1], b[1]), 'full-system ablation prediction mismatch')

    def retrieval(self):
        doc = _json(self.root, 'scalability/results.json')
        rows = doc.get('benchmarks'); sizes = [1000] if self.profile == 'smoke' else [1000, 10000, 100000, 200000]
        if not self.require(isinstance(rows, list) and len(rows) == len(sizes), 'retrieval benchmark count mismatch'): return []
        self.require(doc.get('skipped') == [], 'required retrieval sizes were skipped')
        self.require([r.get('size') for r in rows] == sizes, 'retrieval sizes differ from operator profile')
        for row, size in zip(rows, sizes):
            qn = 16 if self.profile == 'smoke' else 64
            self.require(all(row.get(key) == value for key, value in {'size': size, 'dimension': 192, 'query_count': qn, 'k': 10, 'seed': 712, 'distance': 'squared_euclidean', 'algorithm': 'projection_window_candidates_exact_rerank', 'num_projections': 8, 'window': 256}.items()), 'retrieval configuration mismatch')
            paths = {}
            for name in ('vectors', 'queries', 'neighbors'):
                relative = 'scalability/' + row[name + '_file']; self.artifact_digest(relative, row.get(name + '_sha256')); paths[name] = relative
            x = np.load(_file(self.root, paths['vectors']), allow_pickle=False, mmap_mode='r')
            q = np.load(_file(self.root, paths['queries']), allow_pickle=False, mmap_mode='r')
            if not self.require(x.shape == (size, 192) and q.shape == (qn, 192) and x.dtype == np.float64 and q.dtype == np.float64, 'retrieval input shape/dtype mismatch'): continue
            # Independently confirm the declared seeded benchmark inputs in
            # bounded batches, without duplicating a full large index in RAM.
            rng = np.random.default_rng(712)
            for offset in range(0, size, 8192):
                generated = rng.normal(size=(min(8192, size-offset), 192)); generated /= np.linalg.norm(generated, axis=1, keepdims=True)
                self.require(np.allclose(x[offset:offset+len(generated)], generated, rtol=0, atol=1e-12), 'retrieval vectors do not match seeded protocol')
            generated = rng.normal(size=(qn, 192)); generated /= np.linalg.norm(generated, axis=1, keepdims=True)
            self.require(np.allclose(q, generated, rtol=0, atol=1e-12), 'retrieval queries do not match independent seeded protocol')
            result = _npz(self.root, paths['neighbors'])
            if not self.require(set(result) == {'exact_ids', 'exact_distances', 'approximate_ids', 'approximate_distances'}, 'neighbor archive schema mismatch'): continue
            valid = True
            for mode in ('exact', 'approximate'):
                ids, ds = result[mode + '_ids'], result[mode + '_distances']
                shape_ok = ids.shape == ds.shape == (qn, 10) and np.issubdtype(ids.dtype, np.integer)
                if not self.require(shape_ok, 'neighbor array shape/dtype mismatch'): valid = False; continue
                if not self.require(np.all((ids >= 0) & (ids < size)) and all(len(set(row)) == 10 for row in ids), 'neighbor ids invalid or duplicated'): valid = False; continue
                expected = np.sum((q[:, None, :] - x[ids]) ** 2, axis=2)
                self.require(np.isfinite(ds).all() and np.all(ds >= 0) and np.allclose(ds, expected, rtol=1e-8, atol=1e-8), mode + ' neighbor distances not derived from query-specific vectors')
                self.require(np.all(ds[:, 1:] >= ds[:, :-1]), mode + ' neighbor distances are unsorted')
            if not valid: continue
            ids, ds = _exact_neighbors(x, q, 10)
            self.require(np.array_equal(result['exact_ids'], ids) and np.allclose(result['exact_distances'], ds, atol=1e-8, rtol=1e-8), 'exact neighbor reference mismatch')
            recall = sum(len(set(a) & set(b)) for a, b in zip(ids, result['approximate_ids'])) / (qn * 10)
            self.require(_same(row.get('recall_at_k'), recall), 'retrieval recall does not match measured neighbors')
            self.measurements['recall_at_' + str(size)] = recall
            for name in ('generation_sec', 'exact_query_sec', 'approximate_build_sec', 'approximate_query_sec', 'process_sampled_peak_rss_mb'):
                value = row.get(name)
                self.require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0, 'invalid descriptive retrieval resource field: ' + name)
        return rows

    def claims(self, rows):
        payload = _json(self.root, 'claims.json'); claims = payload.get('claims')
        if not self.require(isinstance(claims, list) and all(isinstance(c, dict) for c in claims), 'claims array of objects missing'): return
        expected = {'evaluation-count': ('evaluation_count', 'evaluation/results.json', 'artifact_summary'),
                    'ablation-count': ('ablation_count', 'ablation/results.json', 'artifact_summary')}
        for row in rows:
            expected['recall-' + str(row['size'])] = ('recall_at_' + str(row['size']), 'scalability/' + row['neighbors_file'], 'retrieval_recall_at_k')
        self.require({c.get('id') for c in claims} == set(expected) and len(claims) == len(expected), 'claim inventory differs from supported measurements')
        for claim in claims:
            identifier = claim.get('id')
            if identifier not in expected: continue
            metric, source, kind = expected[identifier]
            self.require(claim.get('kind') == kind and claim.get('source') == source, 'claim source or kind mismatch: ' + identifier)
            self.artifact_digest(source, claim.get('source_sha256'))
            value = self.measurements.get(metric)
            self.require(metric in self.measurements and _same(claim.get('value'), value), 'claim value unsupported by trusted recomputation: ' + identifier)
            if kind == 'retrieval_recall_at_k':
                benchmark = next(r for r in rows if 'recall-' + str(r['size']) == identifier)
                self.require(all(claim.get(k) == benchmark[k] for k in ('size', 'k', 'query_count')), 'claim retrieval protocol mismatch: ' + identifier)
            self.claim_links.append({'claim_id': identifier, 'metric_id': metric, 'value': value})


def validate_study_directory(path, profile='smoke'):
    """Return failures, checks, trusted numeric measurements, and claim links.

    Rejection is data, never a reason to import/execute candidate modules.
    Resource exhaustion is classified by the enclosing evaluator.
    """
    validator = None
    try:
        validator = _Validator(path, profile)
        for name, function in [('manifest', validator.manifest), ('clustering', validator.clustering), ('retrieval', validator.retrieval)]:
            before = len(validator.failures)
            result = function()
            validator.checks.append({'id': 'study-' + name, 'status': 'REJECTED' if len(validator.failures) > before else 'ACCEPTED'})
            if name == 'retrieval':
                before = len(validator.failures)
                validator.claims(result)
                validator.checks.append({'id': 'study-claims', 'status': 'REJECTED' if len(validator.failures) > before else 'ACCEPTED'})
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, zipfile.BadZipFile) as exc:
        if validator is None:
            return {'failures': ['study artifact validation: ' + str(exc)], 'checks': [], 'measurements': {}, 'claim_links': []}
        validator.failures.append('study artifact validation: ' + str(exc))
        validator.checks.append({'id': 'study-schema', 'status': 'REJECTED'})
    return {'failures': validator.failures[:200], 'checks': validator.checks,
            'measurements': validator.measurements, 'claim_links': validator.claim_links if not validator.failures else []}


def compare_studies(submitted, replica):
    """Compare semantic inputs, frozen parameters and outputs; ignore timing.

    Both directories must separately pass validate_study_directory. This step
    binds predictions to independent isolated reproduction of the candidate.
    """
    failures = []
    a, b = Path(submitted).resolve(), Path(replica).resolve()
    try:
        if _json(a, 'config.json') != _json(b, 'config.json'): failures.append('replica config differs')
        sa, sb = _json(a, 'selected_config.json'), _json(b, 'selected_config.json')
        for key in ('methods', 'scores', 'development_seeds', 'evaluation_seeds', 'maximum_development_fdp'):
            if sa.get(key) != sb.get(key): failures.append('replica frozen selection differs: ' + key)
        for section in ('development', 'evaluation', 'ablation', 'sensitivity'):
            ra, rb = _json(a, section + '/results.json')['results'], _json(b, section + '/results.json')['results']
            if len(ra) != len(rb): failures.append('replica row count differs: ' + section); continue
            for first, second in zip(ra, rb):
                for key in ('method', 'parameters', 'seed', 'scenario_id', 'scenario', 'candidate', 'variant', 'parameter', 'value', 'generator_parameters', 'dataset_sha256', 'metrics'):
                    if first.get(key) != second.get(key): failures.append('replica semantic row differs: ' + section + '/' + key)
                left = _npz(a, first['predictions_file']); right = _npz(b, second['predictions_file'])
                if set(left) != set(right) or any(not np.array_equal(left[key], right[key]) for key in left):
                    failures.append('replica prediction/input arrays differ: ' + first['predictions_file'])
        ra, rb = _json(a, 'scalability/results.json')['benchmarks'], _json(b, 'scalability/results.json')['benchmarks']
        if len(ra) != len(rb): failures.append('replica retrieval count differs')
        for first, second in zip(ra, rb):
            for key in ('size', 'dimension', 'query_count', 'k', 'seed', 'algorithm', 'num_projections', 'window', 'distance', 'recall_at_k'):
                if first.get(key) != second.get(key): failures.append('replica retrieval protocol/result differs: ' + key)
            for name in ('vectors', 'queries'):
                if _digest(_file(a, 'scalability/' + first[name + '_file'])) != _digest(_file(b, 'scalability/' + second[name + '_file'])):
                    failures.append('replica retrieval input differs: ' + name)
            left = _npz(a, 'scalability/' + first['neighbors_file']); right = _npz(b, 'scalability/' + second['neighbors_file'])
            if set(left) != set(right) or any(not np.array_equal(left[key], right[key]) for key in left): failures.append('replica neighbor arrays differ')
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, zipfile.BadZipFile) as exc:
        failures.append('replica comparison: ' + str(exc))
    return failures[:200]
