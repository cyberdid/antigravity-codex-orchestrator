"""Independent acceptance regressions: weak artifacts must not become science."""

import json
import math

import pytest


def test_reference_oracle_penalizes_merging_distinct_identities():
    from antigravity_hybrid_orchestrator.integrity.oracle import reference_metrics
    result = reference_metrics([0, 0, 1, 1], [7, 7, 7, 7])
    assert result['pairwise_precision'] == pytest.approx(1 / 3)
    assert result['pairwise_recall'] == 1
    assert result['b_cubed_precision'] == .5
    assert result['b_cubed_recall'] == 1
    assert result['coverage'] == 1


def test_reference_oracle_does_not_call_zero_decisions_perfect_precision():
    from antigravity_hybrid_orchestrator.integrity.oracle import reference_metrics
    result = reference_metrics([0, 0, 1, 1], [-1, -1, -1, -1])
    assert result['pairwise_precision'] is None
    assert result['coverage'] == 0
    assert result['pairwise_recall'] == 0


def test_metric_comparison_rejects_orion1_constant_metric_shortcut():
    from antigravity_hybrid_orchestrator.integrity.oracle import check_metric_results, metric_cases
    cases = metric_cases(seed=23591)
    fake = [{'pairwise_precision': 1, 'pairwise_recall': 1,
             'b_cubed_precision': 1, 'b_cubed_recall': 1, 'coverage': 1} for _ in cases]
    failures = check_metric_results(cases, fake)
    assert failures
    assert any('coverage' in failure or 'precision' in failure for failure in failures)


def test_retrieval_rejects_ids_unrelated_to_query_and_nan_distances():
    from antigravity_hybrid_orchestrator.integrity.oracle import check_neighbors
    vectors = [[1., 0.], [0., 1.], [-1., 0.]]
    queries = [[-1., 0.], [0., 1.]]
    assert check_neighbors(vectors, queries, [[0], [0]], [[0.], [math.nan]], 1, exact=True)
    assert not check_neighbors(vectors, queries, [[2], [1]], [[0.], [0.]], 1, exact=True)


def test_evidence_is_owner_only_chained_and_bound_to_artifact(tmp_path):
    from antigravity_hybrid_orchestrator.integrity.evidence import EvidenceStore
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    (candidate / 'result.json').write_text('{"value":2}')
    store = EvidenceStore(tmp_path / 'trusted', candidate)
    first = store.append({'status': 'ACCEPTED', 'artifacts': store.hash_files(['result.json'])})
    second = store.append({'status': 'REJECTED'})
    assert first['evidence_id'] != second['evidence_id']
    assert second['previous_digest'] == first['record_digest']
    assert (store.directory / (first['evidence_id'] + '.json')).stat().st_mode & 0o777 == 0o600
    assert store.validate(first['evidence_id'])
    (candidate / 'result.json').write_text('{"value":3}')
    assert not store.validate(first['evidence_id'])


def test_evidence_store_cannot_be_inside_candidate_or_follow_external_symlink(tmp_path):
    from antigravity_hybrid_orchestrator.integrity.evidence import EvidenceStore
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    with pytest.raises(ValueError):
        EvidenceStore(candidate / 'trusted', candidate)
    outside = tmp_path / 'hidden'
    outside.write_text('private')
    (candidate / 'escape').symlink_to(outside)
    with pytest.raises(ValueError):
        EvidenceStore(tmp_path / 'trusted', candidate).hash_files(['escape'])


def test_claims_require_measured_evidence_and_current_artifact_hashes(tmp_path):
    from antigravity_hybrid_orchestrator.integrity.evidence import EvidenceStore, validate_claims
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    (candidate / 'neighbors.json').write_text('[]')
    store = EvidenceStore(tmp_path / 'trusted', candidate)
    evidence = store.append({'status': 'ACCEPTED', 'measurements': {'recall_at_k': .5},
                             'artifacts': store.hash_files(['neighbors.json'])})
    claim = {'claim_id':'C1', 'evidence_ids':[evidence['evidence_id']],
             'metric_id':'recall_at_k', 'operator':'>=', 'value':.9}
    assert validate_claims([claim], store)
    claim['value'] = .4
    assert validate_claims([claim], store) == []
    (candidate / 'neighbors.json').write_text('[1]')
    assert validate_claims([claim], store)


def test_isolated_evaluator_rejects_constant_metrics_and_records_failure(tmp_path):
    from antigravity_hybrid_orchestrator.integrity.evaluator import ResearchIntegrityEvaluator
    candidate = tmp_path / 'candidate'
    (candidate / 'src/orion').mkdir(parents=True)
    (candidate / 'src/orion/metrics.py').write_text(
        'def calculate_metrics(a,b):\n'
        '    return {"pairwise_precision":1,"coverage":1}\n')
    evaluator = ResearchIntegrityEvaluator(candidate, tmp_path / 'trusted')
    decision = evaluator.evaluate('metrics-v1')
    assert decision.status == 'REJECTED'
    assert decision.evidence_id
    record = evaluator.store.get(decision.evidence_id)
    assert record['execution']['isolated'] is True
    assert record['execution']['exit_code'] == 0
    assert record['checks']


def test_isolated_evaluator_rejects_fake_retrieval(tmp_path):
    from antigravity_hybrid_orchestrator.integrity.evaluator import ResearchIntegrityEvaluator
    candidate = tmp_path / 'candidate'
    (candidate / 'src/orion').mkdir(parents=True)
    (candidate / 'src/orion/retrieval.py').write_text(
        'import numpy as np\n'
        'def exact_knn(x,q,k):\n'
        '    return np.tile(np.arange(k),(len(q),1)),np.zeros((len(q),k))\n'
        'approximate_knn=exact_knn\n')
    decision = ResearchIntegrityEvaluator(candidate, tmp_path / 'trusted').evaluate('retrieval-v1')
    assert decision.status == 'REJECTED'


def test_evaluator_accepts_corrected_orion_source_in_real_sandbox(tmp_path):
    from pathlib import Path
    from antigravity_hybrid_orchestrator.integrity.evaluator import ResearchIntegrityEvaluator
    candidate = Path(__file__).resolve().parents[1] / 'temp/project_orion'
    evaluator = ResearchIntegrityEvaluator(candidate, tmp_path / 'trusted')
    decision = evaluator.evaluate('orion-core-v1')
    assert decision.status == 'ACCEPTED', decision.feedback
    assert decision.verification_level == 'scientific'


def test_new_importable_source_invalidates_previously_accepted_evidence(tmp_path):
    from antigravity_hybrid_orchestrator.integrity.evidence import EvidenceStore
    from antigravity_hybrid_orchestrator.integrity.evaluator import ResearchIntegrityEvaluator
    workspace = tmp_path / 'candidate'
    workspace.mkdir()
    (workspace / 'a.py').write_text('value=1')
    evaluator = ResearchIntegrityEvaluator(workspace, tmp_path / 'trusted')
    hashes = evaluator._copy_source(tmp_path / 'snapshot')
    record = evaluator.store.append({'status':'ACCEPTED','source_hashes':hashes})
    assert evaluator.store.validate(record['evidence_id'])
    (workspace / 'sitecustomize.py').write_text('print("changed behavior")')
    assert not evaluator.store.validate(record['evidence_id'])


def test_snapshot_hashes_the_bytes_actually_copied(tmp_path, monkeypatch):
    import hashlib
    import shutil
    from antigravity_hybrid_orchestrator.integrity.evaluator import ResearchIntegrityEvaluator
    workspace = tmp_path / 'candidate'
    workspace.mkdir()
    source = workspace / 'a.py'
    source.write_text('value=1')
    real_copy = shutil.copyfile
    def concurrent_edit(src, dst):
        real_copy(src, dst)
        source.write_text('value=2')
    monkeypatch.setattr(shutil, 'copyfile', concurrent_edit)
    hashes = ResearchIntegrityEvaluator(workspace, tmp_path/'trusted')._copy_source(tmp_path/'snapshot')
    assert hashes['a.py'] == hashlib.sha256(b'value=1').hexdigest()


def test_metric_oracle_checks_bcubed_f1_too():
    from antigravity_hybrid_orchestrator.integrity.oracle import reference_metrics, check_metric_results
    cases = [{'truth':[0,0,1,1], 'prediction':[0,0,0,0]}]
    result = reference_metrics(cases[0]['truth'], cases[0]['prediction'])
    result['b_cubed_f1'] = 999
    assert check_metric_results(cases, [result])


def test_nested_malformed_candidate_envelope_produces_a_recorded_rejection(tmp_path):
    from antigravity_hybrid_orchestrator.integrity.evaluator import ResearchIntegrityEvaluator
    candidate = tmp_path/'candidate'
    (candidate/'src/orion').mkdir(parents=True)
    (candidate/'src/orion/retrieval.py').write_text(
        'import os\n'
        'open("/output/result.json","w").write(\'{"exact":[],"approximate":[]}\')\n'
        'os._exit(0)\n')
    decision = ResearchIntegrityEvaluator(candidate,tmp_path/'trusted').evaluate('retrieval-v1')
    assert decision.status == 'REJECTED'
    assert decision.evidence_id


def test_report_does_not_gain_scientific_acceptance_from_unrelated_metric_oracle(tmp_path):
    from pathlib import Path
    from antigravity_hybrid_orchestrator.integrity.evaluator import ResearchIntegrityEvaluator
    from antigravity_hybrid_orchestrator.planner.dag import SubTask
    import shutil
    source = Path(__file__).resolve().parents[1]/'temp/project_orion'
    candidate = tmp_path/'candidate'
    shutil.copytree(source/'src', candidate/'src')
    (candidate/'reports').mkdir()
    (candidate/'reports/research_report.md').write_text('Empirical recall=1')
    decision = ResearchIntegrityEvaluator(candidate,tmp_path/'trusted').evaluate_subtask(
        SubTask(id='report',outputs=['reports/research_report.md']), 'metrics-v1')
    assert decision['status'] != 'ACCEPTED'


def test_later_operational_failure_does_not_erase_known_scientific_rejection(tmp_path,monkeypatch):
    from antigravity_hybrid_orchestrator.integrity.evaluator import ResearchIntegrityEvaluator
    candidate=tmp_path/'candidate'
    candidate.mkdir()
    evaluator=ResearchIntegrityEvaluator(candidate,tmp_path/'trusted')
    def responses(request):
        if request['kind']=='metrics':
            return [], {'isolated':True,'exit_code':0}, {}, ''
        raise OSError('daemon unavailable')
    monkeypatch.setattr(evaluator,'_invoke',responses)
    assert evaluator.evaluate('orion-core-v1').status == 'REJECTED'


def test_valid_claim_json_does_not_certify_unchecked_narrative(tmp_path):
    from antigravity_hybrid_orchestrator.integrity.evaluator import ResearchIntegrityEvaluator
    from antigravity_hybrid_orchestrator.planner.dag import SubTask
    workspace=tmp_path/'candidate'
    (workspace/'reports').mkdir(parents=True)
    evaluator=ResearchIntegrityEvaluator(workspace,tmp_path/'trusted')
    evidence=evaluator.store.append({'status':'ACCEPTED','measurements':{'recall_at_k':.4}})
    claims=[{'claim_id':'C1','metric_id':'recall_at_k','operator':'==','value':.4,
             'evidence_ids':[evidence['evidence_id']]}]
    (workspace/'reports/claims.json').write_text(json.dumps({'claims':claims}))
    (workspace/'reports/research_report.md').write_text('Measured recall was 100%, the algorithm is perfect.')
    decision=evaluator.evaluate_subtask(SubTask(id='report',outputs=['reports/research_report.md']),'metrics-v1')
    assert decision['status'] == 'REJECTED'


def test_narrative_report_with_matching_cited_claims_is_accepted(tmp_path):
    from antigravity_hybrid_orchestrator.integrity.evaluator import ResearchIntegrityEvaluator
    from antigravity_hybrid_orchestrator.planner.dag import SubTask
    workspace=tmp_path/'candidate'
    (workspace/'reports').mkdir(parents=True)
    evaluator=ResearchIntegrityEvaluator(workspace,tmp_path/'trusted')
    evidence=evaluator.store.append({'status':'ACCEPTED','measurements':{'recall_at_k':.4}})
    claims=[{'claim_id':'C1','metric_id':'recall_at_k','operator':'==','value':.4,
             'evidence_ids':[evidence['evidence_id']]}]
    (workspace/'reports/claims.json').write_text(json.dumps({'claims':claims}))
    (workspace/'reports/research_report.md').write_text(
        f'Claim C1 (evidence {evidence["evidence_id"]}): measured recall_at_k is 0.4 (40%).'
    )
    decision=evaluator.evaluate_subtask(SubTask(id='report',outputs=['reports/research_report.md']),'metrics-v1')
    assert decision['status'] == 'ACCEPTED'

