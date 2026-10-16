"""Study acceptance requires actual artifacts plus an isolated replication."""
from pathlib import Path
import shutil


def candidate_source(tmp_path):
    source = Path(__file__).resolve().parents[1]/'temp/project_orion'
    candidate = tmp_path/'candidate'
    candidate.mkdir()
    for name in ('src','scripts','configs'):
        shutil.copytree(source/name,candidate/name)
    for name in ('pyproject.toml','README.md'):
        shutil.copy2(source/name,candidate/name)
    return candidate


def test_study_cannot_accept_missing_experiments_even_when_core_is_correct(tmp_path):
    from antigravity_hybrid_orchestrator.integrity.evaluator import ResearchIntegrityEvaluator
    evaluator=ResearchIntegrityEvaluator(candidate_source(tmp_path),tmp_path/'trusted',study_profile='smoke')
    result=evaluator.evaluate('orion-study-v1')
    assert result.status == 'REJECTED'
    assert result.evidence_id


def test_study_acceptance_has_independent_replication_and_claim_evidence(tmp_path):
    from antigravity_hybrid_orchestrator.integrity.evaluator import ResearchIntegrityEvaluator
    candidate=candidate_source(tmp_path)
    evaluator=ResearchIntegrityEvaluator(candidate,tmp_path/'trusted',study_profile='smoke')
    output=tmp_path/'initial-execution'
    output.mkdir()
    result,execution,hashes,error=evaluator._invoke({'kind':'study','profile':'smoke'},retained_output=output)
    assert not error
    shutil.copytree(output/'study',candidate/'orion2-run')
    decision=evaluator.evaluate('orion-study-v1')
    assert decision.status == 'ACCEPTED',decision.feedback
    record=evaluator.store.get(decision.evidence_id)
    assert record['execution']['isolated']
    assert record['replica_hashes']
    assert record['claim_links']
    assert evaluator.store.validate(decision.evidence_id)
