import json

import pytest


def test_resume_policy_cannot_be_downgraded_by_candidate_sidecar(tmp_path, monkeypatch):
    from antigravity_hybrid_orchestrator.integrity.policy import RunPolicyStore
    monkeypatch.setenv('ANTIGRAVITY_STATE_HOME', str(tmp_path/'controller'))
    workspace = tmp_path/'candidate'
    workspace.mkdir()
    log = workspace/'run.jsonl'
    policy = {'workspace':str(workspace),'goal':'science','integrity_contract':'orion-core-v1',
              'evidence_directory':str(tmp_path/'evidence'), 'sandbox_backend':'docker'}
    RunPolicyStore.save(log,policy)
    log.write_text('{"integrity_contract":null}')
    loaded = RunPolicyStore.load(log)
    assert loaded['integrity_contract'] == 'orion-core-v1'
    assert loaded['sandbox_backend'] == 'docker'


def test_fresh_plan_cannot_select_own_integrity_policy(tmp_path):
    from antigravity_hybrid_orchestrator.coordinator import HybridCoordinator
    from antigravity_hybrid_orchestrator.architect import MockCloudArchitectClient
    from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG,SubTask
    dag=ExecutionPlanDAG()
    dag.add_task(SubTask(id='x',verification_command='true',status='completed'))
    dag.metadata.update(integrity_contract='metrics-v1',evidence_directory=str(tmp_path/'host-victim'))
    coord=HybridCoordinator(mode='mock',goal='x',workspace=str(tmp_path/'candidate'),no_hud=True)
    coord.architect=MockCloudArchitectClient(custom_plan=dag)
    assert coord.run() == 0
    assert coord.integrity_contract is None
    assert not (tmp_path/'host-victim').exists()
    assert dag.tasks['x'].status == 'completed'


def test_symlinked_run_directory_cannot_redirect_controller_policy(tmp_path,monkeypatch):
    from antigravity_hybrid_orchestrator.integrity.policy import RunPolicyStore
    monkeypatch.setenv('ANTIGRAVITY_STATE_HOME',str(tmp_path/'controller'))
    workspace=tmp_path/'candidate'
    original=workspace/'.antigravity_runs'
    original.mkdir(parents=True)
    log=original/'run.jsonl'
    RunPolicyStore.save(log,{'workspace':str(workspace),'integrity_contract':'metrics-v1'})
    original.rename(workspace/'old-runs')
    evil=workspace/'.evil'
    evil.mkdir()
    original.symlink_to(evil,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        RunPolicyStore.load(log)
