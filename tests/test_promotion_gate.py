"""A zero-exit shell verifier cannot bypass controller acceptance."""

from antigravity_hybrid_orchestrator.engine.loop import SelfCorrectionLoop
from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG, SubTask
from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite
from antigravity_hybrid_orchestrator.workforce.worker import SubtaskResult


class ReportsSuccess:
    def execute_subtask(self, subtask, **kwargs):
        return SubtaskResult(subtask.id, success=True)


def test_rejected_independent_gate_prevents_completed_status(tmp_path, monkeypatch):
    monkeypatch.setenv('RD_MAX_REPAIR_ATTEMPTS', '1')
    dag = ExecutionPlanDAG()
    dag.add_task(SubTask(id='science', verification_command='true'))
    gate = lambda task: {'status':'REJECTED', 'verification_level':'scientific',
                         'feedback':'constant metrics', 'evidence_id':'known-bad'}
    loop = SelfCorrectionLoop(WorkspaceToolSuite(tmp_path), ReportsSuccess(),
                              checkpoint_git=False, acceptance_gate=gate)
    result = loop.execute_dag(dag)
    assert not result.success
    assert dag.tasks['science'].status != 'completed'


def test_accepted_gate_evidence_is_bound_to_verification(tmp_path):
    dag = ExecutionPlanDAG()
    dag.add_task(SubTask(id='science', verification_command='true'))
    gate = lambda task: {'status':'ACCEPTED', 'verification_level':'scientific',
                         'feedback':'', 'evidence_id':'controller-proof'}
    result = SelfCorrectionLoop(WorkspaceToolSuite(tmp_path), ReportsSuccess(),
                               checkpoint_git=False, acceptance_gate=gate).execute_dag(dag)
    assert result.success
    assert result.subtask_results['science'].metadata['verification']['metadata']['integrity']['evidence_id'] == 'controller-proof'


def test_cli_live_defaults_to_docker_and_host_requires_explicit_option():
    from antigravity_hybrid_orchestrator.cli import build_parser
    parser = build_parser()
    assert parser.parse_args(['--goal','x']).sandbox == 'docker'
    assert parser.parse_args(['--goal','x','--sandbox','host']).sandbox == 'host'


def test_gauntlet_invalid_candidates_never_get_reference_patch(tmp_path, monkeypatch):
    from antigravity_hybrid_orchestrator.coordinator import HybridCoordinator
    from antigravity_hybrid_orchestrator.workforce import PatchResult
    coord = HybridCoordinator(mode='mock',target='auth',no_hud=True)
    coord.gauntlet_dir = tmp_path
    original = 'def original():\n    return 17\n'
    (tmp_path/'auth.py').write_text(original)
    for builder in (coord.builder_a, coord.builder_b):
        monkeypatch.setattr(builder, 'generate_patch', lambda **kwargs: PatchResult(builder_name='Builder A',patch_code='not python ??'))
    assert coord.run() == 1
    assert (tmp_path/'auth.py').read_text() == original
