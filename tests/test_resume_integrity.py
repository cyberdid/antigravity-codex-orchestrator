import json
import subprocess

import pytest

from antigravity_hybrid_orchestrator.engine.run_state import RunStateStore
from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG, SubTask


def setup_state(root):
    for name in ('input', 'source', 'output', 'other'):
        (root / name).write_text(name)
    dag = ExecutionPlanDAG(tasks=[
        SubTask(id='a', status='completed', input_files=['input'], affected_files=['source'], outputs=['output']),
        SubTask(id='b', status='completed', dependencies=['a'], outputs=['other']),
        SubTask(id='c', status='completed'),
    ])
    path = root / 'state.json'
    RunStateStore.save(path, goal='g', workspace=str(root), dag=dag)
    return path, dag


def statuses(path):
    return {t['id']: t['status'] for t in RunStateStore.load(path)['dag']['tasks']}


@pytest.mark.parametrize('name', ['input', 'source', 'output'])
@pytest.mark.parametrize('delete', [False, True])
def test_mutation_invalidates_task_and_downstream(tmp_path, name, delete):
    path, _ = setup_state(tmp_path)
    (tmp_path / name).unlink() if delete else (tmp_path / name).write_text('changed')
    assert statuses(path) == {'a': 'pending', 'b': 'pending', 'c': 'completed'}


def test_unchanged_and_checkpoint_head_keep_completion(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    path, _ = setup_state(tmp_path)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=T', '-c', 'user.email=t@t', 'commit', '--allow-empty', '-qm', 'checkpoint'], check=True)
    assert set(statuses(path).values()) == {'completed'}


def test_later_save_does_not_bless_changed_completion(tmp_path):
    path, dag = setup_state(tmp_path)
    (tmp_path / 'source').write_text('changed')
    RunStateStore.save(path, goal='g', workspace=str(tmp_path), dag=dag)
    assert statuses(path)['a'] == 'pending'


def test_legacy_completions_require_revalidation(tmp_path):
    path, _ = setup_state(tmp_path)
    state = json.loads(path.read_text())
    RunStateStore.trusted_path_for(path).unlink()
    state['version'] = 1
    state['dag']['metadata'].pop('task_fingerprints', None)
    path.write_text(json.dumps(state))
    assert set(statuses(path).values()) == {'pending'}


def test_escaping_symlink_is_not_hashed(tmp_path):
    root = tmp_path / 'workspace'
    root.mkdir()
    (tmp_path / 'secret').write_text('secret')
    (root / 'escape').symlink_to(tmp_path / 'secret')
    dag = ExecutionPlanDAG(tasks=[SubTask(id='a', status='completed', outputs=['escape'])])
    path = root / 'state.json'
    RunStateStore.save(path, goal='g', workspace=str(root), dag=dag)
    assert statuses(path)['a'] == 'pending'
    assert 'secret' not in path.read_text()


def test_fresh_verification_can_replace_previous_binding(tmp_path):
    path, dag = setup_state(tmp_path)
    (tmp_path / 'source').write_text('verified replacement')
    RunStateStore.record_completed(str(tmp_path), dag, 'a')
    RunStateStore.save(path, goal='g', workspace=str(tmp_path), dag=dag)
    assert statuses(path)['a'] == 'completed'


def test_changed_task_contract_invalidates_completion(tmp_path):
    path, dag = setup_state(tmp_path)
    dag['a'].verification_command = 'true'
    RunStateStore.save(path, goal='g', workspace=str(tmp_path), dag=dag)
    assert statuses(path)['a'] == 'pending'


def test_parent_symlink_escape_invalidates_completion(tmp_path):
    root = tmp_path / 'workspace'
    root.mkdir()
    (tmp_path / 'artifact').write_text('outside')
    (root / 'linked').symlink_to(tmp_path, target_is_directory=True)
    dag = ExecutionPlanDAG(tasks=[SubTask(id='a', status='completed', outputs=['linked/artifact'])])
    path = root / 'state.json'
    RunStateStore.save(path, goal='g', workspace=str(root), dag=dag)
    assert statuses(path)['a'] == 'pending'


def test_nonregular_fingerprint_path_does_not_leak_descriptors(tmp_path):
    import os
    directory = tmp_path / 'directory'
    directory.mkdir()
    count = len(os.listdir('/proc/self/fd'))
    dag = ExecutionPlanDAG(tasks=[SubTask(id='a', status='completed', outputs=['directory'])])
    for _ in range(10):
        assert not RunStateStore.bind_task(str(tmp_path), dag['a'])['valid']
    assert len(os.listdir('/proc/self/fd')) == count


def test_invalidated_tasks_drop_accepted_integrity_metadata(tmp_path):
    path, dag = setup_state(tmp_path)
    dag.metadata['integrity_by_task'] = {'a': {'outcome': 'ACCEPTED'}, 'b': {'outcome': 'ACCEPTED'}, 'c': {'outcome': 'ACCEPTED'}}
    RunStateStore.save(path, goal='g', workspace=str(tmp_path), dag=dag)
    (tmp_path / 'source').write_text('changed')
    restored = RunStateStore.load(path)
    assert restored['dag']['metadata']['integrity_by_task'] == {'c': {'outcome': 'ACCEPTED'}}


def test_forged_sidecar_cannot_replace_controller_fingerprints_or_contract(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTIGRAVITY_STATE_HOME', str(tmp_path.parent / (tmp_path.name + '-trusted')))
    path, dag = setup_state(tmp_path)
    dag.metadata['integrity_contract'] = 'orion-study-v1'
    dag.metadata['evidence_directory'] = '/controller/evidence'
    RunStateStore.save(path, goal='g', workspace=str(tmp_path), dag=dag)
    (tmp_path / 'source').write_text('candidate changed code')
    forged = json.loads(path.read_text())
    forged['dag']['metadata']['task_fingerprints']['a'] = RunStateStore.bind_task(str(tmp_path), dag['a'])
    forged['dag']['metadata'].pop('integrity_contract')
    forged['dag']['metadata']['evidence_directory'] = '/candidate/evidence'
    forged['dag']['tasks'][0]['status'] = 'completed'
    path.write_text(json.dumps(forged))
    restored = RunStateStore.load(path)
    assert restored['untrusted_state_modified'] is True
    assert restored['dag']['metadata']['integrity_contract'] == 'orion-study-v1'
    assert restored['dag']['metadata']['evidence_directory'] == '/controller/evidence'
    assert {t['id']: t['status'] for t in restored['dag']['tasks']} == {'a': 'pending', 'b': 'pending', 'c': 'completed'}


def test_unmapped_version_two_state_cannot_assert_completion(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTIGRAVITY_STATE_HOME', str(tmp_path.parent / (tmp_path.name + '-trusted')))
    path, _ = setup_state(tmp_path)
    # A candidate can create a plausible copy at a different local path, without
    # gaining a controller mirror for that new run identity.
    forged_path = tmp_path / 'candidate-authored.state.json'
    forged_path.write_bytes(path.read_bytes())
    restored = RunStateStore.load(forged_path)
    assert {t['status'] for t in restored['dag']['tasks']} == {'pending'}


def test_controller_mirror_is_external_atomic_and_owner_only(tmp_path, monkeypatch):
    import hashlib
    import os
    home = tmp_path.parent / (tmp_path.name + '-trusted')
    monkeypatch.setenv('ANTIGRAVITY_STATE_HOME', str(home))
    path, _ = setup_state(tmp_path)
    key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    mirror = home / 'run-states' / (key + '.json')
    assert mirror.is_file()
    assert json.loads(mirror.read_text()) == json.loads(path.read_text())
    assert os.stat(mirror).st_mode & 0o777 == 0o600
    assert os.stat(mirror.parent).st_mode & 0o777 == 0o700
    assert not list(mirror.parent.glob('*.tmp'))


def test_controller_state_home_inside_candidate_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTIGRAVITY_STATE_HOME', str(tmp_path / 'controller'))
    with pytest.raises(ValueError, match='outside candidate workspace'):
        setup_state(tmp_path)


@pytest.mark.parametrize('damage', ['invalid_json', 'missing', 'symlink'])
def test_trusted_mirror_survives_unreadable_or_replaced_sidecar(tmp_path, monkeypatch, damage):
    monkeypatch.setenv('ANTIGRAVITY_STATE_HOME', str(tmp_path.parent / (tmp_path.name + '-trusted')))
    path, _ = setup_state(tmp_path)
    if damage == 'invalid_json':
        path.write_text('candidate removed JSON')
    else:
        path.unlink()
        if damage == 'symlink':
            fake = tmp_path / 'fake'
            fake.write_text('{}')
            path.symlink_to(fake)
    restored = RunStateStore.load(path)
    assert restored['untrusted_state_modified'] is True
    assert {t['status'] for t in restored['dag']['tasks']} == {'completed'}
    assert restored['trusted_state_available'] is True


def test_parent_directory_symlink_cannot_change_controller_state_identity(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTIGRAVITY_STATE_HOME', str(tmp_path / 'controller'))
    workspace = tmp_path / 'candidate'
    workspace.mkdir()
    (workspace / 'artifact').write_text('verified')
    dag = ExecutionPlanDAG(tasks=[SubTask(id='science', status='completed', outputs=['artifact'])],
                           metadata={'integrity_contract': 'orion-study-v1'})
    path = workspace / '.antigravity_runs' / 'run.state.json'
    RunStateStore.save(path, goal='g', workspace=str(workspace), dag=dag)
    original_mirror = RunStateStore.trusted_path_for(path)
    forged = json.loads(path.read_text())
    (workspace / 'artifact').write_text('forged')
    forged['dag']['metadata'].pop('integrity_contract')
    forged['dag']['metadata']['task_fingerprints']['science'] = RunStateStore.bind_task(str(workspace), dag['science'])
    path.parent.rename(workspace / '.old-runs')
    replacement = workspace / '.evil'
    replacement.mkdir()
    (replacement / path.name).write_text(json.dumps(forged))
    path.parent.symlink_to(replacement, target_is_directory=True)
    assert RunStateStore.trusted_path_for(path) == original_mirror
    read_state = RunStateStore._read_state
    def controller_only_read(target):
        assert target == original_mirror, 'Do not open a candidate path through symlinked parents'
        return read_state(target)
    monkeypatch.setattr(RunStateStore, '_read_state', controller_only_read)
    restored = RunStateStore.load(path)
    assert restored['untrusted_state_modified'] is True
    assert restored['dag']['metadata']['integrity_contract'] == 'orion-study-v1'
    assert restored['dag']['tasks'][0]['status'] == 'pending'


def test_save_rejects_symlink_parents_before_writing_controller_or_local_state(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTIGRAVITY_STATE_HOME', str(tmp_path / 'controller'))
    workspace = tmp_path / 'candidate'
    workspace.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (workspace / 'runs').symlink_to(outside, target_is_directory=True)
    sidecar = workspace / 'runs' / 'run.state.json'
    mirror = RunStateStore.trusted_path_for(sidecar)
    with pytest.raises(ValueError, match='symlink'):
        RunStateStore.save(sidecar, goal='g', workspace=str(workspace), dag=ExecutionPlanDAG())
    assert not mirror.exists()
    assert not (outside / sidecar.name).exists()
