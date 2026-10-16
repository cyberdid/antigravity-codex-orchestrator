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
    path, _ = setup_state(tmp_path)
    state = json.loads(path.read_text())
    state['dag']['tasks'][0]['verification_command'] = 'true'
    path.write_text(json.dumps(state))
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
