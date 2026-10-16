"""Sandbox regressions; set RUN_DOCKER_TESTS=1 for real Docker acceptance tests."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time

import pytest

from antigravity_hybrid_orchestrator.engine.control import ExecutionControl
from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite


def test_host_backend_is_explicitly_recorded(tmp_path):
    result = WorkspaceToolSuite(tmp_path).run_command("printf compatible")
    assert result.success
    assert result.output == "compatible"
    assert result.metadata["sandbox_backend"] == "host"


def test_unknown_backend_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="sandbox_backend"):
        WorkspaceToolSuite(tmp_path, sandbox_backend="typo")


def test_docker_missing_fails_closed_without_host_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    result = WorkspaceToolSuite(tmp_path, sandbox_backend="docker").run_command(
        "echo escaped > host-execution", timeout=2
    )
    assert not result.success
    assert not (tmp_path / "host-execution").exists()
    assert result.metadata["sandbox_backend"] == "docker"
    assert "docker" in result.error.lower()


def test_docker_cannot_select_host_cwd(tmp_path):
    from antigravity_hybrid_orchestrator.runtime.sandbox import DockerSandbox
    sandbox = DockerSandbox(tmp_path)
    for cwd in ("..", "/etc"):
        with pytest.raises((ValueError, PermissionError)):
            sandbox.build_command("pwd", cwd_relative=cwd)


def test_docker_rejects_root_uid(tmp_path, monkeypatch):
    from antigravity_hybrid_orchestrator.runtime.sandbox import DockerSandbox
    monkeypatch.setattr(os, "getuid", lambda: 0)
    with pytest.raises(ValueError, match="root"):
        DockerSandbox(tmp_path)


def test_docker_rejects_output_overlapping_workspace(tmp_path):
    from antigravity_hybrid_orchestrator.runtime.sandbox import DockerSandbox
    (tmp_path / "outputs").mkdir()
    with pytest.raises(ValueError, match="overlap"):
        DockerSandbox(tmp_path, output_directory=tmp_path / "outputs")


def test_wall_timeout_applies_even_with_continuous_output(tmp_path):
    result = WorkspaceToolSuite(tmp_path).run_command(
        "while true; do printf x; done", timeout=0.2, idle_timeout=0
    )
    assert not result.success
    assert result.metadata["timed_out"]
    assert result.metadata["duration_s"] < 4


real_docker = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1", reason="set RUN_DOCKER_TESTS=1"
)


@real_docker
def test_real_docker_isolation_workspace_resources_and_runtime(tmp_path, monkeypatch):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    canary = tmp_path / "controller-secret"
    canary.write_text("hidden-evaluation-answer")
    (workspace / "escape").symlink_to(canary)
    monkeypatch.setenv("RESEARCH_SECRET_TOKEN", "credential-canary")
    script = f'''
import json, os, pathlib, socket
import numpy, scipy, sklearn, psutil, pytest, rich, httpx
p = pathlib.Path
p('/workspace/artifact').write_text('allowed')
s = socket.socket(); s.settimeout(0.2)
try:
    s.connect(('1.1.1.1', 53)); network = True
except OSError:
    network = False
try:
    p('/root-write').write_text('denied'); root_write = True
except OSError:
    root_write = False
print(json.dumps(dict(uid=os.getuid(), canary=p({str(canary)!r}).exists(),
    escape=p('/workspace/escape').exists(), credential=os.getenv('RESEARCH_SECRET_TOKEN'),
    socket=p('/var/run/docker.sock').exists(), host_home=p({str(Path.home())!r}).exists(),
    network=network, root_write=root_write, status=p('/proc/self/status').read_text(),
    memory=p('/sys/fs/cgroup/memory.max').read_text().strip(),
    pids=p('/sys/fs/cgroup/pids.max').read_text().strip(),
    cpu=p('/sys/fs/cgroup/cpu.max').read_text().strip())))
'''
    result = WorkspaceToolSuite(workspace, sandbox_backend="docker").run_command(
        "python -c " + shlex.quote(script), timeout=30
    )
    assert result.success, result.error
    data = json.loads(result.output)
    assert data["uid"] == os.getuid() != 0
    assert data["canary"] is False
    assert data["escape"] is False
    assert data["credential"] is None
    assert data["socket"] is False
    assert data["host_home"] is False
    assert data["network"] is False
    assert data["root_write"] is False
    assert "CapEff:\t0000000000000000" in data["status"]
    assert "NoNewPrivs:\t1" in data["status"]
    assert int(data["memory"]) <= 4 * 1024**3
    assert int(data["pids"]) <= 256
    quota, period = map(int, data["cpu"].split())
    assert 0 < quota / period <= 2
    assert (workspace / "artifact").read_text() == "allowed"
    assert result.metadata["sandbox_backend"] == "docker"
    assert result.metadata["sandbox_image"] == "antigravity-orion:local"
    _assert_removed(result.metadata["container_name"])


def _assert_removed(name):
    inspected = subprocess.run(["docker", "container", "inspect", name], capture_output=True)
    assert inspected.returncode != 0, f"container survived: {name}"


@real_docker
@pytest.mark.parametrize("stop_kind", ["timeout", "cancel", "idle"])
def test_real_docker_kills_descendants_and_removes_container(tmp_path, stop_kind):
    control = ExecutionControl()
    suite = WorkspaceToolSuite(tmp_path, sandbox_backend="docker", execution_control=control)
    def cancel_after_start():
        deadline = time.monotonic() + 10
        while not (tmp_path / "started").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        control.cancel("test_cancel")
    thread = threading.Thread(target=cancel_after_start) if stop_kind == "cancel" else None
    if thread:
        thread.start()
    try:
        result = suite.run_command(
            "(trap '' TERM; sleep 30; echo escaped > survivor) & touch started; wait",
            timeout=1 if stop_kind == "timeout" else 15,
            idle_timeout=1 if stop_kind == "idle" else 0,
        )
    finally:
        if thread:
            thread.join(timeout=12)
    assert not result.success
    assert result.metadata["timed_out"]
    assert (tmp_path / "started").exists()
    assert not (tmp_path / "survivor").exists()
    _assert_removed(result.metadata["container_name"])


@real_docker
def test_read_only_candidate_with_explicit_ephemeral_output(tmp_path):
    from antigravity_hybrid_orchestrator.runtime.sandbox import DockerSandbox
    candidate = tmp_path / "candidate"
    output = tmp_path / "output"
    candidate.mkdir(); output.mkdir()
    (candidate / "input").write_text("fixture")
    sandbox = DockerSandbox(candidate, output_directory=output)
    command = sandbox.build_command(
        "cat input > /output/result; echo forbidden > /workspace/mutation", read_only=True
    )
    try:
        result = subprocess.run(command, env=sandbox.client_environment(), capture_output=True, timeout=15)
    finally:
        sandbox.cleanup()
    assert result.returncode != 0
    assert (output / "result").read_text() == "fixture"
    assert not (candidate / "mutation").exists()
    _assert_removed(sandbox.container_name)


@real_docker
def test_missing_image_fails_closed(tmp_path):
    result = WorkspaceToolSuite(
        tmp_path, sandbox_backend="docker", sandbox_image="antigravity-missing:absent"
    ).run_command("touch escaped", timeout=10)
    assert not result.success
    assert not (tmp_path / "escaped").exists()
    _assert_removed(result.metadata["container_name"])


@real_docker
@pytest.mark.parametrize("tool", ["git_status", "git_diff"])
def test_git_tools_cannot_run_candidate_config_commands_on_host(tmp_path, tool):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    marker = tmp_path / "host-git-command"
    subprocess.run(["git", "init", str(workspace)], check=True, capture_output=True)
    (workspace / "tracked").write_text("initial")
    subprocess.run(["git", "-C", str(workspace), "add", "tracked"], check=True)
    subprocess.run(["git", "-C", str(workspace), "-c", "user.name=test", "-c",
                    "user.email=test@example.invalid", "commit", "-m", "fixture"],
                   check=True, capture_output=True)
    (workspace / "tracked").write_text("changed")
    script = "#!/bin/sh\necho escaped > " + shlex.quote(str(marker)) + "\nexit 1\n"
    (workspace / "trigger").write_text(script)
    (workspace / "trigger").chmod(0o755)
    setting = "core.fsmonitor" if tool == "git_status" else "diff.external"
    subprocess.run(["git", "-C", str(workspace), "config", setting, "./trigger"], check=True)
    suite = WorkspaceToolSuite(workspace, sandbox_backend="docker")
    getattr(suite, tool)()
    assert not marker.exists(), "Git executed workspace-owned configuration on the host"


@real_docker
def test_real_docker_limits_candidate_output_file_size(tmp_path):
    result = WorkspaceToolSuite(tmp_path, sandbox_backend="docker").run_command(
        "python -c \"open('oversize', 'wb').write(b'x' * (20 * 1024**2))\"", timeout=10
    )
    assert not result.success
    assert (tmp_path / "oversize").stat().st_size <= 16 * 1024**2


def test_command_output_is_bounded_without_pipe_deadlock(tmp_path):
    result = WorkspaceToolSuite(tmp_path).run_command(
        "python -c \"import sys; sys.stdout.write('x' * (20 * 1024**2)); "
        "sys.stderr.write('y' * (20 * 1024**2))\"", timeout=10
    )
    assert result.success, result.error
    assert len(result.output) <= 16 * 1024**2
    assert len(result.metadata["stderr"]) <= 16 * 1024**2
    assert result.metadata["output_truncated"] is True


@real_docker
def test_runtime_supports_offline_package_install_in_ephemeral_directory(tmp_path):
    from antigravity_hybrid_orchestrator.runtime.sandbox import DockerSandbox
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires=["setuptools>=68", "wheel"]\n'
        'build-backend="setuptools.build_meta"\n[project]\n'
        'name="sandbox-package-probe"\nversion="0.0.1"\n'
    )
    (tmp_path / "sandbox_probe.py").write_text("ANSWER = 42\n")
    sandbox = DockerSandbox(tmp_path)
    try:
        result = subprocess.run(sandbox.build_command(
            "python -m pip install --no-index --no-deps --no-build-isolation --target /tmp/installed . "
            "&& cd /tmp && PYTHONPATH=/tmp/installed python -c "
            "'from sandbox_probe import ANSWER; assert ANSWER == 42'"
        ), env=sandbox.client_environment(), capture_output=True, timeout=30)
    finally:
        sandbox.cleanup()
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_file_size_limit_must_be_positive_integer(tmp_path, limit):
    from antigravity_hybrid_orchestrator.runtime.sandbox import DockerSandbox
    with pytest.raises(ValueError, match='max_file_size_bytes'):
        DockerSandbox(tmp_path, max_file_size_bytes=limit)
    with pytest.raises(ValueError, match='max_file_size_bytes'):
        WorkspaceToolSuite(tmp_path, max_file_size_bytes=limit)


@real_docker
def test_explicit_operator_file_size_limit_permits_larger_scientific_artifact(tmp_path):
    result = WorkspaceToolSuite(
        tmp_path, sandbox_backend='docker', max_file_size_bytes=64 * 1024**2
    ).run_command(
        "python -c \"open('large-artifact', 'wb').write(b'x' * (32 * 1024**2))\"", timeout=10
    )
    assert result.success, result.error
    assert (tmp_path / 'large-artifact').stat().st_size == 32 * 1024**2


@real_docker
def test_candidate_cannot_forge_resume_completion_and_remove_scientific_policy(tmp_path, monkeypatch):
    from antigravity_hybrid_orchestrator.engine.run_state import RunStateStore
    from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG, SubTask
    workspace = tmp_path / 'candidate'
    workspace.mkdir()
    monkeypatch.setenv('ANTIGRAVITY_STATE_HOME', str(tmp_path / 'controller-state'))
    (workspace / 'artifact').write_text('verified')
    dag = ExecutionPlanDAG(
        tasks=[SubTask(id='science', status='completed', outputs=['artifact'])],
        metadata={'integrity_contract': 'orion-core-v1',
                  'integrity_by_task': {'science': {'status': 'ACCEPTED'}}},
    )
    sidecar = workspace / '.antigravity_runs/original.state.json'
    RunStateStore.save(sidecar, goal='g', workspace=str(workspace), dag=dag)
    attack = '''import hashlib, json, pathlib
p = pathlib.Path('.antigravity_runs/original.state.json')
s = json.loads(p.read_text())
pathlib.Path('artifact').write_text('forged')
s['dag']['metadata']['task_fingerprints']['science']['files']['artifact'] = hashlib.sha256(b'forged').hexdigest()
s['dag']['metadata'].pop('integrity_contract')
s['dag']['metadata'].pop('integrity_by_task')
p.write_text(json.dumps(s))
'''
    result = WorkspaceToolSuite(workspace, sandbox_backend='docker').run_command(
        'python -c ' + shlex.quote(attack), timeout=10
    )
    assert result.success, result.error
    restored = RunStateStore.load(sidecar)
    assert restored['dag']['tasks'][0]['status'] == 'pending'
    assert restored['dag']['metadata']['integrity_contract'] == 'orion-core-v1'
