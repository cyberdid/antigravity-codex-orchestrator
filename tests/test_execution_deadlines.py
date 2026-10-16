import contextlib
import http.server
import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from antigravity_hybrid_orchestrator.architect.client import CloudArchitectClient, CloudArchitectError
from antigravity_hybrid_orchestrator.engine.control import ExecutionBudget, ExecutionControl
from antigravity_hybrid_orchestrator.workforce.ollama_client import WorkforceClient, OllamaConnectionError


@contextlib.contextmanager
def slow_server():
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            self.send_response(200)
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b' ')
                    self.wfile.flush()
                    time.sleep(.02)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_native_http_whole_call_deadline_beats_trickling_response():
    with slow_server() as url:
        client = WorkforceClient(base_url=url, timeout=0)
        start = time.monotonic()
        with pytest.raises(OllamaConnectionError, match='deadline'):
            client.chat([{'role': 'user', 'content': 'task'}], deadline=start + .15)
        assert time.monotonic() - start < .7


def test_global_runtime_cancels_http_while_idle_timeout_disabled():
    with slow_server() as url:
        control = ExecutionControl(ExecutionBudget(max_runtime_s=.15, model_idle_timeout_s=0))
        client = WorkforceClient(base_url=url, timeout=0)
        start = time.monotonic()
        with pytest.raises(OllamaConnectionError, match='runtime_budget_exceeded'):
            client.chat([{'role': 'user', 'content': 'task'}], execution_control=control)
        assert time.monotonic() - start < .7


def test_operator_cancellation_cancels_http_socket():
    with slow_server() as url:
        control = ExecutionControl(ExecutionBudget(model_idle_timeout_s=0))
        timer = threading.Timer(.15, control.cancel)
        timer.start()
        try:
            with pytest.raises(OllamaConnectionError, match='operator_cancelled'):
                WorkforceClient(base_url=url, timeout=0).chat([{'role': 'user', 'content': 'task'}], execution_control=control)
        finally:
            timer.join()


def test_zero_timeout_disables_idle_watchdog():
    assert WorkforceClient(timeout=0).timeout is None


def test_cloud_keyboard_interrupt_reaps_process_group(monkeypatch, tmp_path):
    import antigravity_hybrid_orchestrator.architect.client as module
    original = subprocess.Popen
    processes = []
    def capture(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(module.subprocess, 'Popen', capture)
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(module.selectors.DefaultSelector, 'select', interrupt)
    with pytest.raises(KeyboardInterrupt):
        CloudArchitectClient()._run_subprocess([sys.executable, '-c', 'import time; time.sleep(20)'], 'test')
    assert processes[0].poll() is not None


def test_native_worker_can_cancel_while_waiting_for_inference_lock(tmp_path):
    from antigravity_hybrid_orchestrator.planner.dag import SubTask
    from antigravity_hybrid_orchestrator.workforce.worker import AutonomousWorker
    from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite
    control = ExecutionControl(ExecutionBudget(model_idle_timeout_s=0))
    worker = AutonomousWorker(execution_control=control)
    result = []
    worker._inference_lock.acquire()
    thread = threading.Thread(target=lambda: result.append(worker.execute_subtask(SubTask(id='a'), WorkspaceToolSuite(tmp_path))))
    try:
        thread.start()
        control.cancel()
        thread.join(timeout=.5)
        assert not thread.is_alive(), 'GPU lock wait ignored cancellation'
        assert not result[0].success
        assert result[0].metadata['stop_reason'] == 'operator_cancelled'
    finally:
        worker._inference_lock.release()
        thread.join(timeout=3)


def test_native_worker_blocked_http_returns_attempt_limit(tmp_path):
    from antigravity_hybrid_orchestrator.planner.dag import SubTask
    from antigravity_hybrid_orchestrator.workforce.worker import AutonomousWorker
    from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite
    with slow_server() as url:
        worker = AutonomousWorker(client=WorkforceClient(base_url=url, timeout=0))
        start = time.monotonic()
        result = worker.execute_subtask(SubTask(id='a'), WorkspaceToolSuite(tmp_path), deadline=start + .15)
        assert time.monotonic() - start < .7
        assert not result.success
        assert result.metadata.get('limit_reached') is True
