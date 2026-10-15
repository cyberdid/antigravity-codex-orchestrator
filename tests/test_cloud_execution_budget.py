import json
from unittest.mock import patch

from antigravity_hybrid_orchestrator.architect.client import CloudArchitectClient
from antigravity_hybrid_orchestrator.engine.control import ExecutionBudget, ExecutionControl


def test_cloud_planner_accounts_usage_in_shared_run_control():
    control = ExecutionControl(ExecutionBudget(max_remote_tokens=100))
    client = CloudArchitectClient(execution_control=control)
    response = json.dumps({"tasks": [{"id": "task", "title": "Task"}]})

    with patch.object(client, "_invoke_codex", return_value=(response, 60, 50)):
        dag = client.plan_rd_task("Plan task")

    assert control.remote_tokens == 110
    assert control.check().reason == "remote_token_budget_exceeded"
    assert dag.metadata["input_tokens"] == 60
    assert dag.metadata["output_tokens"] == 50


def test_cloud_planner_adds_usage_from_invalid_response_retry():
    control = ExecutionControl(ExecutionBudget(max_remote_tokens=1000))
    client = CloudArchitectClient(execution_control=control)
    valid = json.dumps({"tasks": [{"id": "task", "title": "Task"}]})
    with patch.object(
        client,
        "_invoke_codex",
        side_effect=[("invalid", 15, 5), (valid, 25, 10)],
    ):
        dag = client.plan_rd_task("Plan task")

    assert control.remote_tokens == 55
    assert dag.metadata["input_tokens"] == 40
    assert dag.metadata["output_tokens"] == 15
