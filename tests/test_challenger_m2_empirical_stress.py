"""Adversarial and Empirical Stress Test Suite for Milestone 2.

Authored by Challenger 2 (teamwork_preview_challenger).
Empirically challenges:
1. Zero-Knowledge Pre-Flight Privacy Boundaries:
   - Malicious & unstripped WorkspaceOutline objects (unstripped bodies, forbidden AST logic nodes, sensitive literals).
   - Verifies SecurityInvariantViolationError in 100% of cases for both CloudArchitectClient and MockCloudArchitectClient.
   - Verifies 0 cloud subprocess executions when privacy invariants fail.
2. Resilient LLM JSON Extraction (_extract_json):
   - Markdown fences (tagged & untagged).
   - Conversational pre/post-chatter.
   - Illegal trailing commas in objects and arrays.
   - Malformed structures & non-JSON payloads.
   - Array root and dictionary variations.
3. Self-Correction Retry Loop & Error Recovery:
   - Recovers from malformed JSON, cyclic DAGs, missing dependencies, empty tasks.
   - Strictly enforces max_retries limit.
   - Escalates persistent domain errors (DAGCycleError, DAGDependencyError, CloudArchitectError).
   - Verifies privacy violations NEVER trigger retries.
"""

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch
import pytest

from antigravity_hybrid_orchestrator.architect.client import (
    CloudArchitectClient,
    CloudArchitectError,
)
from antigravity_hybrid_orchestrator.architect.mock_client import (
    MockCloudArchitectClient,
)
from antigravity_hybrid_orchestrator.planner.dag import (
    DAGCycleError,
    DAGDependencyError,
    DAGError,
    ExecutionPlanDAG,
    SubTask,
)
from antigravity_hybrid_orchestrator.privacy.verifier import (
    FORBIDDEN_LOGIC_NODES,
    SecurityInvariantViolationError,
    verify_cloud_payload,
)
from antigravity_hybrid_orchestrator.privacy.workspace_privacy import (
    SENSITIVE_FILENAME_PATTERNS,
    verify_workspace_outline,
)
from antigravity_hybrid_orchestrator.workspace.scanner import (
    WorkspaceOutline,
)


# ==============================================================================
# SECTION 1: ZERO-KNOWLEDGE PRE-FLIGHT PRIVACY BOUNDARY STRESS
# ==============================================================================

class TestPreflightPrivacyBoundaries:
    """Stress tests verifying that unstripped and malicious WorkspaceOutline objects
    are 100% blocked before any cloud subprocess or reasoning can execute."""

    @pytest.mark.parametrize(
        "unstripped_body",
        [
            "def authenticate(key: str):\n    secret = 'super_secret_key'\n    return key == secret",
            "def compute_hash(data):\n    import hashlib\n    return hashlib.sha256(data).hexdigest()",
            "async def fetch_token():\n    token = await get_token()\n    return token",
            "def handler(req):\n    x = 10\n    y = 20\n    z = x + y",
            "def empty_return():\n    return",
            "def single_yield():\n    yield 1",
            "def helper():\n    pass\ndef leaked():\n    print('leaked statement')",
        ],
    )
    def test_unstripped_function_bodies_blocked(self, tmp_path: Path, unstripped_body: str):
        """Verify unstripped function bodies trigger SecurityInvariantViolationError."""
        outline = WorkspaceOutline(
            workspace_root=tmp_path,
            file_tree=["src/module.py"],
            ast_signatures={"src/module.py": unstripped_body},
            summary_metadata={"total_files": 1},
        )

        # 1. Mock client must block offline
        mock_client = MockCloudArchitectClient()
        with pytest.raises(SecurityInvariantViolationError):
            mock_client.plan_rd_task(goal="Optimize algorithm", workspace_outline=outline)

        # 2. Cloud client must block before subprocess invocation
        cloud_client = CloudArchitectClient()
        with patch.object(cloud_client, "_invoke_codex") as mock_invoke:
            with pytest.raises(SecurityInvariantViolationError):
                cloud_client.plan_rd_task(goal="Optimize algorithm", workspace_outline=outline)
            assert mock_invoke.call_count == 0, "Subprocess MUST NOT be called when privacy check fails"

    @pytest.mark.parametrize(
        "logic_snippet",
        [
            "if condition:\n    ...",
            "for item in items:\n    ...",
            "while is_running:\n    ...",
            "try:\n    ...\nexcept Exception:\n    ...",
            "with lock:\n    ...",
            "def foo():\n    return 42",
            "def foo():\n    yield 42",
            "def foo():\n    yield from [1, 2]",
            "result = calculate_something()",
            "total = a + b",
            "counter += 1",
            "fn = lambda x: x * 2",
            "items = [x for x in data]",
            "items = {x for x in data}",
            "mapping = {k: v for k, v in pairs}",
            "gen = (x for x in data)",
            "raise ValueError('forbidden')",
            "assert True",
            "result = 10 if active else 20",
            "check = (a == b)",
            "flag = (not is_valid)",
            "combined = (cond_a and cond_b)",
        ],
    )
    def test_forbidden_ast_logic_nodes_blocked(self, tmp_path: Path, logic_snippet: str):
        """Verify presence of any forbidden executable logic AST node triggers SecurityInvariantViolationError."""
        outline = WorkspaceOutline(
            workspace_root=tmp_path,
            file_tree=["src/logic.py"],
            ast_signatures={"src/logic.py": logic_snippet},
            summary_metadata={"total_files": 1},
        )

        mock_client = MockCloudArchitectClient()
        with pytest.raises(SecurityInvariantViolationError):
            mock_client.plan_rd_task(goal="Refactor logic", workspace_outline=outline)

        cloud_client = CloudArchitectClient()
        with patch.object(cloud_client, "_invoke_codex") as mock_invoke:
            with pytest.raises(SecurityInvariantViolationError):
                cloud_client.plan_rd_task(goal="Refactor logic", workspace_outline=outline)
            assert mock_invoke.call_count == 0

    @pytest.mark.parametrize(
        "sensitive_file",
        [
            ".env",
            "config/.env.production",
            "secrets/id_rsa",
            "ssh/id_ed25519",
            "credentials.json",
            "api_secret.txt",
            "certs/private_key.pem",
        ],
    )
    def test_sensitive_file_in_tree_blocked(self, tmp_path: Path, sensitive_file: str):
        """Verify presence of sensitive filenames in file_tree triggers SecurityInvariantViolationError."""
        outline = WorkspaceOutline(
            workspace_root=tmp_path,
            file_tree=["src/app.py", sensitive_file],
            ast_signatures={"src/app.py": "def run() -> None:\n    ..."},
            summary_metadata={"total_files": 2},
        )

        mock_client = MockCloudArchitectClient()
        with pytest.raises(SecurityInvariantViolationError):
            mock_client.plan_rd_task(goal="Deploy app", workspace_outline=outline)

        cloud_client = CloudArchitectClient()
        with patch.object(cloud_client, "_invoke_codex") as mock_invoke:
            with pytest.raises(SecurityInvariantViolationError):
                cloud_client.plan_rd_task(goal="Deploy app", workspace_outline=outline)
            assert mock_invoke.call_count == 0

    def test_sensitive_string_literal_leakage_blocked(self, tmp_path: Path):
        """Verify that when original source contains sensitive literals >= 5 chars,
        leaking them into ast_signatures triggers SecurityInvariantViolationError."""
        # Create a workspace file on disk with sensitive secrets
        src_file = tmp_path / "src" / "vault.py"
        src_file.parent.mkdir(parents=True, exist_ok=True)
        secret_value = "AKIA_PROD_AWS_KEY_SECRET_98765"
        src_file.write_text(
            f'AWS_KEY = "{secret_value}"\ndef get_key(): return "{secret_value}"\n',
            encoding="utf-8",
        )

        # Construct outline that illicitly retained the secret string
        outline = WorkspaceOutline(
            workspace_root=tmp_path,
            file_tree=["src/vault.py"],
            ast_signatures={
                "src/vault.py": f'AWS_KEY: str = "{secret_value}"\ndef get_key() -> str:\n    ...'
            },
            summary_metadata={"total_files": 1},
        )

        mock_client = MockCloudArchitectClient()
        with pytest.raises(SecurityInvariantViolationError):
            mock_client.plan_rd_task(goal="Audit keys", workspace_outline=outline)

        cloud_client = CloudArchitectClient()
        with patch.object(cloud_client, "_invoke_codex") as mock_invoke:
            with pytest.raises(SecurityInvariantViolationError):
                cloud_client.plan_rd_task(goal="Audit keys", workspace_outline=outline)
            assert mock_invoke.call_count == 0

    def test_clean_workspace_outline_passes_preflight(self, tmp_path: Path):
        """Verify that a compliant stripped outline passes preflight checks cleanly."""
        outline = WorkspaceOutline(
            workspace_root=tmp_path,
            file_tree=["src/clean.py", "tests/test_clean.py"],
            ast_signatures={
                "src/clean.py": (
                    "import math\n\n"
                    "class Calculator:\n"
                    "    def add(self, a: int, b: int) -> int:\n"
                    "        ...\n\n"
                    "    def sqrt(self, n: float) -> float:\n"
                    "        ...\n"
                )
            },
            summary_metadata={"total_files": 2, "total_lines": 15},
        )

        mock_client = MockCloudArchitectClient()
        dag_mock = mock_client.plan_rd_task(goal="Implement calculator", workspace_outline=outline)
        assert len(dag_mock) >= 1
        assert dag_mock.validate() is True

        cloud_client = CloudArchitectClient()
        valid_response = json.dumps({
            "tasks": [
                {
                    "id": "task_1",
                    "title": "Implement math functions",
                    "description": "Implement add and sqrt",
                    "dependencies": [],
                    "affected_files": ["src/clean.py"],
                    "success_criteria": "Tests pass",
                    "verification_command": "pytest tests/test_clean.py",
                }
            ]
        })
        with patch.object(cloud_client, "_invoke_codex", return_value=(valid_response, 100, 50)) as mock_invoke:
            dag_cloud = cloud_client.plan_rd_task(goal="Implement calculator", workspace_outline=outline)
            assert mock_invoke.call_count == 1
            assert len(dag_cloud) == 1
            assert "task_1" in dag_cloud


# ==============================================================================
# SECTION 2: ADVERSARIAL LLM RESPONSE EXTRACTION (_extract_json)
# ==============================================================================

class TestAdversarialJSONExtraction:
    """Stress tests verifying the robustness of _extract_json against hostile,
    adversarial, and unconventional LLM output formats."""

    @pytest.fixture
    def client(self) -> CloudArchitectClient:
        return CloudArchitectClient()

    def test_clean_json_object(self, client: CloudArchitectClient):
        payload = '{"tasks": [{"id": "t1", "dependencies": []}]}'
        result = client._extract_json(payload)
        assert result == {"tasks": [{"id": "t1", "dependencies": []}]}

    def test_markdown_fence_with_json_specifier(self, client: CloudArchitectClient):
        payload = "```json\n{\n  \"tasks\": [\n    {\"id\": \"t1\"}\n  ]\n}\n```"
        result = client._extract_json(payload)
        assert result == {"tasks": [{"id": "t1"}]}

    def test_markdown_fence_without_language_tag(self, client: CloudArchitectClient):
        payload = "```\n{\n  \"tasks\": [{\"id\": \"t1\"}]\n}\n```"
        result = client._extract_json(payload)
        assert result == {"tasks": [{"id": "t1"}]}

    def test_markdown_fence_with_trailing_conversational_chatter(self, client: CloudArchitectClient):
        payload = (
            "Here is the execution plan for your requested project:\n\n"
            "```json\n"
            "{\n"
            '  "tasks": [{"id": "t1", "title": "Scaffold"}]\n'
            "}\n"
            "```\n\n"
            "I hope this helps! Let me know if you need any adjustments to the DAG."
        )
        result = client._extract_json(payload)
        assert result == {"tasks": [{"id": "t1", "title": "Scaffold"}]}

    def test_conversational_chatter_without_markdown_fences(self, client: CloudArchitectClient):
        payload = (
            "Certainly! Based on your goal, here is the structured DAG object: "
            '{"tasks": [{"id": "task_alpha", "dependencies": []}]} '
            "Please execute the subtasks sequentially."
        )
        result = client._extract_json(payload)
        assert result == {"tasks": [{"id": "task_alpha", "dependencies": []}]}

    def test_trailing_commas_in_objects_and_arrays(self, client: CloudArchitectClient):
        # Trailing comma in array and trailing comma in object
        payload = """
        {
          "tasks": [
            {
              "id": "t1",
              "dependencies": [],
            },
            {
              "id": "t2",
              "dependencies": ["t1",],
            },
          ],
        }
        """
        result = client._extract_json(payload)
        assert len(result["tasks"]) == 2
        assert result["tasks"][0]["id"] == "t1"
        assert result["tasks"][1]["dependencies"] == ["t1"]

    def test_bare_array_root(self, client: CloudArchitectClient):
        payload = '[{"id": "t1", "dependencies": []}, {"id": "t2", "dependencies": ["t1"]}]'
        result = client._extract_json(payload)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_multiline_strings_and_special_characters(self, client: CloudArchitectClient):
        payload = json.dumps({
            "tasks": [
                {
                    "id": "t1",
                    "description": "Line 1\nLine 2\tTabbed\nSpecial: \u2705 \U0001f680 \"quoted text\"",
                    "dependencies": [],
                }
            ]
        })
        result = client._extract_json(payload)
        assert "Special: \u2705" in result["tasks"][0]["description"]

    @pytest.mark.parametrize(
        "malformed_payload",
        [
            "",
            "   ",
            "\n\t\n",
            "This is just conversational text with no JSON anywhere in the message.",
            "{tasks: [id: t1]}",  # unquoted keys/values
            '{"tasks": [{"id": "t1"',  # truncated JSON
            "}{ bad json }{",
            "```json\nNot even close to JSON\n```",
        ],
    )
    def test_malformed_payloads_raise_cloud_architect_error(
        self, client: CloudArchitectClient, malformed_payload: str
    ):
        with pytest.raises(CloudArchitectError, match="Failed to extract valid JSON"):
            client._extract_json(malformed_payload)

    def test_empty_json_dict_rejected_in_dag_parser(self, client: CloudArchitectClient):
        """When JSON is valid empty object (e.g. '{}' or '}{}{'), _parse_dag_response raises CloudArchitectError."""
        with pytest.raises(CloudArchitectError, match="Response payload contains no subtasks"):
            client._parse_dag_response("}{}{")

        with pytest.raises(CloudArchitectError, match="Response payload contains no subtasks"):
            client._parse_dag_response("{}")

    def test_dict_of_tasks_format(self, client: CloudArchitectClient):
        """Verify _parse_dag_response parses tasks provided as a dict mapping ID -> task object."""
        payload = json.dumps({
            "tasks": {
                "t1": {"title": "Task 1", "dependencies": []},
                "t2": {"title": "Task 2", "dependencies": ["t1"]},
            }
        })
        dag = client._parse_dag_response(payload)
        assert len(dag) == 2
        assert "t1" in dag
        assert "t2" in dag
        assert dag.tasks["t2"].dependencies == ["t1"]

    def test_bare_dict_without_tasks_wrapper(self, client: CloudArchitectClient):
        """Verify _parse_dag_response parses bare dictionary of subtask objects."""
        payload = json.dumps({
            "step_a": {"description": "Step A", "dependencies": []},
            "step_b": {"description": "Step B", "dependencies": ["step_a"]},
        })
        dag = client._parse_dag_response(payload)
        assert len(dag) == 2
        assert [t.id for t in dag.get_execution_order()] == ["step_a", "step_b"]

    def test_invalid_tasks_field_type_raises(self, client: CloudArchitectClient):
        """Verify that string or int value in 'tasks' field raises CloudArchitectError."""
        payload = json.dumps({"tasks": "not a list or dict"})
        with pytest.raises(CloudArchitectError, match="'tasks' field is neither list nor dict"):
            client._parse_dag_response(payload)


# ==============================================================================
# SECTION 3: ERROR RECOVERY & SELF-CORRECTION RETRY LOOP
# ==============================================================================

class TestSelfCorrectionRetryLoop:
    """Stress tests verifying the self-correction retry feedback loop in CloudArchitectClient."""

    @patch.object(CloudArchitectClient, "_invoke_codex")
    def test_retry_on_malformed_json_then_succeeds(self, mock_invoke):
        """Simulate LLM returning malformed text on attempt 1, then valid JSON on attempt 2."""
        bad_response = "I cannot give you JSON right now, sorry!"
        good_response = json.dumps({
            "tasks": [
                {
                    "id": "task_1",
                    "title": "Initial task",
                    "description": "Do something",
                    "dependencies": [],
                    "verification_command": "true",
                }
            ]
        })

        mock_invoke.side_effect = [
            (bad_response, 100, 20),
            (good_response, 150, 45),
        ]

        client = CloudArchitectClient()
        dag = client.plan_rd_task("Build feature", max_retries=2)

        assert mock_invoke.call_count == 2
        # Check that error diagnostic was fed into second prompt
        second_prompt = mock_invoke.call_args_list[1][0][0]
        assert "CRITICAL ERROR IN PREVIOUS RESPONSE" in second_prompt
        assert "Failed to extract valid JSON" in second_prompt
        assert len(dag) == 1
        assert dag.tasks["task_1"].title == "Initial task"

    @patch.object(CloudArchitectClient, "_invoke_codex")
    def test_retry_on_cycle_then_succeeds(self, mock_invoke):
        """Simulate LLM returning a cyclic DAG on attempt 1, correcting to acyclic on attempt 2."""
        cyclic_response = json.dumps({
            "tasks": [
                {"id": "A", "dependencies": ["B"]},
                {"id": "B", "dependencies": ["A"]},
            ]
        })
        acyclic_response = json.dumps({
            "tasks": [
                {"id": "A", "dependencies": []},
                {"id": "B", "dependencies": ["A"]},
            ]
        })

        mock_invoke.side_effect = [
            (cyclic_response, 100, 30),
            (acyclic_response, 130, 40),
        ]

        client = CloudArchitectClient()
        dag = client.plan_rd_task("Resolve dependencies", max_retries=2)

        assert mock_invoke.call_count == 2
        second_prompt = mock_invoke.call_args_list[1][0][0]
        assert "CRITICAL ERROR IN PREVIOUS RESPONSE" in second_prompt
        assert "Cycle detected" in second_prompt
        order = [t.id for t in dag.get_execution_order()]
        assert order == ["A", "B"]

    @patch.object(CloudArchitectClient, "_invoke_codex")
    def test_retry_on_missing_dependency_then_succeeds(self, mock_invoke):
        """Simulate LLM returning a missing dependency on attempt 1, correcting on attempt 2."""
        missing_dep_response = json.dumps({
            "tasks": [
                {"id": "task_2", "dependencies": ["task_phantom"]},
            ]
        })
        valid_response = json.dumps({
            "tasks": [
                {"id": "task_1", "dependencies": []},
                {"id": "task_2", "dependencies": ["task_1"]},
            ]
        })

        mock_invoke.side_effect = [
            (missing_dep_response, 100, 30),
            (valid_response, 140, 50),
        ]

        client = CloudArchitectClient()
        dag = client.plan_rd_task("Fix pipeline", max_retries=2)

        assert mock_invoke.call_count == 2
        second_prompt = mock_invoke.call_args_list[1][0][0]
        assert "CRITICAL ERROR IN PREVIOUS RESPONSE" in second_prompt
        assert "Missing dependency" in second_prompt or "task_phantom" in second_prompt
        assert len(dag) == 2

    @patch.object(CloudArchitectClient, "_invoke_codex")
    def test_retry_on_empty_tasks_then_succeeds(self, mock_invoke):
        """Simulate LLM returning empty tasks array on attempt 1, valid tasks on attempt 2."""
        empty_tasks_response = json.dumps({"tasks": []})
        valid_response = json.dumps({
            "tasks": [{"id": "t1", "dependencies": [], "verification_command": "true"}]
        })

        mock_invoke.side_effect = [
            (empty_tasks_response, 80, 10),
            (valid_response, 120, 30),
        ]

        client = CloudArchitectClient()
        dag = client.plan_rd_task("Plan task", max_retries=2)

        assert mock_invoke.call_count == 2
        second_prompt = mock_invoke.call_args_list[1][0][0]
        assert "CRITICAL ERROR IN PREVIOUS RESPONSE" in second_prompt
        assert len(dag) == 1

    @patch.object(CloudArchitectClient, "_invoke_codex")
    def test_persistent_malformed_json_exhausts_retries(self, mock_invoke):
        """When LLM repeatedly returns malformed JSON, retries are exhausted and CloudArchitectError raised."""
        mock_invoke.return_value = ("Not JSON at all", 50, 10)

        client = CloudArchitectClient()
        with pytest.raises(CloudArchitectError, match="Failed to generate valid ExecutionPlanDAG after 3 attempts"):
            client.plan_rd_task("Goal", max_retries=2)

        assert mock_invoke.call_count == 3  # attempt 0, attempt 1, attempt 2

    @patch.object(CloudArchitectClient, "_invoke_codex")
    def test_persistent_cycle_exhausts_retries_and_raises_dag_cycle_error(self, mock_invoke):
        """When LLM repeatedly returns a cyclic graph, DAGCycleError is raised after max_retries."""
        cyclic_response = json.dumps({
            "tasks": [
                {"id": "A", "dependencies": ["B"]},
                {"id": "B", "dependencies": ["A"]},
            ]
        })
        mock_invoke.return_value = (cyclic_response, 100, 30)

        client = CloudArchitectClient()
        with pytest.raises(DAGCycleError):
            client.plan_rd_task("Goal with cycle", max_retries=2)

        assert mock_invoke.call_count == 3

    @patch.object(CloudArchitectClient, "_invoke_codex")
    def test_persistent_missing_dependency_exhausts_retries_and_raises_dag_dependency_error(
        self, mock_invoke
    ):
        """When LLM repeatedly returns a missing dependency, DAGDependencyError is raised after max_retries."""
        missing_dep_response = json.dumps({
            "tasks": [
                {"id": "A", "dependencies": ["nonexistent_parent"]},
            ]
        })
        mock_invoke.return_value = (missing_dep_response, 100, 30)

        client = CloudArchitectClient()
        with pytest.raises(DAGDependencyError):
            client.plan_rd_task("Goal with missing dep", max_retries=2)

        assert mock_invoke.call_count == 3

    def test_privacy_violation_never_retries(self, tmp_path: Path):
        """SecurityInvariantViolationError must abort immediately without triggering retries."""
        bad_outline = WorkspaceOutline(
            workspace_root=tmp_path,
            file_tree=["src/bad.py"],
            ast_signatures={"src/bad.py": "def foo(): return 'secret'"},
        )

        client = CloudArchitectClient()
        with patch.object(client, "_invoke_codex") as mock_invoke:
            with pytest.raises(SecurityInvariantViolationError):
                client.plan_rd_task("Goal", workspace_outline=bad_outline, max_retries=5)
            assert mock_invoke.call_count == 0


# ==============================================================================
# SECTION 4: BOUNDARY AND CORNER CASES
# ==============================================================================

class TestBoundaryAndCornerCases:
    """Edge cases for arguments, zero retries, and unusual topologies."""

    @pytest.mark.parametrize("empty_goal", ["", "   ", "\n\t\n"])
    def test_empty_goal_rejected(self, empty_goal: str):
        client = CloudArchitectClient()
        with pytest.raises(ValueError, match="Goal string cannot be empty"):
            client.plan_rd_task(empty_goal)

        mock_client = MockCloudArchitectClient()
        with pytest.raises(ValueError, match="Goal string cannot be empty"):
            mock_client.plan_rd_task(empty_goal)

    @patch.object(CloudArchitectClient, "_invoke_codex")
    def test_zero_retries_single_attempt(self, mock_invoke):
        mock_invoke.return_value = ("Not JSON", 50, 10)
        client = CloudArchitectClient()
        with pytest.raises(CloudArchitectError, match="after 1 attempts"):
            client.plan_rd_task("Goal", max_retries=0)
        assert mock_invoke.call_count == 1
