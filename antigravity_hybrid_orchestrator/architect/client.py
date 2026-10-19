"""Cloud Architect client: luna (codex), haiku (Claude CLI), or agy for cloud reasoning.

This is the Codex fork of the Antigravity Hybrid Orchestrator: the Cloud
Architect strategic planner is driven by the OpenAI Codex CLI (`codex exec`)
instead of the Antigravity CLI (`agy`). Everything downstream — the zero-knowledge
privacy invariant, DAG parsing, and the local Ollama workforce — is identical.
"""

import json
import os
import re
import selectors
import signal
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from antigravity_hybrid_orchestrator.architect import StrategyResult
from antigravity_hybrid_orchestrator.planner.dag import (
    DAGCycleError,
    DAGDependencyError,
    DAGError,
    ExecutionPlanDAG,
    SubTask,
)
from antigravity_hybrid_orchestrator.planner.validator import PlanValidator
from antigravity_hybrid_orchestrator.engine.control import ExecutionControl
from antigravity_hybrid_orchestrator.privacy.verifier import (
    SecurityInvariantViolationError,
    verify_cloud_payload,
)
from antigravity_hybrid_orchestrator.privacy.workspace_privacy import verify_workspace_outline
from antigravity_hybrid_orchestrator.workspace.scanner import WorkspaceOutline


class CloudArchitectError(Exception):
    """Raised when the Cloud Architect invocation or response parsing fails."""
    pass


class CloudArchitectClient:
    """Client for invoking the Cloud Architect via the Codex CLI (`codex exec`)."""

    DEFAULT_CODEX_PATH = os.path.expanduser("~/.local/bin/codex")
    DEFAULT_AGY_PATH = os.path.expanduser("~/.local/bin/agy")
    BACKENDS = ("luna", "haiku", "agy")
    CLI_NAMES = {"codex": "Codex", "claude": "Claude", "agy": "Antigravity"}
    DEFAULT_MODELS = {"luna": "gpt-6-luna", "haiku": "claude-haiku-5-5", "agy": None}

    def __init__(
        self,
        codex_path: Optional[str] = None,
        timeout: Optional[float] = None,
        model: Optional[str] = None,
        execution_control: Optional[ExecutionControl] = None,
        backend: str = "luna",
        cli_path: Optional[str] = None,
    ) -> None:
        """Creates a Cloud Architect client for one backend.

        Args:
            codex_path: Override for the codex binary (luna backend).
            timeout: Wall-clock limit per call in seconds (None = unlimited).
            model: Model override; defaults to the backend's model.
            execution_control: Shared run budget and stop checks.
            backend: "luna" (codex, gpt-6-luna), "haiku" (Claude Haiku 5.5 via the
                claude CLI), or "agy" (Antigravity CLI).
            cli_path: Override for the claude or agy binary.
        """
        backend = (backend or "luna").strip().lower()
        if backend not in self.BACKENDS:
            raise ValueError(
                f"Unknown Cloud Architect backend '{backend}'. "
                f"Choose one of: {', '.join(self.BACKENDS)}"
            )
        self.backend = backend
        if codex_path:
            self.codex_path = codex_path
        else:
            self.codex_path = shutil.which("codex") or self.DEFAULT_CODEX_PATH
        if cli_path:
            self.cli_path = cli_path
        elif backend == "haiku":
            self.cli_path = shutil.which("claude") or "claude"
        elif backend == "agy":
            self.cli_path = shutil.which("agy") or self.DEFAULT_AGY_PATH
        else:
            self.cli_path = self.codex_path
        self.timeout = timeout
        self.model = (
            model
            or os.environ.get("ARCHITECT_MODEL", "").strip()
            or self.DEFAULT_MODELS[backend]
        )
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        # Cost reported by the CLI for the most recent call (None when the CLI reports none).
        self.last_call_cost_usd: Optional[float] = None
        # Per-call usage not yet copied into telemetry by the coordinator.
        self.usage_ledger: List[Dict[str, Any]] = []
        self.execution_control = execution_control

    @staticmethod
    def _extract_codex_usage(stdout: str) -> Tuple[int, int]:
        """Best-effort parse of token usage from `codex exec --json` JSONL events.

        Codex emits one JSON object per line; token accounting events carry
        input/output counts under a few possible key spellings. We scan every
        line and keep the last usage-bearing event. Returns (input, output),
        both 0 when nothing usable is found.
        """
        input_tokens = 0
        output_tokens = 0
        for line in stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Unwrap common envelopes: {"msg": {...}} / {"usage": {...}} / flat.
            candidates = [obj]
            for key in ("msg", "usage", "info", "token_usage", "token_count"):
                inner = obj.get(key) if isinstance(obj, dict) else None
                if isinstance(inner, dict):
                    candidates.append(inner)
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                in_v = c.get("input_tokens", c.get("prompt_tokens"))
                out_v = c.get("output_tokens", c.get("completion_tokens"))
                if in_v is not None:
                    input_tokens = int(in_v)
                if out_v is not None:
                    output_tokens = int(out_v)
        return input_tokens, output_tokens

    def _invoke_cloud(self, prompt: str) -> Tuple[str, int, int]:
        """Sends one planning call and records its usage in ``usage_ledger``.

        Every call is recorded, including calls whose response is later rejected,
        so the caller can account for all cloud spend, not only the accepted plan.
        """
        self.last_call_cost_usd = None
        try:
            raw, in_tokens, out_tokens = self._dispatch_cloud(prompt)
        except Exception:
            # The tokens were spent even though the call failed; the cost of a
            # failed call is unknown here, so it is recorded as unreported.
            self.usage_ledger.append({"input_tokens": 0, "output_tokens": 0,
                                      "cost_usd": None, "failed": True})
            raise
        self.usage_ledger.append({"input_tokens": in_tokens, "output_tokens": out_tokens,
                                  "cost_usd": self.last_call_cost_usd, "failed": False})
        return raw, in_tokens, out_tokens

    def _dispatch_cloud(self, prompt: str) -> Tuple[str, int, int]:
        """Routes one planning call to the selected backend."""
        if self.backend == "agy":
            return self._invoke_agy(prompt)
        if self.backend == "haiku":
            return self._invoke_claude(prompt)
        return self._invoke_codex(prompt)

    def _run_subprocess(self, cmd: List[str], label: str) -> Tuple[int, str, str]:
        """Runs a backend CLI under the execution budget, idle and wall-clock guards.

        Returns (return_code, stdout_text, stderr_text). Raises CloudArchitectError
        if the process cannot start or is stopped by a limit.
        """
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=(os.name == "posix"),
            )
        except FileNotFoundError as e:
            raise CloudArchitectError(
                f"{self.CLI_NAMES.get(label, label)} CLI binary not found at '{cmd[0]}': {e}"
            ) from e
        except Exception as e:
            raise CloudArchitectError(f"Subprocess execution error: {e}") from e

        stdout_chunks: List[bytes] = []
        stderr_chunks: List[bytes] = []
        started = last_activity = time.monotonic()
        idle_timeout = (
            self.execution_control.budget.model_idle_timeout_s
            if self.execution_control is not None else 0
        )
        stop_reason = ""
        def terminate_tree():
            # SIGKILL the session even if its leader has already exited: descendants
            # may ignore SIGTERM or retain the output pipes.
            try:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    elif process.poll() is None:
                        process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=.2)
                except subprocess.TimeoutExpired:
                    pass
            finally:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    elif process.poll() is None:
                        process.kill()
                except ProcessLookupError:
                    pass
                process.wait()

        try:
            with selectors.DefaultSelector() as selector:
                assert process.stdout is not None and process.stderr is not None
                selector.register(process.stdout, selectors.EVENT_READ, stdout_chunks)
                selector.register(process.stderr, selectors.EVENT_READ, stderr_chunks)
                while selector.get_map() or process.poll() is None:
                    now = time.monotonic()
                    if self.timeout is not None and now - started >= self.timeout:
                        stop_reason = f"timed out after {self.timeout}s"
                        break
                    if self.execution_control is not None:
                        decision = self.execution_control.check()
                        if decision.stopped:
                            stop_reason = f"stopped: {decision.reason}"
                            break
                    if idle_timeout > 0 and now - last_activity >= idle_timeout:
                        stop_reason = f"inactive for {idle_timeout}s"
                        break
                    for key, _ in selector.select(timeout=0.2):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if chunk:
                            key.data.append(chunk)
                            last_activity = time.monotonic()
                        else:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
            if stop_reason:
                raise CloudArchitectError(f"Cloud Architect ({label}) {stop_reason}")
        except BaseException:
            terminate_tree()
            raise
        finally:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

        return_code = process.wait()
        stdout_text = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        return return_code, stdout_text, stderr_text

    def _invoke_codex(self, prompt: str) -> Tuple[str, int, int]:
        """Runs `codex exec` (read-only, non-interactive) for the luna backend.

        The agent's final message is captured via ``--output-last-message`` (the
        model's answer, which for our prompts is the JSON strategy/DAG), while token
        usage is parsed best-effort from the ``--json`` event stream.
        """
        with tempfile.NamedTemporaryFile(
            mode="r", suffix=".txt", prefix="codex_last_", delete=False
        ) as tf:
            last_message_path = tf.name

        try:
            cmd = [
                self.codex_path,
                "exec",
                "--json",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--output-last-message",
                last_message_path,
            ]
            if self.model:
                cmd += ["-m", self.model]
            cmd += ["--", prompt]

            return_code, stdout_text, stderr_text = self._run_subprocess(cmd, "codex")
            if return_code != 0:
                raise CloudArchitectError(
                    f"Cloud Architect (codex) returned non-zero exit code "
                    f"({return_code}): {stderr_text or stdout_text}"
                )

            try:
                with open(last_message_path, "r", encoding="utf-8") as fh:
                    raw_response = fh.read().strip()
            except OSError as e:
                raise CloudArchitectError(
                    f"Failed to read codex final message file: {e}"
                ) from e
        finally:
            try:
                os.unlink(last_message_path)
            except OSError:
                pass

        if not raw_response:
            raise CloudArchitectError(
                f"Codex produced an empty final message. stderr: {stderr_text[:200]}"
            )

        input_tokens, output_tokens = self._extract_codex_usage(stdout_text)
        # Fall back to a rough estimate when the event stream carries no usage.
        if input_tokens == 0:
            input_tokens = max(1, len(prompt) // 4)
        if output_tokens == 0:
            output_tokens = max(1, len(raw_response) // 4)
        return raw_response, input_tokens, output_tokens

    def _invoke_claude(self, prompt: str) -> Tuple[str, int, int]:
        """Calls Claude Haiku via the Claude Code CLI in print mode, with tools disabled.

        The planner must only reason, never act, so all built-in tools are turned off
        and the plan permission mode is used. Input tokens include prompt-cache reads
        and writes so the telemetry reflects the full context sent.
        """
        cmd = [
            self.cli_path,
            "-p",
            "--model", self.model or self.DEFAULT_MODELS["haiku"],
            "--output-format", "json",
            "--tools", "",
            "--permission-mode", "plan",
            "--", prompt,
        ]
        return_code, stdout_text, stderr_text = self._run_subprocess(cmd, "claude")
        if return_code != 0:
            raise CloudArchitectError(
                f"Cloud Architect (claude) returned non-zero exit code "
                f"({return_code}): {stderr_text or stdout_text}"
            )

        try:
            payload = json.loads(stdout_text)
        except json.JSONDecodeError as e:
            raise CloudArchitectError(
                f"Failed to parse JSON output from claude: {e}. Output was: {stdout_text[:200]}"
            ) from e

        if payload.get("is_error") or payload.get("subtype", "success") != "success":
            raise CloudArchitectError(
                f"Claude returned an error: {payload.get('result') or payload.get('subtype')}"
            )

        raw_response = str(payload.get("result", "")).strip()
        if not raw_response:
            raise CloudArchitectError("Claude produced an empty result")

        usage = payload.get("usage", {}) or {}
        input_tokens = (
            int(usage.get("input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0))
        )
        output_tokens = int(usage.get("output_tokens", 0))
        cost = payload.get("total_cost_usd")
        if cost is not None:
            self.last_call_cost_usd = float(cost)
        return raw_response, max(1, input_tokens), max(1, output_tokens)

    def _invoke_agy(self, prompt: str) -> Tuple[str, int, int]:
        """Calls the Antigravity CLI (`agy -p`) for the agy backend."""
        cmd = [self.cli_path, "-p", prompt, "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        return_code, stdout_text, stderr_text = self._run_subprocess(cmd, "agy")
        if return_code != 0:
            raise CloudArchitectError(
                f"Cloud Architect (agy) returned non-zero exit code "
                f"({return_code}): {stderr_text or stdout_text}"
            )

        try:
            payload = json.loads(stdout_text)
        except json.JSONDecodeError as e:
            raise CloudArchitectError(
                f"Failed to parse JSON output from agy: {e}. Output was: {stdout_text[:200]}"
            ) from e

        raw_response = str(
            payload.get("response")
            or payload.get("result")
            or payload.get("content")
            or ""
        )
        usage = payload.get("usage", {}) or {}
        return (
            raw_response,
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
        )

    # ==========================================================================
    # LEGACY GAUNTLET STRATEGY GENERATION (BACKWARD COMPATIBILITY)
    # ==========================================================================

    def _build_prompt(self, stripped_code: str, module_name: str) -> str:
        """Construct the AST strategy prompt for Cloud Architect."""
        return (
            "You are the Cloud Chief Architect for an automated security remediation pipeline.\n"
            "You are provided with an AST signature (interfaces, types, and function headers) of a module containing security vulnerabilities.\n"
            "All function bodies and literals have been stripped to preserve zero-knowledge code privacy.\n"
            f"Module Target: {module_name}\n\n"
            "AST Signature:\n"
            "```python\n"
            f"{stripped_code.strip()}\n"
            "```\n\n"
            "Analyze the interface, identify the architectural vulnerability pattern, and provide a structured security strategy for on-device implementation.\n"
            "Remediation Constraints:\n"
            "- Remediate within the existing module's algorithmic architecture (e.g. keep HMAC-SHA256 for HMAC tokens, keep SQLite for SQLite databases, keep thread locks for in-memory billing).\n"
            "- Do NOT suggest external asymmetric PKI/JWKS, remote auth servers, or uninstalled external libraries.\n"
            "- For database.py: Fix SQL Injection (CWE-89) strictly by using parameterized queries ('?' placeholders). Do NOT require environment variables for db_path or connection parameters. Do NOT restrict allowed usernames or roles to a hardcoded whitelist.\n"
            "- For auth.py: Secrets must be retrieved from environment variables (os.environ.get) dynamically inside functions or helper functions (e.g. get_jwt_secret(), get_admin_api_key()). In generate_token, return strictly the standard 64-character SHA-256 hex digest: hmac.new(secret.encode(), user_id.encode(), hashlib.sha256).hexdigest() (do NOT prepend user_id or return user_id.signature), and raise ValueError if JWT_SECRET is unset or empty. In authenticate_api_key, return False if ADMIN_API_KEY is unset, empty, or does not match (do NOT raise). Use hmac.compare_digest for all timing-safe comparisons. NEVER validate secrets or raise ValueError at module import time — the module must import cleanly when environment variables are unset.\n"
            "- For billing.py: Protect balance mutation with threading.Lock() or threading.RLock(). In withdraw, read balances directly or use RLock to prevent self-deadlock.\n\n"
            "Format your response exactly as follows:\n\n"
            "---THREAT_MODEL---\n"
            "<One or two sentences describing the threat model and CWE classification>\n\n"
            "---INVARIANTS---\n"
            "- <Invariant 1>\n"
            "- <Invariant 2>\n"
            "- <Invariant 3>\n\n"
            "---STRATEGY---\n"
            "<Step-by-step remediation instructions for on-device workforce builders>\n"
        )

    def _parse_strategy_response(
        self,
        raw_text: str,
        input_tokens: int,
        output_tokens: int,
    ) -> StrategyResult:
        """Parse structured sections from Gemini's response text."""
        threat_model = ""
        invariants: List[str] = []
        raw_strategy = raw_text.strip()

        tm_match = re.search(
            r"---THREAT_MODEL---\s*(.*?)(?=---INVARIANTS---|---STRATEGY---|$)",
            raw_text,
            re.DOTALL,
        )
        if tm_match:
            threat_model = tm_match.group(1).strip()

        inv_match = re.search(
            r"---INVARIANTS---\s*(.*?)(?=---STRATEGY---|$)",
            raw_text,
            re.DOTALL,
        )
        if inv_match:
            inv_text = inv_match.group(1).strip()
            for line in inv_text.splitlines():
                clean_line = line.strip()
                if clean_line.startswith(("-", "*", "•")) or re.match(r"^\d+\.", clean_line):
                    clean_line = re.sub(r"^[-*•\d\.]+\s*", "", clean_line).strip()
                    if clean_line:
                        invariants.append(clean_line)

        strat_match = re.search(
            r"---STRATEGY---\s*(.*)",
            raw_text,
            re.DOTALL,
        )
        if strat_match:
            strategy_body = strat_match.group(1).strip()
            if strategy_body:
                raw_strategy = strategy_body

        if not threat_model:
            lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
            threat_model = lines[0] if lines else "Identified security vulnerability"

        if not invariants:
            bullet_lines = re.findall(r"^[-*]\s+(.*)$", raw_text, re.MULTILINE)
            if bullet_lines:
                invariants = [b.strip() for b in bullet_lines if b.strip()]
            else:
                invariants = [
                    "Remediate identified vulnerability according to threat model",
                    "Preserve public API interfaces and type contracts",
                ]

        return StrategyResult(
            threat_model=threat_model,
            invariants=invariants,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_strategy=raw_strategy,
        )

    def generate_strategy(self, stripped_code: str, module_name: str) -> StrategyResult:
        """Invoke the Cloud Architect via codex CLI to generate remediation strategy for AST signature.

        Args:
            stripped_code: Zero-knowledge stripped AST Python source.
            module_name: Identifier of the target module (e.g. 'auth.py', 'billing.py').

        Returns:
            StrategyResult with threat model, invariants, token counts, and strategy instructions.

        Raises:
            CloudArchitectError: If the codex subprocess exits with an error or times out.
        """
        prompt = self._build_prompt(stripped_code, module_name)
        raw_response, in_tokens, out_tokens = self._invoke_cloud(prompt)
        self.last_input_tokens = in_tokens
        self.last_output_tokens = out_tokens
        return self._parse_strategy_response(raw_response, in_tokens, out_tokens)

    # Backward-compatible alias
    generate_strategic_plan = generate_strategy

    # ==========================================================================
    # UNIVERSAL R&D DAG PLANNING (FEATURE F6)
    # ==========================================================================

    def _build_rd_prompt(
        self,
        goal: str,
        workspace_outline: Optional[WorkspaceOutline] = None,
    ) -> str:
        """Formulate the structured DAG generation prompt."""
        outline_text = (
            workspace_outline.to_prompt_string()
            if workspace_outline and hasattr(workspace_outline, "to_prompt_string")
            else "# No existing workspace provided (greenfield task)."
        )

        return (
            "You are the Cloud Chief Architect for an autonomous R&D coding and experimentation engine.\n"
            "Your role is to analyze a project goal and workspace interface outline, then synthesize a structured "
            "Execution Plan DAG (Directed Acyclic Graph) of discrete, self-contained subtasks for an on-device local workforce.\n\n"
            f"GOAL:\n{goal.strip()}\n\n"
            f"WORKSPACE ARCHITECTURE OUTLINE (Zero-Knowledge AST Signatures):\n{outline_text}\n\n"
            "INSTRUCTIONS:\n"
            "1. Decompose the goal into discrete, logically ordered subtasks that form a Directed Acyclic Graph (DAG).\n"
            "2. Each subtask must be self-contained and specify:\n"
            '   - "id": A unique string identifier (e.g. "task_1", "task_2", etc.).\n'
            '   - "title": A concise, clear summary of the subtask.\n'
            '   - "description": Concrete instructions for the on-device workforce. Provide algorithmic guidance without prescribing proprietary locked-down code.\n'
            '   - "dependencies": A list of prerequisite task "id"s that must complete successfully before this task can start. The graph MUST be acyclic (no circular dependencies).\n'
            '   - "affected_files": A list of relative file paths that will be inspected, created, modified, or tested.\n'
            '   - "input_files": Existing paths required as inputs; use only exact paths present in the local file inventory.\n'
            '   - "outputs": A list of concrete deliverable paths created or updated by this task (empty only when there is no file deliverable).\n'
            '   - "success_criteria": Specific measurable conditions that define completion.\n'
            '   - "verification_command": Must return exit code 0 only when success criteria hold. For EVERY path listed in "outputs", include a separate "&&"-joined segment that checks that exact path: "test -s <path>", "grep -q <pattern> <path>", or "python3 -m py_compile <path>". Example for outputs ["src/calc.py"]: "test -s src/calc.py && python3 -m pytest tests/test_calc.py". Never use a python -c import as the output check, a file listing (rg --files, find, ls), pipes, ||, ;, or redirects.\n\n'
            "3. For analysis or research deliverables, separate confirmed facts, reported claims, inferences, contested points, and open questions. Cite local evidence with exact relative path and line numbers; label facts that need device or runtime validation.\n"
            "4. Keep each task narrow enough to complete and verify independently. Use actual paths from the supplied inventory for inputs; declare new paths under outputs.\n"
            "OUTPUT REQUIREMENTS:\n"
            "- You must respond ONLY with a valid JSON object conforming to the schema below.\n"
            "- Do NOT wrap the JSON in commentary or conversational text.\n"
            '- Do NOT introduce cycles or nonexistent task IDs in "dependencies".\n\n'
            "JSON SCHEMA:\n"
            "{\n"
            '  "tasks": [\n'
            "    {\n"
            '      "id": "task_1",\n'
            '      "title": "Title of task 1",\n'
            '      "description": "Detailed description of task 1",\n'
            '      "dependencies": [],\n'
            '      "affected_files": ["path/to/file.py"],\n'
            '      "input_files": ["README.md"],\n'
            '      "outputs": ["path/to/output.py"],\n'
            '      "success_criteria": "Observable success condition",\n'
            '      "verification_command": "test -s path/to/output.py && pytest tests/test_task1.py"\n'
            "    },\n"
            "    {\n"
            '      "id": "task_2",\n'
            '      "title": "Title of task 2",\n'
            '      "description": "Detailed description of task 2",\n'
            '      "dependencies": ["task_1"],\n'
            '      "affected_files": ["path/to/file.py"],\n'
            '      "success_criteria": "Observable success condition",\n'
            '      "verification_command": "pytest tests/test_task2.py"\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

    def _extract_json(self, raw_text: str) -> Any:
        """Safely extract and parse JSON from LLM response, stripping markdown fences if present."""
        text = raw_text.strip()

        # 1. Match markdown code fence
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if code_block_match:
            candidate = code_block_match.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                cleaned_cand = re.sub(r",\s*([\]}])", r"\1", candidate)
                try:
                    return json.loads(cleaned_cand)
                except json.JSONDecodeError:
                    pass

        # 2. Try direct json.loads
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 3. Find outermost JSON object or array
        obj_start = text.find("{")
        arr_start = text.find("[")

        if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
            obj_end = text.rfind("}")
            if obj_end > obj_start:
                candidate = text[obj_start : obj_end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    cleaned = re.sub(r",\s*([\]}])", r"\1", candidate)
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        pass

        if arr_start != -1:
            arr_end = text.rfind("]")
            if arr_end > arr_start:
                candidate = text[arr_start : arr_end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    cleaned = re.sub(r",\s*([\]}])", r"\1", candidate)
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        pass

        raise CloudArchitectError(
            f"Failed to extract valid JSON from Cloud Architect response: {raw_text[:200]}"
        )

    def _parse_dag_response(self, raw_text: str) -> ExecutionPlanDAG:
        """Deserialize JSON response into ExecutionPlanDAG and validate topology."""
        payload = self._extract_json(raw_text)

        tasks_list: List[Dict[str, Any]] = []
        if isinstance(payload, list):
            tasks_list = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            if "tasks" in payload:
                raw_tasks = payload["tasks"]
                if isinstance(raw_tasks, list):
                    tasks_list = [item for item in raw_tasks if isinstance(item, dict)]
                elif isinstance(raw_tasks, dict):
                    for tid, tval in raw_tasks.items():
                        if isinstance(tval, dict):
                            tval_copy = dict(tval)
                            if "id" not in tval_copy:
                                tval_copy["id"] = tid
                            tasks_list.append(tval_copy)
                else:
                    raise CloudArchitectError(f"'tasks' field is neither list nor dict: {type(raw_tasks)}")
            elif "id" in payload:
                tasks_list = [payload]
            else:
                for tid, tval in payload.items():
                    if isinstance(tval, dict):
                        tval_copy = dict(tval)
                        if "id" not in tval_copy:
                            tval_copy["id"] = tid
                        tasks_list.append(tval_copy)

        if not tasks_list:
            raise CloudArchitectError("Response payload contains no subtasks (empty plan)")

        dag = ExecutionPlanDAG()
        for idx, item in enumerate(tasks_list):
            tid = str(item.get("id") or f"task_{idx + 1}")
            subtask = SubTask(
                id=tid,
                title=str(item.get("title") or item.get("name") or tid),
                description=str(item.get("description") or ""),
                dependencies=[str(d) for d in (item.get("dependencies") or [])],
                affected_files=[str(f) for f in (item.get("affected_files") or item.get("files") or [])],
                input_files=[str(f) for f in (item.get("input_files") or [])],
                success_criteria=str(item.get("success_criteria") or ""),
                verification_command=str(item.get("verification_command") or item.get("command") or ""),
                outputs=[str(f) for f in (item.get("outputs") or [])],
                status="pending",
                retry_count=0,
            )
            dag.add_task(subtask)

        # Validate DAG: Kahn's topological sort and dependency check
        # This raises DAGCycleError or DAGDependencyError if invalid
        dag.validate()

        return dag

    def complete(self, prompt: str) -> Tuple[str, int, int]:
        """One free-form cloud call (recorded in ``usage_ledger``); used by the harness editor."""
        raw, in_tokens, out_tokens = self._invoke_cloud(prompt)
        return str(raw or ""), in_tokens, out_tokens

    ASSIST_RESPONSE_CHARS = 4000

    def assist_subtask(self, payload: str) -> Tuple[str, int, int]:
        """Ask for step-by-step guidance on a subtask the local worker keeps failing.

        ``payload`` was built and privacy-checked by ``CloudAssist``. The answer is
        steps or pseudocode for the local worker to implement, never complete code.
        """
        prompt = (
            "You are the planning architect of a hybrid coding system. A small local model "
            "keeps failing the subtask below. You see only signatures, docstrings and the "
            "failing tests' messages; the local model will write the code.\n"
            "Reply with at most 12 numbered steps or short pseudocode that lead to a fix "
            "satisfying the docstrings and the failing tests. Do NOT write complete code, "
            "whole functions or files.\n\n" + payload
        )
        raw, in_tokens, out_tokens = self._invoke_cloud(prompt)
        return str(raw or "").strip()[: self.ASSIST_RESPONSE_CHARS], in_tokens, out_tokens

    def plan_rd_task(
        self,
        goal: str,
        workspace_outline: Optional[WorkspaceOutline] = None,
        max_retries: int = 1,
    ) -> ExecutionPlanDAG:
        """Plan an autonomous R&D execution DAG from goal and workspace outline.

        Args:
            goal: Freeform description of the R&D goal or task.
            workspace_outline: Optional zero-knowledge workspace structural outline.
            max_retries: Maximum number of correction attempts if LLM output fails validation.

        Returns:
            Validated, cycle-free ExecutionPlanDAG.

        Raises:
            ValueError: If goal is empty.
            SecurityInvariantViolationError: If workspace outline violates code privacy.
            CloudArchitectError: If invocation or DAG construction fails.
            DAGCycleError: If cycle detected after retries exhausted.
            DAGDependencyError: If missing dependency detected after retries exhausted.
        """
        if not goal or not isinstance(goal, str) or not goal.strip():
            raise ValueError("Goal string cannot be empty")

        # 1. Pre-flight Zero-Knowledge Privacy Check (MUST NOT catch SecurityInvariantViolationError)
        if workspace_outline is not None:
            if hasattr(workspace_outline, "file_tree") and hasattr(workspace_outline, "ast_signatures"):
                verify_workspace_outline(workspace_outline)

            outline_payload = (
                workspace_outline.to_prompt_string()
                if hasattr(workspace_outline, "to_prompt_string")
                else str(workspace_outline)
            )
            verify_cloud_payload(outline_payload)

        # 2. Prompt Formulation
        prompt = self._build_rd_prompt(goal, workspace_outline)
        current_prompt = prompt
        last_error: Optional[Exception] = None
        failures: List[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        reported_cost = 0.0
        unreported_in = 0
        unreported_out = 0

        # 3. Invocation & Self-Correction Retry Loop
        for attempt in range(max_retries + 1):
            try:
                if self.execution_control is not None:
                    stop = self.execution_control.check()
                    if stop.stopped:
                        raise CloudArchitectError(f"Planning paused: {stop.reason}")
                raw_text, in_tokens, out_tokens = self._invoke_cloud(current_prompt)
                total_input_tokens += in_tokens
                total_output_tokens += out_tokens
                if self.last_call_cost_usd is not None:
                    reported_cost += self.last_call_cost_usd
                else:
                    unreported_in += in_tokens
                    unreported_out += out_tokens
                if self.execution_control is not None:
                    self.execution_control.record_remote_tokens(in_tokens + out_tokens)
                self.last_input_tokens = total_input_tokens
                self.last_output_tokens = total_output_tokens
                dag = self._parse_dag_response(raw_text)
                inventory = getattr(workspace_outline, "file_tree", []) if workspace_outline else []
                issues = PlanValidator.validate(dag, inventory)
                if issues:
                    raise CloudArchitectError(
                        "Plan validation issues: "
                        + "; ".join(f"{issue.task_id}: {issue.message}" for issue in issues)
                    )
                dag.metadata["input_tokens"] = total_input_tokens
                dag.metadata["output_tokens"] = total_output_tokens
                dag.metadata["reported_cost_usd"] = reported_cost if reported_cost > 0 else None
                dag.metadata["unreported_input_tokens"] = unreported_in
                dag.metadata["unreported_output_tokens"] = unreported_out
                return dag
            except SecurityInvariantViolationError:
                raise
            except Exception as e:
                if self.execution_control is not None and self.execution_control.check().stopped:
                    raise
                last_error = e
                failures.append(str(e))
                if attempt < max_retries:
                    current_prompt = (
                        f"{prompt}\n\n"
                        f"CRITICAL ERROR IN PREVIOUS RESPONSE: {str(e)}\n"
                        f"Please re-generate the execution plan, strictly correcting this error. "
                        f"Ensure valid JSON with 'tasks' array, valid dependencies, and NO cycles."
                    )
                else:
                    if isinstance(last_error, (DAGCycleError, DAGDependencyError)):
                        raise last_error
                    earlier = " | ".join(failures[:-1])
                    suffix = f" (earlier attempts: {earlier})" if earlier else ""
                    raise CloudArchitectError(
                        f"Failed to generate valid ExecutionPlanDAG after {max_retries + 1} attempts: {last_error}{suffix}"
                    ) from last_error

        # Fallback if loop finishes unexpectedly
        raise CloudArchitectError(f"Failed to plan R&D task: {last_error}")
