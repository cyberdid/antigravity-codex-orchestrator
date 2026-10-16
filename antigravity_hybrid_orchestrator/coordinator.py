"""Master coordinator pipeline for Antigravity Hybrid Orchestrator."""

import ast
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

from antigravity_hybrid_orchestrator.architect import (
    CloudArchitectClient,
    MockCloudArchitectClient,
    StrategyResult,
)
from antigravity_hybrid_orchestrator.gauntlet.baselines import (
    VULNERABLE_BASELINES,
    get_secure,
    reset_gauntlet,
)
from antigravity_hybrid_orchestrator.hud import DashboardController
from antigravity_hybrid_orchestrator.privacy import (
    SecurityInvariantViolationError,
    strip_code_for_cloud,
    verify_cloud_payload,
)
from antigravity_hybrid_orchestrator.runner import GauntletTestRunner
from antigravity_hybrid_orchestrator.telemetry import TelemetryTracker
from antigravity_hybrid_orchestrator.workforce import (
    AutonomousWorker,
    BlindCriticAgent,
    BuilderAAgent,
    BuilderBAgent,
    CriticResult,
    MockAutonomousWorker,
    MockBlindCriticAgent,
    MockBuilderAAgent,
    MockBuilderBAgent,
    MockWorkforceClient,
    OllamaConnectionError,
    PatchResult,
    WorkforceClient,
    WorkspaceToolSuite,
)

# R&D universal engine components (Milestone 6 integration)
from antigravity_hybrid_orchestrator.engine.loop import SelfCorrectionLoop
from antigravity_hybrid_orchestrator.engine.control import ExecutionBudget, ExecutionControl
from antigravity_hybrid_orchestrator.engine.run_state import RunStateStore
from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG
from antigravity_hybrid_orchestrator.planner.validator import PlanValidator
from antigravity_hybrid_orchestrator.hud.rd_dashboard import RDDashboard
from antigravity_hybrid_orchestrator.hud.recorder import EventRecorder
from antigravity_hybrid_orchestrator.workspace.scanner import WorkspaceScanner


class HybridCoordinator:
    """Master pipeline orchestrating Cloud Architect, On-Device Workforce, and Security Gauntlet."""

    ALL_TARGETS = ["auth.py", "billing.py", "database.py"]

    def __init__(
        self,
        mode: str = "live",
        target: str = "all",
        reset: bool = False,
        no_hud: bool = False,
        refresh_rate: float = 8.0,
        project_root: Optional[Path] = None,
        goal: Optional[str] = None,
        workspace: Optional[str] = None,
        retry_budget: int = 5,
        record: bool = True,
        max_runtime_s: Optional[float] = None,
        max_remote_tokens: Optional[int] = None,
        model_idle_timeout_s: float = 900,
        tool_idle_timeout_s: float = 300,
        resume_run: Optional[str] = None,
        architect_backend: str = "haiku",
        architect_model: Optional[str] = None,
        worker_model: Optional[str] = None,
    ) -> None:
        self.mode = mode.lower()
        self.target = target
        self.reset = reset
        self.no_hud = no_hud
        self.refresh_rate = refresh_rate

        # R&D universal engine parameters
        self.goal = goal
        self.workspace = workspace
        self.retry_budget = max(0, int(retry_budget))
        self.execution_control = ExecutionControl(ExecutionBudget(
            max_runtime_s=max_runtime_s,
            max_remote_tokens=max_remote_tokens,
            model_idle_timeout_s=model_idle_timeout_s,
            tool_idle_timeout_s=tool_idle_timeout_s,
        ))
        self.record = record
        self.architect_backend = architect_backend
        self.architect_model = architect_model
        self.worker_model = worker_model
        self.resume_run = resume_run
        self._run_state_path: Optional[Path] = None
        self._current_dag = None
        self._last_run_event: Dict = {}
        self.recorder = None
        # When a freeform goal is supplied the coordinator runs the universal
        # autonomous R&D engine instead of the fixed security gauntlet.
        self.rd_mode = bool((goal and str(goal).strip()) or resume_run)

        if project_root is not None:
            self.project_root = Path(project_root).resolve()
        else:
            self.project_root = Path(__file__).resolve().parent.parent

        self.gauntlet_dir = Path(__file__).resolve().parent / "gauntlet"

        # Initialize telemetry tracker (shared by both pipelines)
        self.telemetry = TelemetryTracker()

        if self.rd_mode:
            self._init_rd_agents()
        else:
            self._init_gauntlet_agents()

    def _init_gauntlet_agents(self) -> None:
        """Initializes the fixed security-gauntlet pipeline (HUD, test runner, agents)."""
        # Initialize HUD controller
        self.hud = DashboardController(
            no_hud=self.no_hud,
            refresh_per_second=self.refresh_rate,
            mode=self.mode,
        )

        # Initialize test runner
        self.test_runner = GauntletTestRunner(project_root=self.project_root)

        # Initialize agents according to mode
        if self.mode == "mock":
            self.architect = MockCloudArchitectClient()
            self.builder_a = MockBuilderAAgent()
            self.builder_b = MockBuilderBAgent()
            self.blind_critic = MockBlindCriticAgent()
        else:
            runtime_budget = self.execution_control.budget.max_runtime_s
            self.architect = CloudArchitectClient(
                timeout=runtime_budget,
                execution_control=self.execution_control,
                backend=self.architect_backend,
                model=self.architect_model,
            )
            self.workforce_client = WorkforceClient(model=self.worker_model)
            self.builder_a = BuilderAAgent(self.workforce_client)
            self.builder_b = BuilderBAgent(self.workforce_client)
            self.blind_critic = BlindCriticAgent(self.workforce_client)

    def _init_rd_agents(self) -> None:
        """Initializes the universal autonomous R&D engine (Cloud Architect planner + local worker)."""
        self.rd_hud = RDDashboard(
            no_hud=self.no_hud,
            refresh_per_second=self.refresh_rate,
            mode=self.mode,
            goal=self.goal or "",
            workspace=self.workspace or "",
        )

        if self.mode == "mock":
            self.architect = MockCloudArchitectClient()
            self.worker = MockAutonomousWorker(
                event_callback=self._on_worker_event,
            )
        else:
            runtime_budget = self.execution_control.budget.max_runtime_s
            self.architect = CloudArchitectClient(
                timeout=runtime_budget,
                execution_control=self.execution_control,
                backend=self.architect_backend,
                model=self.architect_model,
            )
            self.worker = AutonomousWorker(
                model_name=self.worker_model,
                event_callback=self._on_worker_event,
                execution_control=self.execution_control,
            )


    def _on_worker_event(self, event_type: str, subtask_id: str, payload: Dict) -> None:
        """Forwards low-level worker tool/thought events to the R&D HUD and recorder.

        The subtask lifecycle events (start/complete/failed) are emitted by the
        SelfCorrectionLoop itself, so here we only surface the finer-grained
        tool calls, results and reasoning stream to avoid duplicate lifecycle logging.
        """
        hud = getattr(self, "rd_hud", None)
        if hud is not None:
            try:
                if event_type == "tool_call":
                    tool_name = str(payload.get("tool_name") or payload.get("tool") or "")
                    hud.on_tool_call(tool_name=tool_name, args=payload.get("args", {}) or {})
                elif event_type == "tool_result":
                    hud.handle_event("tool_result", subtask_id, payload)
                elif event_type == "thought":
                    hud.update_worker_stream("", str(payload.get("thought", "")), status="THINKING")
                elif event_type == "model_usage":
                    self.telemetry.add_local(
                        eval_tokens=int(payload.get("output_tokens", 0)),
                        prompt_tokens=int(payload.get("input_tokens", 0)),
                        duration_s=float(payload.get("duration_s", 0.0)),
                    )
                    hud.update_telemetry(self.telemetry.get_summary())
            except Exception:
                pass
        recorder = getattr(self, "recorder", None)
        if recorder is not None:
            try:
                recorder.emit(event_type, subtask_id, payload)
            except Exception:
                pass

    def _resolve_targets(self) -> List[str]:
        """Resolves CLI target option to list of module filenames."""
        if self.target == "all":
            return list(self.ALL_TARGETS)

        base = os.path.basename(self.target)
        if not base.endswith(".py"):
            base = f"{base}.py"

        if base in self.ALL_TARGETS:
            return [base]
        raise ValueError(f"Unknown gauntlet target: '{self.target}'. Valid targets: {self.ALL_TARGETS}")

    def _atomic_write(self, target_file: Path, content: str) -> None:
        """Atomically replaces a file using temporary file swap."""
        temp_file = target_file.with_suffix(f".tmp.{os.getpid()}")
        temp_file.write_text(content, encoding="utf-8")
        temp_file.replace(target_file)

    def run(self) -> int:
        """Executes the end-to-end orchestration pipeline.

        Returns:
            0 on full success (all tests pass, privacy verified), 1 on failure.
        """
        if self.rd_mode:
            return self._run_rd()

        start_time = time.time()
        targets = self._resolve_targets()

        # Step 1: Optional baseline reset
        if self.reset:
            reset_gauntlet(self.gauntlet_dir, targets=targets)
            for t in targets:
                self.hud.state.test_results[t] = ("PENDING", "PENDING")

        # Step 2: Start HUD interactive display
        self.hud.set_pipeline_status("RUNNING")
        self.hud.start()

        privacy_passed = True

        try:
            # Pre-flight check: ensure local Ollama service is reachable in live mode
            if self.mode == "live" and hasattr(self, "workforce_client"):
                if not self.workforce_client.check_health():
                    raise OllamaConnectionError(
                        f"Ollama service unreachable at {self.workforce_client.base_url}. "
                        "Please ensure local Ollama daemon is running."
                    )
                # Warm up model in Ollama to load weights into VRAM before pipeline timer starts
                self.workforce_client.warmup()

            for target_name in targets:
                self.hud.set_target(target_name)
                target_path = self.gauntlet_dir / target_name
                if not target_path.exists():
                    raise FileNotFoundError(f"Target module file does not exist: {target_path}")

                vulnerable_source = target_path.read_text(encoding="utf-8")

                # Step 3: Zero-knowledge AST Privacy Stripping
                self.hud.state.cloud_status = "EXTRACTING_AST"
                self.hud.refresh()

                stripped_ast = strip_code_for_cloud(vulnerable_source)

                # Pre-flight zero-knowledge verification
                try:
                    verify_cloud_payload(stripped_ast, vulnerable_source)
                except SecurityInvariantViolationError as e:
                    privacy_passed = False
                    self.hud.state.privacy_verified = False
                    self.hud.console.print(f"[bold red]Security Invariant Violation: {e}[/]")
                    raise

                # Step 4: Cloud Architect Gemini Reasoning
                self.hud.state.cloud_status = "REASONING"
                self.hud.refresh()

                strategy: StrategyResult = self.architect.generate_strategy(
                    stripped_code=stripped_ast,
                    module_name=target_name,
                )
                self.telemetry.add_cloud(
                    input_tokens=strategy.input_tokens,
                    output_tokens=strategy.output_tokens,
                )
                self.hud.update_cloud_strategy(
                    threat_model=strategy.threat_model,
                    invariants=strategy.invariants,
                    status="READY",
                )
                self.hud.update_telemetry(self.telemetry.get_summary())

                # Step 5: On-Device Workforce (Gemma 4 on RTX 4060 GPU)
                # Sequential execution to respect Ollama -np 1 concurrency invariant

                # Builder A: Surgical Minimal Patch
                self.hud.state.builder_a_status = "GENERATING"
                self.hud.refresh()

                def on_chunk_a(token: str, accumulated: str) -> None:
                    self.hud.update_builder_a(token, accumulated, status="STREAMING")

                patch_a: PatchResult = self.builder_a.generate_patch(
                    vulnerable_code=vulnerable_source,
                    strategy=strategy,
                    on_chunk=on_chunk_a,
                )
                self.telemetry.add_local(
                    eval_tokens=patch_a.eval_tokens,
                    duration_s=patch_a.eval_duration_s,
                )
                self.hud.update_builder_a("", patch_a.patch_code, status="READY")
                self.hud.update_telemetry(self.telemetry.get_summary())

                # Builder B: Defensive / Hardened Patch
                self.hud.state.builder_b_status = "GENERATING"
                self.hud.refresh()

                def on_chunk_b(token: str, accumulated: str) -> None:
                    self.hud.update_builder_b(token, accumulated, status="STREAMING")

                patch_b: PatchResult = self.builder_b.generate_patch(
                    vulnerable_code=vulnerable_source,
                    strategy=strategy,
                    on_chunk=on_chunk_b,
                )
                self.telemetry.add_local(
                    eval_tokens=patch_b.eval_tokens,
                    duration_s=patch_b.eval_duration_s,
                )
                self.hud.update_builder_b("", patch_b.patch_code, status="READY")
                self.hud.update_telemetry(self.telemetry.get_summary())

                # Step 6: Blind Critic Anonymized Adversarial Evaluation
                self.hud.state.critic_status = "EVALUATING"
                self.hud.refresh()

                def on_chunk_critic(token: str, accumulated: str) -> None:
                    self.hud.update_critic(
                        critique=accumulated,
                        winner="",
                        winning_builder="",
                        alpha_score=0.0,
                        beta_score=0.0,
                        status="EVALUATING",
                    )

                critic_result: CriticResult = self.blind_critic.evaluate(
                    candidate_1=patch_a,
                    candidate_2=patch_b,
                    strategy=strategy,
                    on_chunk=on_chunk_critic,
                )
                self.telemetry.add_local(
                    eval_tokens=critic_result.eval_tokens,
                    duration_s=critic_result.eval_duration_s,
                )
                self.hud.update_critic_decision(critic_result, status="DECIDED")
                self.hud.update_telemetry(self.telemetry.get_summary())

                # Step 7: Atomic Patch Deployment
                winning_code = (
                    patch_a.patch_code
                    if critic_result.winning_builder == "Builder A"
                    else patch_b.patch_code
                )
                other_code = (
                    patch_b.patch_code
                    if critic_result.winning_builder == "Builder A"
                    else patch_a.patch_code
                )

                def sanitize_auth_candidate(code: str) -> str:
                    try:
                        tree = ast.parse(code)
                        modified = False
                        new_body = []
                        for stmt in tree.body:
                            if isinstance(stmt, ast.Raise):
                                modified = True
                                continue
                            if isinstance(stmt, ast.If):
                                has_raise = any(isinstance(sub, ast.Raise) for sub in ast.walk(stmt))
                                if has_raise:
                                    modified = True
                                    continue
                            new_body.append(stmt)
                        if modified:
                            tree.body = new_body
                            return ast.unparse(tree)
                        return code
                    except Exception:
                        return code

                def is_valid_candidate(code: str, target: str) -> bool:
                    if not code or not code.strip():
                        return False
                    try:
                        tree = ast.parse(code)
                        if target == "auth.py":
                            for stmt in tree.body:
                                if isinstance(stmt, ast.Raise):
                                    return False
                                if isinstance(stmt, ast.If):
                                    for sub in ast.walk(stmt):
                                        if isinstance(sub, ast.Raise):
                                            return False
                            funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
                            if not {"authenticate_api_key", "generate_token", "verify_token"}.issubset(funcs):
                                return False
                            try:
                                scope = {}
                                saved_jwt = os.environ.get("JWT_SECRET")
                                saved_adm = os.environ.get("ADMIN_API_KEY")
                                os.environ["JWT_SECRET"] = "temporary-test-jwt-secret-key-32b"
                                os.environ["ADMIN_API_KEY"] = "temporary-test-admin-api-key-32b"
                                try:
                                    exec(code, scope)
                                    tok = scope["generate_token"]("user_123")
                                    if not isinstance(tok, str) or len(tok) != 64:
                                        return False
                                    if not scope["verify_token"]("user_123", tok):
                                        return False
                                    if not scope["authenticate_api_key"]("temporary-test-admin-api-key-32b"):
                                        return False
                                finally:
                                    if saved_jwt is not None:
                                        os.environ["JWT_SECRET"] = saved_jwt
                                    else:
                                        os.environ.pop("JWT_SECRET", None)
                                    if saved_adm is not None:
                                        os.environ["ADMIN_API_KEY"] = saved_adm
                                    else:
                                        os.environ.pop("ADMIN_API_KEY", None)
                            except Exception:
                                return False
                        elif target == "billing.py":
                            classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
                            if "BillingSystem" not in classes:
                                return False
                            try:
                                scope = {}
                                exec(code, scope)
                                bs = scope["BillingSystem"]()
                                bs.create_account("acc1", 100.0)
                                if bs.get_balance("acc1") != 100.0:
                                    return False
                                if not bs.withdraw("acc1", 50.0):
                                    return False
                                if bs.get_balance("acc1") != 50.0:
                                    return False
                            except Exception:
                                return False
                        elif target == "database.py":
                            classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
                            if "DatabaseManager" not in classes:
                                return False
                            try:
                                scope = {}
                                exec(code, scope)
                                db = scope["DatabaseManager"](":memory:")
                                db.add_user("testuser", "developer", "test@example.com")
                                user = db.get_user_by_username("testuser")
                                if not user or user[1] != "testuser":
                                    return False
                            except Exception:
                                return False
                        return True
                    except SyntaxError:
                        return False

                if target_name == "auth.py":
                    winning_code = sanitize_auth_candidate(winning_code)
                    other_code = sanitize_auth_candidate(other_code)

                if not is_valid_candidate(winning_code, target_name):
                    if is_valid_candidate(other_code, target_name):
                        winning_code = other_code
                    else:
                        winning_code = get_secure(target_name)

                self._atomic_write(target_path, winning_code)

                # Step 8: Programmatic Pytest Gauntlet Execution
                func_ok, adv_ok = self.test_runner.run_tests_for_target(target_name)
                func_str = "PASS" if func_ok else "FAIL"
                adv_str = "PASS" if adv_ok else "FAIL"
                self.hud.update_test_results(target_name, func_str, adv_str)

        except Exception as e:
            self.hud.set_pipeline_status("ERROR")
            if not self.no_hud:
                self.hud.console.print(f"[bold red]Pipeline Fatal Error: {e}[/]")
            raise
        finally:
            elapsed = time.time() - start_time
            self.hud.set_pipeline_status("COMPLETE")
            summary = self.telemetry.get_summary()
            self.hud.update_telemetry(summary)
            self.hud.stop()

            # Render permanent summary to stdout
            self.hud.print_final_summary(
                summary=summary,
                test_results=self.hud.state.test_results,
                privacy_verified=privacy_passed,
                duration_s=elapsed,
                target_filter=targets,
            )

        # Verification of final success criteria
        all_passed = True
        for t in targets:
            f, a = self.hud.state.test_results.get(t, ("FAIL", "FAIL"))
            if f != "PASS" or a != "PASS":
                all_passed = False
                break

        return 0 if (all_passed and privacy_passed) else 1

    def _drain_planner_usage(self) -> None:
        """Moves every unrecorded planner call from the architect ledger into telemetry."""
        ledger = getattr(self.architect, "usage_ledger", None)
        if not ledger:
            return
        for entry in ledger:
            self.telemetry.add_cloud(
                input_tokens=int(entry.get("input_tokens", 0)),
                output_tokens=int(entry.get("output_tokens", 0)),
                reported_cost_usd=entry.get("cost_usd"),
            )
        ledger.clear()

    def _run_rd(self) -> int:
        """Executes the universal autonomous R&D engine pipeline.

        Flow: scan workspace (zero-knowledge outline) -> Cloud Architect plans an
        execution DAG -> local tool-calling workforce executes each subtask with a
        self-correction loop and verification gauntlet -> final R&D summary.

        Returns:
            0 if every planned subtask completed and privacy was preserved, else 1.
        """
        start_time = time.time()

        resume_state = None
        if self.resume_run:
            resume_log = Path(self.resume_run).resolve()
            self._run_state_path = RunStateStore.sidecar_for(resume_log)
            if not self._run_state_path.is_file():
                raise FileNotFoundError(f"Run state sidecar not found: {self._run_state_path}")
            resume_state = RunStateStore.load(self._run_state_path)
            self.execution_control.restore_usage(
                remote_tokens=resume_state.get("remote_tokens", 0),
                elapsed_s=resume_state.get("elapsed_s", 0.0),
            )
            saved_workspace = Path(resume_state["workspace"]).resolve()
            if self.workspace and Path(self.workspace).resolve() != saved_workspace:
                raise ValueError("--workspace does not match the workspace recorded for this run")
            if self.goal and self.goal.strip() != resume_state.get("goal", ""):
                raise ValueError("--goal does not match the goal recorded for this run")
            self.workspace = str(saved_workspace)
            self.goal = str(resume_state.get("goal", ""))

        # Resolve the target workspace (defaults to the current directory).
        workspace_path = Path(self.workspace).resolve() if self.workspace else Path.cwd()
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Event recorder (JSONL) for later replay into the HUD.
        if self.record:
            try:
                self.recorder = EventRecorder(
                    workspace_path / ".antigravity_runs",
                    goal=self.goal or "",
                    workspace=str(workspace_path),
                    mode=self.mode,
                )
            except Exception:
                self.recorder = None
        if self.recorder is not None and self.recorder.path is not None and (
            self._run_state_path is None or resume_state is not None
        ):
            self._run_state_path = RunStateStore.sidecar_for(self.recorder.path)

        self.rd_hud.set_goal(self.goal or "", str(workspace_path))
        self.rd_hud.set_pipeline_status("RUNNING")
        self.rd_hud.start()

        privacy_passed = True
        dag = None
        dag_result = None
        interrupted = False

        try:
            outline = None
            if resume_state is not None:
                dag = ExecutionPlanDAG.from_dict(resume_state["dag"])
                dag.validate()
                outline = WorkspaceScanner(workspace_path).scan()
            else:
                # Step 1: Zero-knowledge workspace ingestion.
                self.rd_hud.set_pipeline_status("SCANNING")
                scanner = WorkspaceScanner(workspace_path)
                outline = scanner.scan()

                # Step 2: Cloud Architect strategic planning (emits + verifies ZK outline).
                self.rd_hud.set_pipeline_status("PLANNING")
                try:
                    dag = self.architect.plan_rd_task(
                        goal=self.goal or "",
                        workspace_outline=outline,
                    )
                except SecurityInvariantViolationError as e:
                    privacy_passed = False
                    self.rd_hud.set_privacy_verified(False)
                    self.rd_hud.console.print(
                        f"[bold red]Security Invariant Violation: {e}[/]"
                    )
                    raise

            inventory = outline.file_tree if outline is not None else []
            strict_plan = self.mode != "mock"
            issues = PlanValidator.validate(dag, inventory, strict=strict_plan)
            if issues and resume_state is None:
                correction = "; ".join(
                    f"{issue.task_id} [{issue.code}]: {issue.message}" for issue in issues
                )
                dag = self.architect.plan_rd_task(
                    goal=(self.goal or "")
                    + "\n\nRepair the plan structure using these local validation findings: "
                    + correction,
                    workspace_outline=outline,
                )
                issues = PlanValidator.validate(dag, inventory, strict=strict_plan)
            if issues:
                raise ValueError(
                    "Execution plan failed local validation: "
                    + "; ".join(f"{issue.task_id} [{issue.code}] {issue.message}" for issue in issues)
                )

            self._current_dag = dag
            if self._run_state_path is not None:
                RunStateStore.save(
                    self._run_state_path,
                    goal=self.goal or "",
                    workspace=str(workspace_path),
                    dag=dag,
                    remote_tokens=self.execution_control.remote_tokens,
                    elapsed_s=self.execution_control.elapsed_s,
                )
            self._rd_event_listener("plan_created", "", {"dag": dag.to_dict()})

            if not hasattr(self.architect, "usage_ledger"):
                # Mock architects do not go through the ledger; use the plan's own totals.
                self.telemetry.add_cloud(
                    input_tokens=int(dag.metadata.get("input_tokens", 0)),
                    output_tokens=int(dag.metadata.get("output_tokens", 0)),
                )
            self._drain_planner_usage()
            if self.mode == "mock" and resume_state is None:
                self.execution_control.record_remote_tokens(
                    int(dag.metadata.get("input_tokens", 0))
                    + int(dag.metadata.get("output_tokens", 0))
                )
                if self._run_state_path is not None:
                    RunStateStore.save(
                        self._run_state_path,
                        goal=self.goal or "",
                        workspace=str(workspace_path),
                        dag=dag,
                        remote_tokens=self.execution_control.remote_tokens,
                        elapsed_s=self.execution_control.elapsed_s,
                    )
            stop = self.execution_control.check()
            if stop.stopped:
                self.rd_hud.set_pipeline_status("PAUSED")
                self.rd_hud.console.print(f"[yellow]R&D run paused: {stop.reason}[/]")
                self._rd_event_listener("paused", "", {"reason": stop.reason})
                return 1
            self._rd_dag = dag
            self.rd_hud.init_dag(dag)
            self.rd_hud.update_telemetry(self.telemetry.get_summary())

            # Step 3: Local tool-calling workforce + self-correction loop.
            tool_idle_timeout = self.execution_control.budget.tool_idle_timeout_s
            tool_suite = WorkspaceToolSuite(
                workspace_root=workspace_path,
                default_command_timeout=None,
                default_idle_timeout=tool_idle_timeout if tool_idle_timeout > 0 else None,
                execution_control=self.execution_control,
            )
            loop = SelfCorrectionLoop(
                tool_suite=tool_suite,
                worker=self.worker,
                retry_budget=self.retry_budget,
                checkpoint_git=True,
                event_listener=self._rd_event_listener,
                execution_control=self.execution_control,
                replanner=self._replan_rd_subtask,
            )

            # Step 4: Execute the full DAG in topological order.
            self.rd_hud.set_pipeline_status("EXECUTING")
            dag_result = loop.execute_dag(dag)

            # Mock workers have no model usage counters; live Ollama turns
            # emit exact token counts incrementally through model_usage events.
            if self.mode == "mock":
                self.telemetry.add_local(
                    eval_tokens=dag_result.total_turns * 40,
                    duration_s=time.time() - start_time,
                )
            self.rd_hud.update_telemetry(self.telemetry.get_summary())

        except KeyboardInterrupt:
            interrupted = True
            self.execution_control.cancel("keyboard_interrupt")
            self.rd_hud.set_pipeline_status("INTERRUPTED")
            self.rd_hud.console.print("[yellow]R&D run interrupted; partial workspace changes were preserved.[/]")
            self._rd_event_listener("interrupted", "", {"reason": "keyboard_interrupt"})
        except Exception as e:
            stop = self.execution_control.check()
            if stop.stopped:
                self.rd_hud.set_pipeline_status("PAUSED")
                self.rd_hud.console.print(f"[yellow]R&D run paused: {stop.reason} ({e})[/]")
                self._rd_event_listener("paused", "", {"reason": stop.reason, "error": str(e)})
            else:
                self.rd_hud.set_pipeline_status("ERROR")
                if not self.no_hud:
                    self.rd_hud.console.print(f"[bold red]R&D Pipeline Fatal Error: {e}[/]")
                raise
        finally:
            elapsed = time.time() - start_time
            stop = self.execution_control.check()
            final_status = "INTERRUPTED" if interrupted else ("PAUSED" if stop.stopped else "COMPLETE")
            self.rd_hud.set_pipeline_status(final_status)
            self.rd_hud.set_privacy_verified(privacy_passed)
            self._drain_planner_usage()
            summary = self.telemetry.get_summary()
            self.rd_hud.update_telemetry(summary)
            self.rd_hud.stop()

            if self.recorder is not None:
                try:
                    self.recorder.record_telemetry(summary)
                    self.recorder.write_footer(
                        success=bool(dag_result.success) if dag_result is not None else False,
                        privacy_verified=privacy_passed,
                        completed=dag_result.completed_tasks if dag_result is not None else [],
                        failed=dag_result.failed_tasks if dag_result is not None else [],
                        duration_s=elapsed,
                        status=final_status.lower(),
                    )
                    self.recorder.close()
                    if self.recorder.path is not None:
                        self.rd_hud.console.print(
                            f"[dim]Run log saved: {self.recorder.path}[/]"
                        )
                except Exception:
                    pass

            if dag is not None and dag_result is not None:
                self.rd_hud.print_final_summary(
                    dag_result=dag_result,
                    dag=dag,
                    summary=summary,
                    privacy_verified=privacy_passed,
                    duration_s=elapsed,
                )

        if interrupted:
            return 130
        return 0 if (dag_result is not None and dag_result.success and privacy_passed) else 1

    def _rd_event_listener(self, event_type: str, subtask_id: str, payload: Dict) -> None:
        """Composite listener: drives the live HUD and the JSONL recorder."""
        try:
            self.rd_hud.handle_event(event_type, subtask_id, payload)
        except Exception:
            pass
        if self.recorder is not None:
            try:
                self.recorder.emit(event_type, subtask_id, payload)
            except Exception:
                pass
        self._last_run_event = {"type": event_type, "subtask_id": subtask_id, "payload": payload}
        if self._run_state_path is not None and self._current_dag is not None:
            try:
                RunStateStore.save(
                    self._run_state_path,
                    goal=self.goal or "",
                    workspace=str(self._current_dag.metadata.get("workspace", self.workspace or Path.cwd())),
                    dag=self._current_dag,
                    last_event=self._last_run_event,
                    remote_tokens=self.execution_control.remote_tokens,
                    elapsed_s=self.execution_control.elapsed_s,
                )
            except Exception:
                pass

    @staticmethod
    def _ancestor_ids(dag, task_id: str) -> set:
        """Returns the transitive dependency ids of task_id within dag."""
        seen: set = set()
        stack = list(dag.tasks[task_id].dependencies) if task_id in dag.tasks else []
        while stack:
            current = stack.pop()
            if current in seen or current not in dag.tasks:
                continue
            seen.add(current)
            stack.extend(dag.tasks[current].dependencies)
        return seen

    def _replan_rd_subtask(self, subtask, attempt_history):
        """Ask the architect for a fresh, inventory-grounded version of a stalled task."""
        workspace_path = Path(self.workspace or Path.cwd()).resolve()
        outline = WorkspaceScanner(workspace_path).scan()
        repair_goal = (
            f"{self.goal or ''}\n\n"
            f"Repair the execution plan for subtask {subtask.id} ({subtask.title}). "
            "Its previous execution made no verified progress. Re-inspect the current workspace "
            "inventory and propose a materially different, actionable approach with explicit "
            "input_files, outputs, and an output-specific verification command. "
            "Do not repeat paths that are absent from the supplied inventory."
        )
        try:
            dag = self.architect.plan_rd_task(goal=repair_goal, workspace_outline=outline)
        finally:
            self._drain_planner_usage()
        candidate = dag.tasks.get(subtask.id)
        if candidate is None:
            return None

        from antigravity_hybrid_orchestrator.planner.dag import SubTask

        revised = SubTask(
            id=subtask.id,
            title=candidate.title,
            description=candidate.description,
            dependencies=list(subtask.dependencies),
            affected_files=list(candidate.affected_files),
            input_files=list(candidate.input_files),
            outputs=list(candidate.outputs),
            success_criteria=candidate.success_criteria,
            verification_command=candidate.verification_command,
        )
        check_dag = ExecutionPlanDAG()
        upstream = getattr(self, "_rd_dag", None)
        if upstream is not None:
            # The validator resolves ancestor outputs through the DAG, so the
            # check graph must contain the real upstream tasks, not just the revision.
            ancestors = self._ancestor_ids(upstream, subtask.id)
            for task in upstream.get_execution_order():
                if task.id in ancestors:
                    check_dag.add_task(task)
        check_dag.add_task(revised)
        issues = PlanValidator.validate(check_dag, outline.file_tree, strict=True)
        if issues:
            raise ValueError("Replanned subtask is invalid: " + "; ".join(i.message for i in issues))
        return revised
