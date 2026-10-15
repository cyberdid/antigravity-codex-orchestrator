"""Deterministic mock for Cloud Architect client used in testing and offline mode."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from antigravity_hybrid_orchestrator.architect import StrategyResult
from antigravity_hybrid_orchestrator.planner.dag import (
    DAGCycleError,
    DAGDependencyError,
    DAGError,
    ExecutionPlanDAG,
    SubTask,
)
from antigravity_hybrid_orchestrator.privacy.verifier import (
    SecurityInvariantViolationError,
    verify_cloud_payload,
)
from antigravity_hybrid_orchestrator.privacy.workspace_privacy import verify_workspace_outline
from antigravity_hybrid_orchestrator.workspace.scanner import WorkspaceOutline


class MockCloudArchitectClient:
    """Mock Cloud Architect providing deterministic AST strategy blueprints and DAGs."""

    STRATEGIES = {
        "auth": StrategyResult(
            threat_model="CWE-798 Hardcoded Secrets and CWE-208 Timing Leak in authentication credentials",
            invariants=[
                "Eliminate all plaintext secret literals from source code",
                "Retrieve secrets dynamically via os.environ with secure fallbacks",
                "Use hmac.compare_digest for constant-time string comparison to prevent timing side-channels",
                "Preserve public function signatures: authenticate_api_key, generate_token, verify_token",
            ],
            input_tokens=180,
            output_tokens=65,
            raw_strategy=(
                "1. Import os, hmac, and hashlib.\n"
                "2. Fetch ADMIN_API_KEY and JWT_SECRET from os.environ instead of hardcoded strings.\n"
                "3. Replace direct equality operators (==) with hmac.compare_digest in authenticate_api_key and verify_token.\n"
                "4. Maintain identical function signatures and return types."
            ),
        ),
        "billing": StrategyResult(
            threat_model="CWE-362 Race Condition in check-then-act balance mutation",
            invariants=[
                "Acquire mutual exclusion lock during balance verification and deduction",
                "Hold lock through any simulation latency or I/O delay",
                "Ensure thread-safe balance reading and atomic updates under concurrent threads",
                "Preserve BillingSystem API signatures: create_account, get_balance, withdraw",
            ],
            input_tokens=195,
            output_tokens=72,
            raw_strategy=(
                "1. Import threading.\n"
                "2. Instantiate self.lock = threading.Lock() in BillingSystem.__init__.\n"
                "3. Wrap withdraw method body inside 'with self.lock:' context manager.\n"
                "4. Guarantee atomic check-then-act state transitions."
            ),
        ),
        "database": StrategyResult(
            threat_model="CWE-89 SQL Injection via raw string formatting and f-string interpolation",
            invariants=[
                "Use SQLite parameterized queries (?) for all user input parameters",
                "Eliminate raw f-strings and format strings in cursor.execute() calls",
                "Pass parameters as tuples to cursor.execute(query, params)",
                "Preserve DatabaseManager API signatures: add_user, get_user_by_username, get_users_by_role",
            ],
            input_tokens=210,
            output_tokens=80,
            raw_strategy=(
                "1. Rewrite SQL statements using ? placeholders instead of f-strings.\n"
                "2. Pass user inputs as tuples in cursor.execute(query, (param,)).\n"
                "3. Ensure add_user, get_user_by_username, and get_users_by_role are fully parameterized.\n"
                "4. Retain table schema and query result formats."
            ),
        ),
    }

    def __init__(
        self,
        inject_cycle: bool = False,
        inject_missing_dependency: bool = False,
        inject_error: Optional[Exception] = None,
        custom_plan: Optional[ExecutionPlanDAG] = None,
    ) -> None:
        self.inject_cycle = inject_cycle
        self.inject_missing_dependency = inject_missing_dependency
        self.inject_error = inject_error
        self.custom_plan = custom_plan
        self.last_input_tokens: int = 150
        self.last_output_tokens: int = 100

    def generate_strategy(self, stripped_code: str, module_name: str) -> StrategyResult:
        """Return a deterministic StrategyResult based on target module name."""
        mod_key = module_name.lower().replace(".py", "").strip()

        for key, strategy in self.STRATEGIES.items():
            if key in mod_key:
                return strategy

        return StrategyResult(
            threat_model=f"Security vulnerabilities identified in module {module_name}",
            invariants=[
                "Remediate all input validation and security boundary flaws",
                "Preserve public API interfaces, types, and expected behaviors",
            ],
            input_tokens=150,
            output_tokens=60,
            raw_strategy="1. Audit AST signatures.\n2. Apply defense-in-depth sanitization.\n3. Verify against regression tests.",
        )

    # Backward-compatible alias
    generate_strategic_plan = generate_strategy

    def plan_rd_task(
        self,
        goal: str,
        workspace_outline: Optional[WorkspaceOutline] = None,
    ) -> ExecutionPlanDAG:
        """Return a deterministic, offline ExecutionPlanDAG in <1 millisecond."""
        if not goal or not isinstance(goal, str) or not goal.strip():
            raise ValueError("Goal string cannot be empty")

        # 1. Pre-flight Zero-Knowledge Privacy Check (Enforced offline)
        if workspace_outline is not None:
            if hasattr(workspace_outline, "file_tree") and hasattr(workspace_outline, "ast_signatures"):
                verify_workspace_outline(workspace_outline)

            outline_payload = (
                workspace_outline.to_prompt_string()
                if hasattr(workspace_outline, "to_prompt_string")
                else str(workspace_outline)
            )
            verify_cloud_payload(outline_payload)

        # 2. Check injected error
        if self.inject_error is not None:
            raise self.inject_error

        # 3. Check custom plan override
        if self.custom_plan is not None:
            return self.custom_plan

        goal_lower = goal.lower()

        # Update simulated token accounting
        self.last_input_tokens = len(goal.split()) * 4 + 120
        self.last_output_tokens = 95

        # 4. Check cycle injection
        if self.inject_cycle or "cycle" in goal_lower:
            dag = ExecutionPlanDAG()
            dag.add_task(
                SubTask(
                    id="task_1",
                    title="Cyclic Subtask 1",
                    description="Cyclic dependency test task 1",
                    dependencies=["task_2"],
                    verification_command="true",
                )
            )
            dag.add_task(
                SubTask(
                    id="task_2",
                    title="Cyclic Subtask 2",
                    description="Cyclic dependency test task 2",
                    dependencies=["task_1"],
                    verification_command="true",
                )
            )
            dag.metadata["input_tokens"] = self.last_input_tokens
            dag.metadata["output_tokens"] = self.last_output_tokens
            return dag

        # 5. Check missing dependency injection
        if self.inject_missing_dependency or "missing_dep" in goal_lower:
            dag = ExecutionPlanDAG()
            dag.add_task(
                SubTask(
                    id="task_1",
                    title="Task with missing dependency",
                    description="References nonexistent dependency",
                    dependencies=["nonexistent_task_999"],
                    verification_command="true",
                )
            )
            dag.metadata["input_tokens"] = self.last_input_tokens
            dag.metadata["output_tokens"] = self.last_output_tokens
            return dag

        # 6. Extract affected files & test command heuristics
        affected_files: List[str] = []
        test_command = "pytest tests/ -v"

        if workspace_outline is not None and getattr(workspace_outline, "file_tree", None):
            py_src = [
                f for f in workspace_outline.file_tree
                if f.endswith(".py") and not Path(f).name.startswith("test_")
            ]
            test_files = [
                f for f in workspace_outline.file_tree
                if "test" in f.lower() and f.endswith(".py")
            ]

            if py_src:
                affected_files.extend(py_src[:2])
            if test_files:
                affected_files.append(test_files[0])
                test_command = f"pytest {test_files[0]} -v"

        if not affected_files:
            affected_files = ["src/solution.py", "tests/test_solution.py"]

        # 7. Topology synthesis based on goal keywords
        dag = ExecutionPlanDAG()

        if any(k in goal_lower for k in ("fibonacci", "sieve", "algorithm")):
            dag.add_task(
                SubTask(
                    id="task_1_scaffold",
                    title="Scaffold algorithm module and unit test",
                    description=f"Create initial scaffold and unit tests for: {goal}",
                    dependencies=[],
                    affected_files=["tests/test_algorithm.py"],
                    success_criteria="Algorithm unit test scaffold created.",
                    verification_command="pytest tests/test_algorithm.py -v",
                )
            )
            dag.add_task(
                SubTask(
                    id="task_2_implement",
                    title="Implement core algorithm logic",
                    description=f"Implement core logic to solve: {goal}",
                    dependencies=["task_1_scaffold"],
                    affected_files=["src/algorithm.py"],
                    success_criteria="Algorithm logic implemented and passing tests.",
                    verification_command="pytest tests/test_algorithm.py -v",
                )
            )
            dag.add_task(
                SubTask(
                    id="task_3_benchmark",
                    title="Create performance benchmark and verify throughput",
                    description=f"Benchmark and stress-test: {goal}",
                    dependencies=["task_2_implement"],
                    affected_files=["src/benchmark.py"],
                    success_criteria="Benchmark script executed and verified.",
                    verification_command="python src/benchmark.py",
                )
            )
        elif any(k in goal_lower for k in ("bug", "fix", "reproduce", "patch")):
            dag.add_task(
                SubTask(
                    id="task_1_reproduce",
                    title="Create reproduction test case exposing the bug",
                    description=f"Construct minimal reproduction exposing: {goal}",
                    dependencies=[],
                    affected_files=["tests/test_reproduce.py"],
                    success_criteria="Reproduction test created.",
                    verification_command="pytest tests/test_reproduce.py -v",
                )
            )
            dag.add_task(
                SubTask(
                    id="task_2_patch",
                    title="Apply bugfix patch and confirm test passes",
                    description=f"Remediate defect and confirm green test: {goal}",
                    dependencies=["task_1_reproduce"],
                    affected_files=["src/patch.py"],
                    success_criteria="Patch applied and reproduction test passes.",
                    verification_command="pytest tests/test_reproduce.py -v",
                )
            )
        elif "refactor" in goal_lower:
            dag.add_task(
                SubTask(
                    id="task_1_interface",
                    title="Define updated abstract interface",
                    description=f"Define abstract base classes and interfaces: {goal}",
                    dependencies=[],
                    affected_files=["src/interfaces.py"],
                    success_criteria="Interfaces defined.",
                    verification_command="python -m py_compile src/interfaces.py",
                )
            )
            dag.add_task(
                SubTask(
                    id="task_2a_migrate_auth",
                    title="Migrate auth subsystem to new interface",
                    description="Refactor authentication layer against new interface.",
                    dependencies=["task_1_interface"],
                    affected_files=["src/auth.py"],
                    success_criteria="Auth subsystem migrated.",
                    verification_command="pytest tests/test_auth.py",
                )
            )
            dag.add_task(
                SubTask(
                    id="task_2b_migrate_db",
                    title="Migrate database subsystem to new interface",
                    description="Refactor database storage layer against new interface.",
                    dependencies=["task_1_interface"],
                    affected_files=["src/db.py"],
                    success_criteria="Database subsystem migrated.",
                    verification_command="pytest tests/test_db.py",
                )
            )
            dag.add_task(
                SubTask(
                    id="task_3_integration",
                    title="Execute comprehensive integration test suite",
                    description="Verify integration between refactored components.",
                    dependencies=["task_2a_migrate_auth", "task_2b_migrate_db"],
                    affected_files=["tests/test_integration.py"],
                    success_criteria="Integration tests passing.",
                    verification_command="pytest tests/test_integration.py",
                )
            )
        elif "diamond" in goal_lower:
            dag.add_task(
                SubTask(
                    id="task_1",
                    title="Phase 1: Setup & Environment",
                    description="Initialize test scaffold and baseline.",
                    dependencies=[],
                    affected_files=affected_files,
                    success_criteria="Scaffold initialized.",
                    verification_command=test_command,
                )
            )
            dag.add_task(
                SubTask(
                    id="task_2a",
                    title="Phase 2A: Implement Core Algorithm",
                    description="Implement main algorithmic logic.",
                    dependencies=["task_1"],
                    affected_files=affected_files[:1],
                    success_criteria="Algorithm unit tests passing.",
                    verification_command=test_command,
                )
            )
            dag.add_task(
                SubTask(
                    id="task_2b",
                    title="Phase 2B: Implement Interfaces & Types",
                    description="Implement public APIs and types.",
                    dependencies=["task_1"],
                    affected_files=affected_files[1:2] if len(affected_files) > 1 else affected_files[:1],
                    success_criteria="Type checks and interface tests passing.",
                    verification_command=test_command,
                )
            )
            dag.add_task(
                SubTask(
                    id="task_3",
                    title="Phase 3: Integration & Final Verification",
                    description="Integrate modules and run full regression suite.",
                    dependencies=["task_2a", "task_2b"],
                    affected_files=affected_files,
                    success_criteria="100% green verification suite.",
                    verification_command=test_command,
                )
            )
        elif "single" in goal_lower:
            dag.add_task(
                SubTask(
                    id="task_1",
                    title=f"Execute: {goal[:50]}",
                    description=f"Complete autonomous goal: {goal}",
                    dependencies=[],
                    affected_files=affected_files,
                    success_criteria="Goal completed with green verification.",
                    verification_command=test_command,
                )
            )
        elif "independent" in goal_lower or "parallel" in goal_lower:
            dag.add_task(
                SubTask(
                    id="task_1",
                    title=f"Independent Task 1: {goal[:30]}",
                    description="Independent component 1.",
                    dependencies=[],
                    affected_files=affected_files[:1],
                    success_criteria="Component 1 passing.",
                    verification_command=test_command,
                )
            )
            dag.add_task(
                SubTask(
                    id="task_2",
                    title=f"Independent Task 2: {goal[:30]}",
                    description="Independent component 2.",
                    dependencies=[],
                    affected_files=affected_files[1:2] if len(affected_files) > 1 else affected_files[:1],
                    success_criteria="Component 2 passing.",
                    verification_command=test_command,
                )
            )
        else:
            # Standard 3-stage linear pipeline
            dag.add_task(
                SubTask(
                    id="task_1",
                    title=f"Setup and reproduce: {goal[:40]}",
                    description=f"Set up test harness and baseline reproduction for: {goal}",
                    dependencies=[],
                    affected_files=affected_files,
                    success_criteria="Baseline test scaffold configured and executable.",
                    verification_command=test_command,
                )
            )
            dag.add_task(
                SubTask(
                    id="task_2",
                    title=f"Implement solution: {goal[:40]}",
                    description=f"Implement core logic changes to address: {goal}",
                    dependencies=["task_1"],
                    affected_files=affected_files,
                    success_criteria="Core implementation complete with passing unit tests.",
                    verification_command=test_command,
                )
            )
            dag.add_task(
                SubTask(
                    id="task_3",
                    title=f"Verify and benchmark: {goal[:40]}",
                    description=f"Run full regression suite and performance verification for: {goal}",
                    dependencies=["task_2"],
                    affected_files=affected_files,
                    success_criteria="100% green tests and zero regression errors.",
                    verification_command=test_command,
                )
            )

        dag.validate()
        dag.metadata["input_tokens"] = self.last_input_tokens
        dag.metadata["output_tokens"] = self.last_output_tokens
        return dag
