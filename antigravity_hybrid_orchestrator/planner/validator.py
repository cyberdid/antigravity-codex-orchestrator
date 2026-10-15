"""Local structural checks for architect generated execution plans."""

from dataclasses import dataclass
from pathlib import PurePosixPath
import shlex
from typing import Iterable, List

from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG


@dataclass(frozen=True)
class PlanIssue:
    code: str
    task_id: str
    message: str


class PlanValidator:
    """Reject unsafe paths and plans whose verifier does not inspect outputs."""

    @staticmethod
    def validate(
        dag: ExecutionPlanDAG,
        workspace_inventory: Iterable[str],
        *,
        strict: bool = False,
    ) -> List[PlanIssue]:
        # Inventory is included in this interface so callers validate plans
        # against the exact local scan used to create them. Output paths may be
        # absent from the inventory because they are expected to be produced.
        known_files = set(workspace_inventory)
        issues: List[PlanIssue] = []
        generated_by_task = {task_id: set(task.outputs) for task_id, task in dag.tasks.items()}

        def ancestor_outputs(task_id: str, visiting: set[str] | None = None) -> set[str]:
            visiting = visiting or set()
            if task_id in visiting:
                return set()
            visiting.add(task_id)
            task = dag.tasks[task_id]
            outputs: set[str] = set()
            for dependency in task.dependencies:
                outputs.update(generated_by_task.get(dependency, set()))
                outputs.update(ancestor_outputs(dependency, visiting))
            return outputs

        for task in dag.tasks.values():
            paths = list(task.affected_files) + list(task.input_files) + list(task.outputs)
            for raw_path in paths:
                path = PurePosixPath(str(raw_path))
                if path.is_absolute() or ".." in path.parts or not str(path).strip():
                    issues.append(PlanIssue(
                        "unsafe_path", task.id,
                        f"Path must be workspace relative and may not escape the workspace: {raw_path}",
                    ))

            for input_path in task.input_files:
                if input_path not in known_files and input_path not in ancestor_outputs(task.id):
                    issues.append(PlanIssue(
                        "unknown_input", task.id,
                        f"Input path is not present in the scanned workspace inventory: {input_path}",
                    ))

            declared_outputs = set(task.outputs)
            available_from_dependencies = ancestor_outputs(task.id)
            for affected_path in task.affected_files:
                if (
                    strict
                    and affected_path not in known_files
                    and affected_path not in declared_outputs
                    and affected_path not in available_from_dependencies
                ):
                    issues.append(PlanIssue(
                        "unknown_path", task.id,
                        f"Affected path is absent from the workspace inventory and is not declared as an output: {affected_path}",
                    ))

            if strict and not task.verification_command.strip():
                issues.append(PlanIssue(
                    "missing_verification", task.id,
                    "Every executable subtask must have a non-empty verification command.",
                ))

            if task.outputs:
                command = task.verification_command.strip()
                try:
                    tokens = shlex.split(command)
                except ValueError:
                    tokens = command.split()
                # Keep the accepted verifier language intentionally small:
                # shell fallbacks, pipelines, separators, and redirects can
                # otherwise turn a failing check into a successful command.
                unsafe_shell_control = any(token in {"||", "|", ";", "&", ">", "<"} for token in tokens)
                segments: List[List[str]] = [[]]
                for token in tokens:
                    if token == "&&":
                        segments.append([])
                    else:
                        segments[-1].append(token)
                is_inventory_only = len(tokens) >= 2 and tokens[0] in {"rg", "find", "tree"} and (
                    "--files" in tokens or "-type" in tokens or tokens[0] == "tree"
                )
                for output in task.outputs:
                    has_existence_check = any(
                        segment in (["test", flag, output] for flag in ("-e", "-f", "-s"))
                        for segment in segments
                    )
                    has_content_check = any(
                        len(segment) >= 3
                        and segment[0] in {"grep", "rg"}
                        and segment[-1] == output
                        and bool(segment[-2])
                        and all(arg in {"-q", "--quiet", "-n"} for arg in segment[1:-2] if arg.startswith("-"))
                        and not any(arg.startswith("-") for arg in segment[-2:-1])
                        for segment in segments
                    )
                    has_compile_check = any(
                        segment in (["python", "-m", "py_compile", output], ["python3", "-m", "py_compile", output])
                        for segment in segments
                    )
                    if (
                        not (has_existence_check or has_content_check or has_compile_check)
                        or is_inventory_only
                        or unsafe_shell_control
                    ):
                        issues.append(PlanIssue(
                            "output_not_verified", task.id,
                            f"Verification command must check declared output '{output}' directly; "
                            "a file listing is not proof that the deliverable exists or is valid.",
                        ))
        return issues
