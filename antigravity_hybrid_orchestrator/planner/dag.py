"""Execution DAG data model and Kahn's topological sort for Antigravity Orchestrator."""

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Set, Union


class DAGError(Exception):
    """Base exception for all DAG and task execution plan errors."""
    pass


class DAGCycleError(DAGError):
    """Raised when a dependency cycle is detected in the ExecutionPlanDAG."""
    pass


class DAGDependencyError(DAGError):
    """Raised when a task references an undefined or missing dependency."""
    pass


class DAGTaskNotFoundError(DAGError, KeyError):
    """Raised when a requested task ID is not found in the DAG."""
    pass


VALID_SUBTASK_STATUSES: Set[str] = {
    "pending",
    "in_progress",
    "completed",
    "failed",
}


@dataclass
class SubTask:
    """Discrete unit of work within an ExecutionPlanDAG.

    Attributes:
        id: Unique task identifier (e.g. 'task_1', 'setup_db').
        title: Human-readable task title.
        description: Abstract guidance and implementation instructions.
        dependencies: Task IDs that must be completed before this task can start.
        affected_files: Files expected to be created, edited, or inspected.
        success_criteria: Verifiable conditions required for completion.
        verification_command: Bash command executed to verify task success.
        status: Current execution status ('pending', 'in_progress', 'completed', 'failed').
        retry_count: Number of attempted retries during self-correction.
    """

    id: str
    title: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    affected_files: List[str] = field(default_factory=list)
    input_files: List[str] = field(default_factory=list)
    success_criteria: str = ""
    verification_command: str = ""
    outputs: List[str] = field(default_factory=list)
    status: str = "pending"
    retry_count: int = 0

    def __post_init__(self) -> None:
        self.id = str(self.id).strip()
        if not self.id:
            raise ValueError("SubTask 'id' must be a non-empty string.")

        self.title = str(self.title).strip()
        self.description = str(self.description).strip()
        self.success_criteria = str(self.success_criteria).strip()
        self.verification_command = str(self.verification_command).strip()

        # Deduplicate while preserving order
        self.dependencies = list(
            dict.fromkeys(str(d).strip() for d in self.dependencies if str(d).strip())
        )
        self.affected_files = list(
            dict.fromkeys(str(f).strip() for f in self.affected_files if str(f).strip())
        )
        self.outputs = list(dict.fromkeys(str(f).strip() for f in self.outputs if str(f).strip()))
        self.input_files = list(dict.fromkeys(str(f).strip() for f in self.input_files if str(f).strip()))

        status_raw = str(self.status).lower().strip()
        normalization_map = {
            "running": "in_progress",
            "verifying": "in_progress",
            "repairing": "in_progress",
            "done": "completed",
            "pass": "completed",
            "passed": "completed",
            "success": "completed",
            "fail": "failed",
            "error": "failed",
        }
        self.status = normalization_map.get(status_raw, status_raw)
        if self.status not in VALID_SUBTASK_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'. Must be one of: {sorted(VALID_SUBTASK_STATUSES)}"
            )

        try:
            self.retry_count = int(self.retry_count)
        except (ValueError, TypeError):
            self.retry_count = 0

        if self.retry_count < 0:
            raise ValueError(f"retry_count cannot be negative, got {self.retry_count}")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes SubTask to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "affected_files": list(self.affected_files),
            "input_files": list(self.input_files),
            "success_criteria": self.success_criteria,
            "verification_command": self.verification_command,
            "outputs": list(self.outputs),
            "status": self.status,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SubTask":
        """Instantiates a SubTask from a dictionary with resilient type coercion."""
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict for SubTask, got {type(d).__name__}")

        if "id" not in d or not str(d["id"]).strip():
            raise ValueError("SubTask dictionary must contain a non-empty 'id' field.")

        status_raw = str(d.get("status", "pending")).lower().strip()
        normalization_map = {
            "running": "in_progress",
            "verifying": "in_progress",
            "repairing": "in_progress",
            "done": "completed",
            "pass": "completed",
            "passed": "completed",
            "success": "completed",
            "fail": "failed",
            "error": "failed",
        }
        status = normalization_map.get(status_raw, status_raw)
        if status not in VALID_SUBTASK_STATUSES:
            status = "pending"

        deps_raw = d.get("dependencies", [])
        if isinstance(deps_raw, str):
            deps = [deps_raw.strip()] if deps_raw.strip() else []
        elif isinstance(deps_raw, (list, tuple, set)):
            deps = [str(x).strip() for x in deps_raw if str(x).strip()]
        else:
            deps = []

        files_raw = d.get("affected_files", d.get("files", []))
        if isinstance(files_raw, str):
            files = [files_raw.strip()] if files_raw.strip() else []
        elif isinstance(files_raw, (list, tuple, set)):
            files = [str(x).strip() for x in files_raw if str(x).strip()]
        else:
            files = []

        inputs_raw = d.get("input_files", [])
        if isinstance(inputs_raw, str):
            input_files = [inputs_raw.strip()] if inputs_raw.strip() else []
        elif isinstance(inputs_raw, (list, tuple, set)):
            input_files = [str(x).strip() for x in inputs_raw if str(x).strip()]
        else:
            input_files = []

        try:
            retry_count = int(d.get("retry_count", 0))
        except (ValueError, TypeError):
            retry_count = 0

        return cls(
            id=str(d["id"]).strip(),
            title=str(d.get("title", d.get("name", ""))),
            description=str(d.get("description", "")),
            dependencies=deps,
            affected_files=files,
            input_files=input_files,
            success_criteria=str(d.get("success_criteria", "")),
            verification_command=str(d.get("verification_command", d.get("command", ""))),
            outputs=[str(f) for f in d.get("outputs", [])],
            status=status,
            retry_count=max(0, retry_count),
        )


@dataclass
class ExecutionPlanDAG:
    """Directed Acyclic Graph of SubTasks driving R&D task execution.

    Attributes:
        tasks: Mapping of task ID to SubTask instance.
        metadata: Arbitrary plan metadata (e.g. token counts, timestamp, model).
    """

    tasks: Dict[str, SubTask] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.tasks, list):
            task_dict: Dict[str, SubTask] = {}
            for t in self.tasks:
                if t.id in task_dict:
                    raise ValueError(f"Duplicate task id: '{t.id}'")
                task_dict[t.id] = t
            self.tasks = task_dict
        elif not isinstance(self.tasks, dict):
            self.tasks = {}

        if not isinstance(self.metadata, dict):
            self.metadata = {}

    def add_task(self, task: SubTask, replace: bool = False) -> None:
        """Adds a SubTask to the execution DAG.

        Args:
            task: SubTask instance to register.
            replace: If False, raises ValueError on duplicate task ID.

        Raises:
            TypeError: If task is not a SubTask instance.
            ValueError: If task ID already exists and replace is False.
        """
        if not isinstance(task, SubTask):
            raise TypeError(f"Expected SubTask instance, got {type(task).__name__}")
        if task.id in self.tasks and not replace:
            raise ValueError(f"Duplicate task id: '{task.id}'")
        self.tasks[task.id] = task

    def get_task(self, task_id: str) -> SubTask:
        """Retrieves a SubTask by ID, raising DAGTaskNotFoundError if absent."""
        if task_id not in self.tasks:
            raise DAGTaskNotFoundError(f"Task '{task_id}' not found in DAG.")
        return self.tasks[task_id]

    def _validate_dependencies(self) -> None:
        """Verifies all task dependencies exist in the DAG and detects self-cycles."""
        for task_id, task in self.tasks.items():
            for dep_id in task.dependencies:
                if dep_id == task_id:
                    raise DAGCycleError(
                        f"Task '{task_id}' depends on itself (self-cycle detected)."
                    )

        missing: List[str] = []
        for task_id, task in self.tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    missing.append(f"Task '{task_id}' -> missing '{dep_id}'")
        if missing:
            raise DAGDependencyError(
                f"Missing dependencies in ExecutionPlanDAG: {', '.join(missing)}"
            )

    def validate(self) -> bool:
        """Validates that all dependencies exist and that graph contains no cycles.

        Returns:
            True if graph is complete, dependency-valid, and acyclic.

        Raises:
            DAGDependencyError: If a task references an undefined dependency.
            DAGCycleError: If the graph contains a circular dependency.
        """
        self._validate_dependencies()
        self.get_execution_order()
        return True

    def get_execution_order(self) -> List[SubTask]:
        """Calculates topological execution order using Kahn's algorithm.

        Returns:
            List of SubTasks in valid topological dependency order.
            Deterministic tie-breaking sorts candidate tasks by ID.

        Raises:
            DAGDependencyError: If a task references a missing dependency.
            DAGCycleError: If the graph contains a dependency cycle.
        """
        self._validate_dependencies()

        if not self.tasks:
            return []

        # in_degree: count of unmet prerequisites
        in_degree: Dict[str, int] = {
            task_id: len(task.dependencies) for task_id, task in self.tasks.items()
        }

        # dependents: prereq_id -> list of tasks waiting for prereq_id
        dependents: Dict[str, List[str]] = {task_id: [] for task_id in self.tasks}
        for task_id, task in self.tasks.items():
            for dep_id in task.dependencies:
                dependents[dep_id].append(task_id)

        # Initial queue: tasks with in_degree == 0 (no prerequisites)
        # Sort queue by ID for deterministic topological order
        queue: List[str] = sorted(
            [tid for tid, deg in in_degree.items() if deg == 0]
        )

        execution_order: List[SubTask] = []

        while queue:
            curr_id = queue.pop(0)
            execution_order.append(self.tasks[curr_id])

            for dependent_id in sorted(dependents[curr_id]):
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)
                    queue.sort()

        if len(execution_order) != len(self.tasks):
            cyclic_ids = sorted([tid for tid, deg in in_degree.items() if deg > 0])
            raise DAGCycleError(
                f"Cycle detected in ExecutionPlanDAG involving tasks: {cyclic_ids}"
            )

        return execution_order

    def get_ready_tasks(
        self, completed_task_ids: Optional[Set[str]] = None
    ) -> List[SubTask]:
        """Returns all tasks whose dependencies are satisfied and not yet completed.

        Args:
            completed_task_ids: Set of completed task IDs. If None, defaults to
                tasks in self.tasks where status == 'completed'.

        Returns:
            List of SubTasks eligible for execution, sorted by ID for determinism.
        """
        if completed_task_ids is None:
            completed_set = {
                t.id for t in self.tasks.values() if t.status == "completed"
            }
        else:
            completed_set = set(completed_task_ids)

        ready: List[SubTask] = []
        for task_id in sorted(self.tasks.keys()):
            task = self.tasks[task_id]
            if task_id in completed_set or task.status == "completed":
                continue
            if set(task.dependencies).issubset(completed_set):
                ready.append(task)

        return ready

    def mark_completed(self, task: Union[str, SubTask]) -> None:
        """Marks a subtask as completed in the DAG."""
        task_id = task.id if isinstance(task, SubTask) else str(task)
        if task_id in self.tasks:
            self.tasks[task_id].status = "completed"

    def mark_failed(self, task: Union[str, SubTask]) -> None:
        """Marks a subtask as failed in the DAG."""
        task_id = task.id if isinstance(task, SubTask) else str(task)
        if task_id in self.tasks:
            self.tasks[task_id].status = "failed"

    def is_completed(self) -> bool:
        """Returns True if all tasks in the DAG are marked completed."""
        if not self.tasks:
            return True
        return all(t.status == "completed" for t in self.tasks.values())

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the ExecutionPlanDAG into a dictionary."""
        return {
            "tasks": [task.to_dict() for task in self.tasks.values()],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, data: Union[Dict[str, Any], List[Any]]
    ) -> "ExecutionPlanDAG":
        """Deserializes ExecutionPlanDAG from dict or list representations."""
        metadata: Dict[str, Any] = {}
        tasks_raw = data

        if isinstance(data, dict):
            metadata = dict(data.get("metadata", {}))
            tasks_raw = data.get("tasks", data)

        dag = cls(metadata=metadata)

        if isinstance(tasks_raw, dict):
            for key, val in tasks_raw.items():
                if isinstance(val, dict):
                    task_dict = dict(val)
                    if "id" not in task_dict:
                        task_dict["id"] = key
                    dag.add_task(SubTask.from_dict(task_dict))
        elif isinstance(tasks_raw, list):
            for item in tasks_raw:
                if isinstance(item, dict):
                    dag.add_task(SubTask.from_dict(item))
        else:
            raise TypeError(
                f"Expected dict or list for ExecutionPlanDAG, got {type(data).__name__}"
            )

        return dag

    def to_json(self, indent: int = 2) -> str:
        """Serializes the ExecutionPlanDAG into a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "ExecutionPlanDAG":
        """Deserializes ExecutionPlanDAG from JSON string, stripping markdown fences."""
        cleaned = s.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Resilient fallback: scan for outermost JSON object {...} or array [...]
            obj_start = cleaned.find("{")
            arr_start = cleaned.find("[")
            candidate = None
            if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
                obj_end = cleaned.rfind("}")
                if obj_end > obj_start:
                    candidate = cleaned[obj_start : obj_end + 1]
            elif arr_start != -1:
                arr_end = cleaned.rfind("]")
                if arr_end > arr_start:
                    candidate = cleaned[arr_start : arr_end + 1]

            if candidate:
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    cleaned_cand = re.sub(r",\s*([\]}])", r"\1", candidate)
                    try:
                        parsed = json.loads(cleaned_cand)
                    except json.JSONDecodeError as exc:
                        raise DAGError(f"Failed to parse DAG JSON: {exc}") from exc
            else:
                raise DAGError(f"Failed to parse DAG JSON from: {cleaned[:100]}")

        return cls.from_dict(parsed)

    def __len__(self) -> int:
        return len(self.tasks)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self.tasks

    def __getitem__(self, task_id: str) -> SubTask:
        return self.get_task(task_id)

    def __iter__(self):
        return iter(self.get_execution_order())

    @property
    def task_ids(self) -> List[str]:
        """Returns list of all task IDs in insertion order."""
        return list(self.tasks.keys())

    def reset_status(self) -> None:
        """Resets all tasks to 'pending' with retry_count = 0."""
        for task in self.tasks.values():
            task.status = "pending"
            task.retry_count = 0

    def to_mermaid(self) -> str:
        """Generates Mermaid diagram definition for visual inspection."""
        lines = ["graph TD"]
        for task in self.tasks.values():
            label = task.title.replace('"', "'") if task.title else task.id
            lines.append(f'    {task.id}["{label}"]')
            for dep in task.dependencies:
                lines.append(f"    {dep} --> {task.id}")
        return "\n".join(lines)
