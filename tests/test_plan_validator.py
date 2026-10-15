import pytest

from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG, SubTask
from antigravity_hybrid_orchestrator.planner.validator import PlanValidator


def test_plan_validator_rejects_file_list_as_only_verifier_for_declared_report():
    dag = ExecutionPlanDAG()
    dag.add_task(SubTask(
        id="report", title="Write overview", outputs=["docs/overview.md"],
        affected_files=["docs/overview.md"], verification_command="rg --files docs",
    ))

    issues = PlanValidator.validate(dag, ["README.md"])

    assert any(issue.code == "output_not_verified" for issue in issues)


def test_plan_validator_accepts_verifier_that_checks_declared_output():
    dag = ExecutionPlanDAG()
    dag.add_task(SubTask(
        id="report", title="Write overview", outputs=["docs/overview.md"],
        affected_files=["docs/overview.md"],
        verification_command="test -s docs/overview.md && grep -q '^# Overview' docs/overview.md",
    ))

    assert PlanValidator.validate(dag, []) == []


def test_plan_validator_rejects_command_that_only_mentions_output_path():
    dag = ExecutionPlanDAG()
    dag.add_task(SubTask(
        id="report", title="Write report", outputs=["docs/report.md"],
        affected_files=["docs/report.md"], verification_command="echo docs/report.md",
    ))

    assert any(issue.code == "output_not_verified" for issue in PlanValidator.validate(dag, []))


@pytest.mark.parametrize("command", [
    "echo docs/report.md | grep -q docs/report.md",
    "test -s docs/report.md || true",
])
def test_plan_validator_rejects_shell_bypasses_for_output_verification(command):
    dag = ExecutionPlanDAG()
    dag.add_task(SubTask(
        id="report", title="Write report", outputs=["docs/report.md"],
        affected_files=["docs/report.md"], verification_command=command,
    ))

    assert any(issue.code == "output_not_verified" for issue in PlanValidator.validate(dag, []))


def test_plan_validator_rejects_unsafe_output_paths():
    dag = ExecutionPlanDAG()
    dag.add_task(SubTask(
        id="report", title="Write report", outputs=["../outside.md"],
        affected_files=["../outside.md"], verification_command="test -s ../outside.md",
    ))

    assert any(issue.code == "unsafe_path" for issue in PlanValidator.validate(dag, []))


def test_plan_validator_rejects_input_paths_missing_from_local_inventory():
    dag = ExecutionPlanDAG()
    dag.add_task(SubTask(
        id="inspect", title="Inspect input", input_files=["missing/source.py"],
        verification_command="true",
    ))

    assert any(
        issue.code == "unknown_input" for issue in PlanValidator.validate(dag, ["README.md"])
    )


def test_plan_validator_accepts_input_generated_by_dependency():
    dag = ExecutionPlanDAG()
    dag.add_task(SubTask(
        id="produce", title="Produce", outputs=["build/intermediate.json"],
        affected_files=["build/intermediate.json"], verification_command="test -s build/intermediate.json",
    ))
    dag.add_task(SubTask(
        id="consume", title="Consume", dependencies=["produce"],
        input_files=["build/intermediate.json"], verification_command="true",
    ))

    assert not any(
        issue.code == "unknown_input" for issue in PlanValidator.validate(dag, [], strict=True)
    )


def test_plan_validator_rejects_nonexistent_affected_path_without_output_declaration():
    dag = ExecutionPlanDAG()
    dag.add_task(SubTask(
        id="inspect", title="Inspect source", affected_files=["app/main.py"],
        verification_command="python -m py_compile app/main.py",
    ))

    assert any(
        issue.code == "unknown_path" for issue in PlanValidator.validate(dag, ["README.md"], strict=True)
    )


def test_plan_validator_rejects_missing_verification_command():
    dag = ExecutionPlanDAG()
    dag.add_task(SubTask(id="task", title="Do work"))

    assert any(
        issue.code == "missing_verification" for issue in PlanValidator.validate(dag, [], strict=True)
    )
