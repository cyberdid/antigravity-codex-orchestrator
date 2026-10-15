from __future__ import annotations

from antigravity_hybrid_orchestrator.engine.control import ExecutionBudget, ExecutionControl


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_control_runs_without_optional_total_budgets():
    clock = FakeClock()
    control = ExecutionControl(ExecutionBudget(), clock=clock)

    clock.now = 10_000_000

    assert control.check().stopped is False


def test_control_stops_on_runtime_budget():
    clock = FakeClock()
    control = ExecutionControl(ExecutionBudget(max_runtime_s=5), clock=clock)
    clock.now = 5

    decision = control.check()

    assert decision.stopped is True
    assert decision.reason == "runtime_budget_exceeded"


def test_control_stops_on_remote_token_budget():
    control = ExecutionControl(ExecutionBudget(max_remote_tokens=50))
    control.record_remote_tokens(50)

    decision = control.check()

    assert decision.stopped is True
    assert decision.reason == "remote_token_budget_exceeded"


def test_control_cancel_returns_operator_reason():
    control = ExecutionControl(ExecutionBudget())
    control.cancel("keyboard_interrupt")

    decision = control.check()

    assert decision.stopped is True
    assert decision.reason == "keyboard_interrupt"


def test_restored_usage_counts_toward_run_wide_budgets():
    clock = FakeClock(100)
    control = ExecutionControl(ExecutionBudget(max_runtime_s=120, max_remote_tokens=500), clock=clock)
    control.restore_usage(remote_tokens=400, elapsed_s=119)
    control.record_remote_tokens(100)

    assert control.check().reason == "remote_token_budget_exceeded"
    clock.now = 101
    assert control.check().reason == "runtime_budget_exceeded"
