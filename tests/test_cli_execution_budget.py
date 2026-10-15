from antigravity_hybrid_orchestrator.cli import build_parser


def test_cli_run_budgets_default_to_unlimited():
    args = build_parser().parse_args(["--goal", "inspect", "--mode", "mock"])

    assert args.max_runtime is None
    assert args.max_remote_tokens is None
    assert args.model_idle_timeout == 900
    assert args.tool_idle_timeout == 300


def test_cli_accepts_operator_selected_budgets():
    args = build_parser().parse_args([
        "--goal", "inspect", "--max-runtime", "120",
        "--max-remote-tokens", "5000", "--model-idle-timeout", "0",
        "--tool-idle-timeout", "45",
    ])

    assert args.max_runtime == 120
    assert args.max_remote_tokens == 5000
    assert args.model_idle_timeout == 0
    assert args.tool_idle_timeout == 45


def test_cli_accepts_resume_log_without_a_new_goal(tmp_path):
    run_log = tmp_path / "run.jsonl"
    run_log.write_text("{}\n")
    args = build_parser().parse_args(["--resume", str(run_log), "--workspace", str(tmp_path)])

    assert args.resume == str(run_log)
