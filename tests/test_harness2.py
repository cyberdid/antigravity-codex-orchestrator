"""Harness² adaptation: editable harness tree, proposer, selection and recursion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from antigravity_hybrid_orchestrator.harness.tree import DEFAULT_SYSTEM_PROMPT, DEFAULT_TREE, HarnessTree
from antigravity_hybrid_orchestrator.planner.dag import SubTask
from antigravity_hybrid_orchestrator.workforce.ollama_client import WorkforceClient
from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite
from antigravity_hybrid_orchestrator.workforce.worker import AutonomousWorker


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OLLAMA_THINK", "false")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")
    monkeypatch.delenv("OLLAMA_NUM_PREDICT", raising=False)


def _run_worker(monkeypatch, tmp_path, responses, harness=None):
    sent: List[Dict[str, Any]] = []
    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4")

    def fake_post(url, payload, **kwargs):
        sent.append(json.loads(json.dumps(payload)))
        return responses.pop(0)

    monkeypatch.setattr(client, "_post_json", fake_post)
    res = AutonomousWorker(client=client, harness=harness).execute_subtask(
        SubTask(id="h", title="t", input_files=[], affected_files=[]),
        WorkspaceToolSuite(workspace_root=tmp_path),
    )
    return res, sent


def _text(content):
    return {"message": {"role": "assistant", "content": content}, "done": True,
            "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 5}


def _call(name, arguments):
    return {"message": {"role": "assistant", "content": "",
                        "tool_calls": [{"function": {"name": name, "arguments": arguments}}]},
            "done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 5}


# --- tree --------------------------------------------------------------------

def test_default_tree_renders_the_previous_system_prompt(monkeypatch, tmp_path):
    _, sent = _run_worker(monkeypatch, tmp_path, [_text("Done.")])
    assert sent[0]["messages"][0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert DEFAULT_TREE.render_system_prompt() == DEFAULT_SYSTEM_PROMPT
    assert all(t["function"]["name"] != "read_skill" for t in sent[0]["tools"])


def test_tree_round_trip_and_fingerprint_changes_with_content():
    tree = HarnessTree(guardrails="Run the tests before stopping.", skills={"regex_fix": "Use re.fullmatch."},
                       tool_guidance="For parsing tasks call read_skill regex_fix.")
    again = HarnessTree.from_dict(json.loads(json.dumps(tree.to_dict())))
    assert again == tree and again.fingerprint() == tree.fingerprint()
    assert tree.fingerprint() != DEFAULT_TREE.fingerprint()


@pytest.mark.parametrize("tree, fragment", [
    (HarnessTree(system_prompt=""), "system_prompt is empty"),
    (HarnessTree(memory="x" * 9000), "always-on text exceeds"),
    (HarnessTree(guardrails="use {{ var }}"), "template braces"),
    (HarnessTree(skills={"Bad Name": "x"}, tool_guidance="Bad Name"), "invalid skill name"),
    (HarnessTree(skills={"orphan": "steps"}), "no trigger"),
])
def test_validate_rejects_invalid_trees(tree, fragment):
    assert any(fragment in e for e in tree.validate())


def test_valid_tree_with_triggered_skill_passes():
    tree = HarnessTree(tool_guidance="When parsing text formats, call read_skill regex_fix.",
                       skills={"regex_fix": "Use re.fullmatch with an anchored pattern."})
    assert tree.validate() == []


def test_worker_renders_components_and_serves_skills(monkeypatch, tmp_path):
    tree = HarnessTree(guardrails="Never skip tests.", memory="Tests use unittest.",
                       tool_guidance="For parsing, call read_skill regex_fix.",
                       skills={"regex_fix": "SKILL BODY: use re.fullmatch."})
    res, sent = _run_worker(monkeypatch, tmp_path,
                            [_call("read_skill", {"name": "regex_fix"}), _text("Done.")], harness=tree)
    system = sent[0]["messages"][0]["content"]
    assert "# Guardrails\nNever skip tests." in system and "# Memory\nTests use unittest." in system
    assert "- regex_fix" in system and "SKILL BODY" not in system
    assert any(t["function"]["name"] == "read_skill" for t in sent[0]["tools"])
    tool_messages = [m for m in sent[1]["messages"] if m["role"] == "tool"]
    assert "SKILL BODY: use re.fullmatch." in tool_messages[-1]["content"]
    assert res.success is True


def test_unknown_skill_is_a_tool_error(monkeypatch, tmp_path):
    tree = HarnessTree(tool_guidance="call read_skill a", skills={"a": "x"})
    _, sent = _run_worker(monkeypatch, tmp_path, [_call("read_skill", {"name": "zzz"}), _text("Done.")],
                          harness=tree)
    tool_messages = [m for m in sent[1]["messages"] if m["role"] == "tool"]
    assert "Unknown skill 'zzz'" in tool_messages[-1]["content"]


# --- proposer ------------------------------------------------------------------

from antigravity_hybrid_orchestrator.harness.evidence import Rollout, render_evidence  # noqa: E402
from antigravity_hybrid_orchestrator.harness.proposer import HarnessProposer, extract_json  # noqa: E402
from antigravity_hybrid_orchestrator.harness.recursion import (  # noqa: E402
    gated_choice, run_task_step, standard_choice, utility,
)

EDITED = {"system_prompt": DEFAULT_SYSTEM_PROMPT, "guardrails": "Enumerate every docstring rule before editing.",
          "memory": "", "tool_guidance": "", "skills": {}, "notes": "agent missed validation rules"}


def _fenced(obj):
    return "Here you go\n```json\n" + json.dumps(obj) + "\n```"


class ScriptedCloud:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: List[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, 1000, 200


def test_extract_json_prefers_last_fenced_block():
    assert extract_json('```json\n{"a": 1}\n```\ntext\n```json\n{"b": 2}\n```') == {"b": 2}
    assert extract_json("no json here") is None


def test_proposer_returns_valid_edited_tree():
    cloud = ScriptedCloud([_fenced(EDITED)])
    p = HarnessProposer(cloud).propose("INV", DEFAULT_TREE, "evidence", ["T1: lesson"])
    assert not p.kept_base and p.tree.guardrails.startswith("Enumerate")
    assert p.notes == "agent missed validation rules" and p.calls == 1
    assert "# YOUR DIMENSION: INV" in cloud.prompts[0] and "T1: lesson" in cloud.prompts[0]


def test_proposer_retries_invalid_tree_then_accepts():
    bad = dict(EDITED, skills={"orphan": "x"})
    cloud = ScriptedCloud([_fenced(bad), _fenced(EDITED)])
    p = HarnessProposer(cloud).propose("DOM", DEFAULT_TREE, "evidence", [])
    assert not p.kept_base and p.calls == 2
    assert "no trigger" in cloud.prompts[1]


def test_proposer_keep_base_and_fallback():
    keep = HarnessProposer(ScriptedCloud([_fenced({"keep_base": True, "notes": "fine"})])).propose(
        "AGG", DEFAULT_TREE, "e", [])
    assert keep.kept_base and not keep.fell_back and keep.tree == DEFAULT_TREE
    broken = HarnessProposer(ScriptedCloud(["nonsense", "still nonsense"])).propose("INV", DEFAULT_TREE, "e", [])
    assert broken.kept_base and broken.fell_back and "no JSON" in broken.error
    failing = HarnessProposer(ScriptedCloud([RuntimeError("agy exited 1")])).propose("INV", DEFAULT_TREE, "e", [])
    assert failing.fell_back and "agy exited 1" in failing.error


# --- evidence, utility, selection ------------------------------------------------

def _rollout(label, ok=True, wall=100.0, trunc=0, holdout=None):
    return Rollout(label=label, task_id="T", harness_fingerprint=label * 4, visible_passed=ok,
                   holdout_passed=ok if holdout is None else holdout, worker_success=ok, wall_s=wall,
                   tokens=10000, truncations=trunc, bad_tool_calls=0, context_overflows=0, turns=6,
                   trajectory=[{"turn": 1, "tool_calls": [{"name": "read_file", "target": "a.py", "success": True}],
                                "output_tokens": 20, "eval_duration_s": 1.0}],
                   visible_output="FAIL: test_x\nAssertionError: 1 != 2", diff="--- a/a.py\n+++ b/a.py")


def test_evidence_shows_agent_view_but_never_holdout():
    r = _rollout("base", ok=False, holdout=True)
    text = render_evidence(r)
    assert "[T1] read_file a.py ok" in text and "AssertionError: 1 != 2" in text
    assert "holdout" not in text.lower()


def test_utility_orders_verified_fast_runs_first():
    assert utility(_rollout("a", wall=50)) > utility(_rollout("b", wall=200))
    assert utility(_rollout("a")) > utility(_rollout("c", ok=False, wall=10))
    assert utility(_rollout("a")) > utility(_rollout("d", trunc=2))


class FakeProposer:
    def __init__(self, trees):
        self.trees = trees
        self.calls: List[str] = []

    def propose(self, dim, base, evidence, library, pool=""):
        from antigravity_hybrid_orchestrator.harness.proposer import Proposal

        self.calls.append(dim)
        tree = self.trees.get(dim)
        return Proposal(dim, tree or base, tree is None, notes=f"{dim} notes")


def _step(rollouts, trees=None):
    trees = trees if trees is not None else {d: HarnessTree(guardrails=f"{d} rule") for d in ("INV", "DOM", "AGG")}
    seen: List[str] = []

    def rollout(tree, label):
        seen.append(label)
        return rollouts[label]

    proposer = FakeProposer(trees)
    return run_task_step("T", DEFAULT_TREE, proposer=proposer, rollout=rollout, library=[]), seen, proposer


def test_task_step_runs_propose_probe_compose_in_order():
    rs = {"base": _rollout("base"), "inv": _rollout("inv"), "dom": _rollout("dom"), "agg": _rollout("agg")}
    step, seen, proposer = _step(rs)
    assert seen == ["base", "inv", "dom", "agg"]
    assert proposer.calls == ["INV", "DOM", "AGG"]


def test_kept_base_proposals_reuse_base_rollout():
    rs = {"base": _rollout("base"), "agg": _rollout("agg")}
    step, seen, _ = _step(rs, trees={"AGG": HarnessTree(guardrails="g")})
    assert seen == ["base", "agg"]
    assert step.rollouts["inv"] is step.base


def test_h2_adopts_composition_even_if_it_fails_verification():
    rs = {"base": _rollout("base"), "inv": _rollout("inv"), "dom": _rollout("dom"),
          "agg": _rollout("agg", ok=False)}
    step, _, _ = _step(rs)
    tree, decision = standard_choice(step)
    assert tree.guardrails == "AGG rule" and decision == "ADOPT agg"


def test_h3_promotes_only_verified_and_better_candidates():
    rs = {"base": _rollout("base", wall=200), "inv": _rollout("inv", wall=60),
          "dom": _rollout("dom", wall=40, holdout=False), "agg": _rollout("agg", ok=False)}
    step, _, _ = _step(rs)
    tree, decision = gated_choice(step)
    assert tree.guardrails == "INV rule" and decision.startswith("ADOPT inv")


def test_h3_keeps_base_when_no_candidate_is_clearly_better():
    rs = {"base": _rollout("base", wall=100), "inv": _rollout("inv", wall=95),
          "dom": _rollout("dom", ok=False), "agg": _rollout("agg", ok=False)}
    step, _, _ = _step(rs)
    tree, decision = gated_choice(step)
    assert tree == DEFAULT_TREE and decision.startswith("KEEP_BASE (best inv")


# --- experiment wiring (no GPU, no cloud) ---------------------------------------

def test_experiment_script_end_to_end_with_fakes(tmp_path, monkeypatch):
    import sys as _sys

    root = Path(__file__).resolve().parent.parent
    monkeypatch.syspath_prepend(str(root / "scripts"))
    import benchmark_gemma_orion_phase2b as bench
    import run_harness2_experiment as exp
    import antigravity_hybrid_orchestrator.architect.client as client_mod

    replies = {"INV": _fenced(EDITED), "DOM": _fenced({"keep_base": True, "notes": "nothing"}),
               "AGG": _fenced(dict(EDITED, notes="composed"))}

    class FakeCloud:
        def __init__(self, backend="agy"):
            self.backend = backend

        def complete(self, prompt):
            dim = next(d for d in ("INV", "DOM", "COMPOSE") if f"# YOUR {'DIMENSION: ' + d if d != 'COMPOSE' else 'JOB: COMPOSE'}" in prompt)
            return replies["AGG" if dim == "COMPOSE" else dim], 500, 100

    calls = []

    def fake_run(policy_id, seed, task_id, setup_fn, holdout_fn, attempts=1, cloud_client=None, harness=None):
        calls.append((task_id, seed, harness.fingerprint()))
        edited = harness.guardrails != ""
        meta = {"visible_tests_passed": True, "holdout_tests_passed": True, "input_tokens": 8000,
                "output_tokens": 500, "truncated_generations": 0 if edited else 1, "invalid_tool_arguments": 0,
                "context_overflows": 0, "turn_trajectory": [], "workspace_diff": "", "visible_test_output": "OK"}
        return {"task_id": task_id, "worker_reported_success": True, "wall_clock_s": 60.0 if edited else 150.0,
                "turns": 6, "error": None, "metadata": meta}

    monkeypatch.setattr(client_mod, "CloudArchitectClient", FakeCloud)
    monkeypatch.setattr(bench, "run_single_task", fake_run)
    out = tmp_path / "h2.json"
    monkeypatch.setattr(_sys, "argv", ["x", "--output", str(out), "--eval-seeds", "303"])
    exp.main()

    data = json.loads(out.read_text())
    assert [d["decision"].split()[0] for d in data["arms"]["H3"]["decisions"]][0] == "ADOPT"
    assert data["arms"]["H2"]["decisions"][0]["decision"] == "ADOPT agg"
    assert set(data["held_out_summary"]) == {"H0", "H2", "H3"}
    assert data["held_out_summary"]["H0"]["runs"] == 4
    assert data["cloud_tokens"] > 0
    assert (out.with_suffix("") / "final_H3" / "harness.json").exists()
