"""Cloud harness editor for the Propose and Compose steps (INV, DOM, AGG).

Follows Harness²'s split: INV owns how the agent works (procedure, tool use,
completion), DOM owns what a practitioner knows, AGG composes one harness from the
contrast between the candidates' executions. Replies are a JSON harness or
``{"keep_base": true}``; an invalid reply is retried once with the validation
errors, then the base harness is kept.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from antigravity_hybrid_orchestrator.harness.tree import (
    MAX_ALWAYS_ON_CHARS, MAX_SKILL_CHARS, MAX_SKILLS, HarnessTree,
)

SURFACE = f"""\
# THE HARNESS YOU EDIT
A small local model (Gemma 4 12B, 16K context, 2048-token turns, thinking off) solves Python
repair tasks by calling tools: read_file, write_file, patch_file, list_dir, run_command,
git_status, git_diff. Its harness has five editable components:
- system_prompt (c1): role and core procedure, always in context.
- guardrails (c2): standing rules and stop conditions, always in context.
- memory (c3): short facts true for EVERY task of this kind, always in context.
- tool_guidance (c4): how to use the tools well, always in context.
- skills (c5): name -> text, loaded on demand when the model calls read_skill(name). Each skill
  name must appear in system_prompt, guardrails or tool_guidance together with its trigger.
Limits: c1+c2+c3+c4 together <= {MAX_ALWAYS_ON_CHARS} characters; at most {MAX_SKILLS} skills of
<= {MAX_SKILL_CHARS} characters; no template braces. The tool loop, stage hints, recovery,
sandbox and test verification are frozen code you cannot change. Hidden tests exist that you
cannot see; do not try to guess or encode their contents. Every always-on character is paid on
every turn of every task, so prefer short rules and on-demand skills.
"""

MISSIONS = {
    "INV": """\
# YOUR DIMENSION: INV -- how the agent WORKS
You own procedure, tool use and completion, not subject matter. Trace the workflow in the
evidence, find its earliest divergence (wasted turns, truncated or malformed tool calls,
re-reading, stopping before verifying, incomplete fixes) and fix it with the cheapest effective
component. Do not add domain rules.""",
    "DOM": """\
# YOUR DIMENSION: DOM -- what a PRACTITIONER knows
You own the knowledge the agent lacks to write correct code for this kind of task: reading a
docstring as a contract, enumerating every stated rule (validation, edge cases, ordering) before
editing, idioms such as anchored full-string regex matching or input validation. Encode it as
checkable rules, preferably in on-demand skills with clear triggers. Do not diagnose procedure.""",
    "AGG": """\
# YOUR JOB: COMPOSE
Two candidate harnesses (INV and DOM) were executed on the same task. Compare their executions
with the base execution, keep only edits whose effect the evidence supports, resolve conflicts,
and return ONE composed harness. KEEP_BASE is a valid answer when no edit helped.""",
}

OUTPUT_CONTRACT = """\
# REPLY FORMAT
Reply with ONE fenced ```json block containing either the complete edited harness
{"system_prompt": "...", "guardrails": "...", "memory": "...", "tool_guidance": "...",
 "skills": {"name": "text"}, "notes": "what you changed and which evidence argues for it"}
or {"keep_base": true, "notes": "why no edit is justified"}.
Write the task-agnostic procedure and knowledge for this KIND of task; this harness will be
reused on other tasks. KEEP_BASE is a real answer, not a failure."""


@dataclass
class Proposal:
    dim: str
    tree: HarnessTree
    kept_base: bool
    notes: str = ""
    fell_back: bool = False
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = fenced or ([text[text.index("{"): text.rindex("}") + 1]] if "{" in text and "}" in text else [])
    for raw in reversed(candidates):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


class HarnessProposer:
    """Wraps a cloud completion function ``complete(prompt) -> (text, in_tokens, out_tokens)``."""

    def __init__(self, complete: Callable[[str], Any], *, audit: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.complete = complete
        self.audit = audit or (lambda record: None)

    def build_prompt(self, dim: str, base: HarnessTree, evidence: str, library: List[str],
                     pool: str = "", error: str = "") -> str:
        parts = [MISSIONS[dim], SURFACE,
                 "# BASE HARNESS (JSON)\n```json\n" + json.dumps(base.to_dict(), indent=1, ensure_ascii=False) + "\n```",
                 "# EVIDENCE: BASE EXECUTION\n" + evidence]
        if pool:
            parts.append("# EVIDENCE: CANDIDATE EXECUTIONS\n" + pool)
        if library:
            parts.append("# LESSONS FROM EARLIER TASKS (hypotheses, re-check against this evidence)\n"
                         + "\n".join(f"- {line}" for line in library[-8:]))
        parts.append(OUTPUT_CONTRACT)
        if error:
            parts.append(f"# YOUR PREVIOUS REPLY WAS REJECTED\n{error}\nFix it and reply again.")
        return "\n\n".join(parts)

    def propose(self, dim: str, base: HarnessTree, evidence: str, library: List[str],
                pool: str = "", attempts: int = 2) -> Proposal:
        error = ""
        tokens_in = tokens_out = calls = 0
        for _ in range(attempts):
            prompt = self.build_prompt(dim, base, evidence, library, pool, error)
            try:
                text, t_in, t_out = self.complete(prompt)
            except Exception as exc:  # backend failure: keep the base harness
                error = f"cloud call failed: {exc}"
                self.audit({"dim": dim, "error": error})
                break
            calls += 1
            tokens_in += int(t_in)
            tokens_out += int(t_out)
            self.audit({"dim": dim, "prompt_chars": len(prompt), "reply": text,
                        "input_tokens": int(t_in), "output_tokens": int(t_out)})
            data = extract_json(text or "")
            if data is None:
                error = "no JSON object found in the reply"
                continue
            notes = str(data.get("notes", ""))
            if data.get("keep_base"):
                return Proposal(dim, base, True, notes, input_tokens=tokens_in,
                                output_tokens=tokens_out, calls=calls)
            try:
                tree = HarnessTree.from_dict(data)
            except ValueError as exc:
                error = str(exc)
                continue
            problems = tree.validate()
            if problems:
                error = "; ".join(problems)
                continue
            return Proposal(dim, tree, tree == base, notes, input_tokens=tokens_in,
                            output_tokens=tokens_out, calls=calls)
        return Proposal(dim, base, True, fell_back=True, error=error[:400],
                        input_tokens=tokens_in, output_tokens=tokens_out, calls=calls)
