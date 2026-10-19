"""Editable worker harness (Harness² components c1..c5) for AutonomousWorker.

c1 system prompt, c2 guardrails, c3 memory and c4 tool guidance are always-on text
in the system message; c5 skills are loaded on demand through the ``read_skill``
tool. Subagents, scripts and plugins (c6..c8) are frozen: the tool loop, the
recovery controller, the sandbox and the verifiers stay in Python and are never
edited by a proposer (see docs/HARNESS2_INTEGRATION_DESIGN.md).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert autonomous R&D coding workforce running locally on an "
    "RTX 4060 GPU. You accomplish the task by CALLING the provided tools "
    "(read_file, write_file, patch_file, list_dir, run_command, git_status, git_diff) — do not just "
    "describe what to do. Treat Input Files and Expected Outputs as authoritative. "
    "Never invent source paths. If a path is missing, inspect the supplied file "
    "list and report the gap instead of guessing. Do not install packages, use "
    "system package managers, or access the network; report unavailable runtime "
    "dependencies as limitations. Write real file contents with write_file. When a "
    "write_file call is required, always pass BOTH required JSON arguments with "
    "\"path\" FIRST (the exact relative path from Expected Outputs) followed by \"content\" "
    "(the full text), and emit at most ONE write_file call per turn so output is never truncated. "
    "Never call write_file with content alone. When a "
    "verification command is given, run it with run_command and fix failures. "
    "Only stop (reply with a short summary and NO tool call) once the work is "
    "done and verification passes."
)

# Always-on text (c1..c4) must leave the 16K context room for files and output.
MAX_ALWAYS_ON_CHARS = 8192  # ~2048 tokens at 4 chars/token
MAX_SKILL_CHARS = 6000
MAX_SKILLS = 6
SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
TEXT_COMPONENTS = ("system_prompt", "guardrails", "memory", "tool_guidance")

READ_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "read_skill",
        "description": "Load the full text of a named skill listed in the system prompt.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Skill name"}},
            "required": ["name"],
        },
    },
}


@dataclass(frozen=True)
class HarnessTree:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    guardrails: str = ""
    memory: str = ""
    tool_guidance: str = ""
    skills: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "system_prompt": self.system_prompt,
            "guardrails": self.guardrails,
            "memory": self.memory,
            "tool_guidance": self.tool_guidance,
            "skills": dict(sorted(self.skills.items())),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "HarnessTree":
        skills = data.get("skills") or {}
        if not isinstance(skills, Mapping):
            raise ValueError("skills must be an object of name -> text")
        values = {name: data.get(name, "") for name in TEXT_COMPONENTS}
        for name, value in values.items():
            if not isinstance(value, str):
                raise ValueError(f"{name} must be text")
        return cls(**values, skills={str(k): str(v) for k, v in skills.items()})

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def always_on_text(self) -> str:
        return "\n".join(getattr(self, name) for name in TEXT_COMPONENTS)

    def validate(self) -> List[str]:
        """Harness² structural rules; an empty list means the tree is usable."""
        errors: List[str] = []
        if not self.system_prompt.strip():
            errors.append("system_prompt is empty")
        if len(self.always_on_text()) > MAX_ALWAYS_ON_CHARS:
            errors.append(f"always-on text exceeds {MAX_ALWAYS_ON_CHARS} characters")
        if len(self.skills) > MAX_SKILLS:
            errors.append(f"more than {MAX_SKILLS} skills")
        texts = [self.always_on_text(), *self.skills.values()]
        if any("{{" in t or "{%" in t for t in texts):
            errors.append("template braces ({{ or {%) are not allowed")
        triggers = " ".join((self.system_prompt, self.guardrails, self.tool_guidance))
        for name, text in self.skills.items():
            if not SKILL_NAME.match(name):
                errors.append(f"invalid skill name {name!r}")
                continue
            if not text.strip():
                errors.append(f"skill {name!r} is empty")
            if len(text) > MAX_SKILL_CHARS:
                errors.append(f"skill {name!r} exceeds {MAX_SKILL_CHARS} characters")
            if name not in triggers:
                errors.append(f"skill {name!r} has no trigger in system_prompt, guardrails or tool_guidance")
        return errors

    def render_system_prompt(self) -> str:
        parts = [self.system_prompt.strip()]
        for title, text in (("Guardrails", self.guardrails), ("Memory", self.memory),
                            ("Tool guidance", self.tool_guidance)):
            if text.strip():
                parts.append(f"# {title}\n{text.strip()}")
        if self.skills:
            listing = "\n".join(f"- {name}" for name in sorted(self.skills))
            parts.append(
                "# Skills\nCall read_skill with a name below when its trigger applies:\n" + listing
            )
        return "\n\n".join(parts)

    def write(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "harness.json").write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return directory


DEFAULT_TREE = HarnessTree()
