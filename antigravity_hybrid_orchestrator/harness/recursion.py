"""Task-level Propose–Probe–Compose step and the two promotion rules compared.

H2 (standard Harness²): the composed harness becomes the next base, judged by the
composer's reading of the evidence only.
H3 (verifier-gated): a harness is promoted only if its execution passes trusted
verification (visible and hidden holdout tests) and beats the base on the
deterministic EfficiencyJudge utility by more than ``epsilon``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from antigravity_hybrid_orchestrator.harness.evidence import Rollout, render_evidence
from antigravity_hybrid_orchestrator.harness.proposer import HarnessProposer, Proposal
from antigravity_hybrid_orchestrator.harness.tree import HarnessTree

RolloutFn = Callable[[HarnessTree, str], Rollout]

EPSILON = 10.0


def utility(r: Rollout) -> float:
    """U = 1000·VCR − 25·truncations − 15·bad tool calls − 0.5·wall s − 0.01·tokens."""
    return (1000.0 * float(r.gate_accepted) - 25.0 * r.truncations - 15.0 * r.bad_tool_calls
            - 0.5 * r.wall_s - 0.01 * r.tokens)


@dataclass
class TaskStep:
    task_id: str
    base_tree: HarnessTree
    base: Rollout
    proposals: Dict[str, Proposal] = field(default_factory=dict)
    rollouts: Dict[str, Rollout] = field(default_factory=dict)

    def candidates(self) -> List[Tuple[str, HarnessTree, Rollout]]:
        return [(label, self.proposals[label].tree, self.rollouts[label])
                for label in ("inv", "dom", "agg") if label in self.proposals]


def run_task_step(task_id: str, base_tree: HarnessTree, *, proposer: HarnessProposer,
                  rollout: RolloutFn, library: List[str]) -> TaskStep:
    step = TaskStep(task_id, base_tree, rollout(base_tree, "base"))
    evidence = render_evidence(step.base)
    for label, dim in (("inv", "INV"), ("dom", "DOM")):
        proposal = proposer.propose(dim, base_tree, evidence, library)
        step.proposals[label] = proposal
        step.rollouts[label] = step.base if proposal.kept_base else rollout(proposal.tree, label)
    pool = "\n\n".join(
        f"## Candidate {label.upper()} (notes: {step.proposals[label].notes or 'none'})\n"
        f"Harness:\n{step.proposals[label].tree.to_dict()}\n\n{render_evidence(step.rollouts[label])}"
        for label in ("inv", "dom")
    )
    agg = proposer.propose("AGG", base_tree, evidence, library, pool=pool)
    step.proposals["agg"] = agg
    step.rollouts["agg"] = step.base if agg.kept_base else rollout(agg.tree, "agg")
    return step


def standard_choice(step: TaskStep) -> Tuple[HarnessTree, str]:
    """H2: adopt the composition, as Harness² does without ground-truth feedback."""
    agg = step.proposals["agg"]
    if agg.kept_base:
        return step.base_tree, "KEEP_BASE (composer)"
    return agg.tree, "ADOPT agg"


def gated_choice(step: TaskStep, epsilon: float = EPSILON) -> Tuple[HarnessTree, str]:
    """H3: trusted verification gate, then the deterministic utility, else KEEP_BASE."""
    accepted = [(label, tree, r) for label, tree, r in step.candidates()
                if r.gate_accepted and tree != step.base_tree]
    if not accepted:
        return step.base_tree, "KEEP_BASE (no candidate passed trusted verification)"
    label, tree, best = max(accepted, key=lambda item: utility(item[2]))
    if step.base.gate_accepted and utility(best) <= utility(step.base) + epsilon:
        return step.base_tree, (f"KEEP_BASE (best {label} U={utility(best):.1f} "
                                f"<= base U={utility(step.base):.1f} + {epsilon})")
    return tree, f"ADOPT {label} (U={utility(best):.1f} vs base {utility(step.base):.1f})"


def lesson(task_id: str, decision: str, proposal: Optional[Proposal]) -> str:
    notes = (proposal.notes if proposal else "").strip().replace("\n", " ")
    return f"{task_id}: {decision}" + (f" — {notes[:300]}" if notes else "")
