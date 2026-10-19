# Technical Design Document: Verifier-Gated Recursive Harnessing (`OrchestratorSubstrate` for Harness²)

## 1. Objective & Architectural Scope

Following the completion of **GEMMA-ORION Phase 1** (`G0 / G1 / G2`), this document specifies the technical design and experimental protocol for integrating Google Research's **Harness²** (`arXiv:2610.14224v1`, `google-research/harness2`) with `antigravity-codex-orchestrator`.

### Core Principle
We **do not** embed Harness²'s `Propose → Probe → Compose` recursion inside `SelfCorrectionLoop`. Instead, we implement a clean, isolated adapter (`OrchestratorSubstrate`) that implements the `harness2.substrate.base.Substrate` and `harness2.core.spec.BenchmarkAdapter` protocols around `AutonomousWorker`, while using **Trusted RIE (`ResearchIntegrityEvaluator`)** as a mandatory **verification gate** before an objective multi-criteria **Efficiency Judge** selects and promotes any harness mutation.

---

## 2. Editable vs. Immutable Surface (`SurfaceSpec`)

Harness² defines an 8-component surface (`c1..c8`). To prevent reward hacking, sandbox escape, or evaluator tampering by autonomous worker/proposer edits, we partition the surface into **Phase-1 Editable Components (Text/Skill Layer)** and **Strictly Immutable / Frozen Components**:

| Component ID | Harness² Role | Phase-1 Policy | Security & Isolation Rationale |
|---|---|---|---|
| `c1: systemprompt.md` | Core worker role & execution procedure | **EDITABLE** (bounded token cap $\le 1200$ tokens) | Controls reasoning conciseness and tool-First discipline. Rendered without Jinja `{{` execution. |
| `c2: guardrails.md` | Standing operational guardrails | **EDITABLE** (append-only on top of frozen safety core) | Frozen base security rules cannot be deleted; proposer may add task-level stop conditions. |
| `c3: LongTermMEMORY.md` | Verified workspace/domain facts | **EDITABLE (Verifier-Gated)** | Only promoted to library if rollout passes **Trusted RIE (`ACCEPTED`)**. |
| `c4: tool_guidance.md` | Tool usage rules (`read_file`, `patch_file`, `run_command`) | **EDITABLE** | Optimizes tool call formatting and prevents redundant file reads. |
| `c5: skills/<name>/SKILL.md` | On-demand procedural workflows | **EDITABLE** (requires verbatim trigger in `c1`/`c2`) | Loaded on demand to avoid always-on context bloat (`ALWAYS-ON PROSE IS THE EXPENSIVE OPTION`). |
| `c6: sub_agents/` | Delegated subagents | **DISABLED (`allow_subagents=False`)** | Disabled on single-GPU 8 GB VRAM (RTX 4060) to prevent concurrent KV-cache OOM. |
| `c7: scripts/<name>.py` | Executable helper scripts | **FROZEN in Phase 1** | Prevents untrusted scripts from reading hidden oracle tests or faking benchmark outputs. |
| `c8: plugin/<name>.ts` | Tool-loop middleware hooks | **FROZEN in Phase 1** | Middleware (`ContextCompactor`, truncation guard, `WorkspaceToolSuite` path sandbox) stays locked in Python core. |

### Explicitly Forbidden Targets (`leak_globs` & Sandbox Policy)
- `antigravity_hybrid_orchestrator/integrity/*` (`ResearchIntegrityEvaluator`, `EvidenceStore`, `report_builder.py`)
- Hidden oracle tests and independent verification fixtures
- `DockerSandbox` / `SubprocessSandbox` execution policies and timeouts
- Trusted evidence directories (`orion2-run/.rie_evidence/*`, `claims.json` schemas)

---

## 3. `OrchestratorSubstrate` Protocol Design

```python
@dataclass(frozen=True)
class OrchestratorHarnessTree:
    """Materialized c1..c5 text/skill harness for AutonomousWorker."""
    system_prompt: str
    guardrails: str
    memory: str
    tool_guidance: str
    skills: dict[str, str]          # skill_name -> SKILL.md content
    adaptive_think_policy: str      # "off" | "on" | "adaptive_by_subtask"
    fingerprint: str                # SHA-256 of canonical tree
```

### Implementer Contract (`antigravity_hybrid_orchestrator/harness/substrate.py` — Prototype Only)
1. **`surface() -> SurfaceSpec`**:
   - Validates that `c1 + c2 + c3 + c4` always-on tokens $\le 2048$ tokens so `preflight_context_budget` retains $\ge 10\text{K}$ tokens for workspace files and `num_predict` under `OLLAMA_NUM_CTX=16384`.
   - Enforces Harness²'s 3 hard validation rules:
     1. Every `skills/<name>/SKILL.md` must have an explicit trigger line in `systemprompt.md` or `guardrails.md`.
     2. No Jinja template braces (`{{` or `{%`) in always-on text.
     3. No executable scripts/plugins in Phase 1 (`c7`, `c8` rejected at `validate_tree()`).
2. **`materialise(tree_dir, spec, ctx) -> Install`**:
   - Stages an isolated temporary workspace copy for each `run_id`, injecting the candidate `OrchestratorHarnessTree` into `AutonomousWorker` without touching the repository's base files.
3. **`execute(task_ref, install, run_ids, timeout) -> list[Path]`**:
   - Runs `SelfCorrectionLoop` + `AutonomousWorker` (`gemma4:12b`) inside the isolated workspace and writes `trajectory.jsonl` and `telemetry.json`.
4. **`read_trajectory(stream_path, *, obs_trunc, max_chars, args_trunc=400) -> str`**:
   - Renders `AutonomousWorker` turns into Harness²'s canonical `[ACTION] / [OBSERVATION] / [ERROR]` grammar, including `[TRUNCATED_NUM_PREDICT]` and `[CONTEXT_COMPACTED]` markers so the proposer can diagnose thinking-budget starvation directly.
5. **`preflight() -> None`**:
   - Verifies local Ollama reachability, `gemma4:12b` digest pin, available GPU VRAM ($\ge 6.5\text{ GiB}$ free before load), and sandbox isolation.

---

## 4. Verifier-Gated Selection & Objective Promotion Criteria

Standard Harness² uses an LLM selector (`core/selection.py`) to compare candidate trajectories without ground-truth scores. In scientific R&D orchestration, subjective LLM selection is vulnerable to plausible-looking but invalid artifacts.

We replace single-stage subjective selection with a **Two-Stage Verifier-Gated Selector**:

### Stage 1: Mandatory Hard Gate — Trusted RIE (`ResearchIntegrityEvaluator`)
Every candidate rollout $c \in \{1 \dots K\}$ is evaluated by the immutable `ResearchIntegrityEvaluator` (unit verification + independent mathematical oracle + `validate_claims` + `validate_narrative`):
$$\text{Gate}(c) \in \{\text{ACCEPTED}, \text{REJECTED}\}$$
- Any candidate with $\text{Gate}(c) = \text{REJECTED}$ or `context_overflows > 0` is **immediately disqualified** from promotion into `LongTermMEMORY.md` or the domain harness library.

### Stage 2: Objective Multi-Criteria Utility Function (`EfficiencyJudge`)
Among candidates where $\text{Gate}(c) = \text{ACCEPTED}$, we rank candidates deterministically by lexicographic / composite efficiency utility rather than subjective LLM prose preference:

$$U(c) = 1000 \cdot \text{VCR}(c) - 25 \cdot N_{\text{trunc}}(c) - 15 \cdot N_{\text{bad\_tool}}(c) - 0.5 \cdot T_{\text{wall}}(c) - 0.01 \cdot \text{Tokens}_{\text{total}}(c)$$

where:
- $\text{VCR}(c) \in [0, 1]$: **Verified Completion Rate** across the probe task set (primary lexicographic key).
- $N_{\text{trunc}}(c)$: count of `done_reason == "length"` truncated generations.
- $N_{\text{bad\_tool}}(c)$: count of malformed tool calls + failed tool executions.
- $T_{\text{wall}}(c)$: wall-clock execution time in seconds (directly proportional to single-GPU `GPUHours` on RTX 4060).
- $\text{Tokens}_{\text{total}}(c) = \text{input\_tokens} + \text{output\_tokens}$ (including thinking overhead and on-demand skill loading cost).

**Promotion Invariant (`KEEP_BASE` fallback)**:
A candidate harness $H_{\text{cand}}$ replaces $H_{\text{base}}$ if and only if:
$$\text{Gate}(H_{\text{cand}}) = \text{ACCEPTED} \quad \land \quad U(H_{\text{cand}}) > U(H_{\text{base}}) + \epsilon$$
Otherwise, the system records `KEEP_BASE` and does not pollute `LongTermMEMORY.md`.

---

## 5. Controlled Experimental Protocol (`H0` vs `H1` vs `H2` vs `H3`)

To evaluate whether recursive harness adaptation improves local Gemma 4 12B R&D execution without confounding variables:

| Arm | Configuration | Harness Adaptation | Selection / Promotion Gate |
|---|---|---|---|
| **H0** | Base `AutonomousWorker` (`gemma4:12b`, winning `G0/G1/G2` inference config) | Static default prompt (`worker.py`) | None |
| **H1** | Manually optimized compact core prompt + on-demand skills (`python_debug`, `rie_report`) | Static human-engineered `c1..c5` | None |
| **H2** | Standard Harness² (`k=1`, `Propose → Probe → Compose` on `c1..c5`) | Automated trajectory-driven edits | Standard LLM trajectory selector (no RIE gate during selection) |
| **H3** | **Verifier-Gated Harness²** (`k=1`, `Propose → Probe → Compose` on `c1..c5`) | Automated trajectory-driven edits | **Trusted RIE hard gate + deterministic `EfficiencyJudge`** |

### Invariants Across `H0–H3`
- **Fixed Local Solver Model**: `gemma4:12b` (`digest: 4eb23ef187e2...`, `Q4_K_M`, `OLLAMA_NUM_CTX=16384`).
- **Fixed Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM).
- **Held-Out Task Split**:
  - *Adaptation / Probe Split*: 4 R&D tasks (clustering abstention fix, ANN candidate reranking, risk-coverage bootstrap CI, narrative claim scrubbing).
  - *Held-Out Evaluation Split*: 4 unseen Orion R&D tasks evaluated with frozen harnesses produced by each arm.
- **Recorded Metrics**:
  1. Held-out Verified Completion Rate (`RIE ACCEPTED %`)
  2. Total tokens per verified task (measuring whether on-demand `SKILL.md` genuinely reduces total tokens vs. always-on prompt)
  3. Wall-clock time & GPU-hours per verified task
  4. Invalid tool calls & output truncations (`done_reason == "length"`)
  5. False-promotion rate (harnesses promoted in `H2` that fail independent RIE vs. `H3`)

---

## 6. Implementation status (2026-10-19, `02d6954`)

Implemented natively in `antigravity_hybrid_orchestrator/harness/` (the upstream `harness2` package is not installed or executed; its design was read as reference only):

- `tree.py` — `HarnessTree` with c1 `system_prompt`, c2 `guardrails`, c3 `memory`, c4 `tool_guidance`, c5 `skills` (served on demand by the worker through a `read_skill` tool). Validation: always-on text ≤ 8192 characters (~2048 tokens), ≤ 6 skills of ≤ 6000 characters, no template braces, every skill named in c1/c2/c4. c6–c8 stay frozen Python. `DEFAULT_TREE` renders the previous worker system prompt byte-for-byte (H0).
- `proposer.py` — cloud harness editor with Harness²'s roles: INV (procedure, tool use, completion), DOM (practitioner knowledge, preferably as triggered skills), AGG (composition from execution contrasts). JSON reply or `keep_base`; one retry with validation errors, then fallback to the base harness.
- `evidence.py` — renders the agent's own view of a rollout (turns, tool calls and errors, truncations, visible test output, final diff). Hidden holdout results are never shown to the proposer; they only feed the H3 gate.
- `recursion.py` — one parallel Propose–Probe–Compose step per task (k=1: INV + DOM candidates probed on the same task, one composition probed); `standard_choice` (H2: adopt the composition) and `gated_choice` (H3: trusted verification gate, then `EfficiencyJudge` utility `U = 1000·VCR − 25·trunc − 15·bad_tool − 0.5·wall − 0.01·tokens`, `KEEP_BASE` unless `U` beats the base by more than ε = 10).
- `scripts/run_harness2_experiment.py` — adaptation on H1–H4 (seed 303) for arms H2 and H3, lessons carried to later tasks (harnessing-level recursion, as short notes), then frozen H0/H2/H3 harnesses evaluated on held-out H5–H8 (seeds 303, 404). Worker policy fixed: `P1_STAGE` stage hints + adaptive generation limit, `think=false`, `num_predict=2048`, one attempt, 300 s.

Deviations from sections 2–5: arm H1 (hand-written harness) is not run; c4 tool guidance is always-on text rather than tool descriptions; the experiment uses the synthetic H-tasks rather than Orion R&D tasks.

**Privacy.** The harness editor sees trajectories and diffs, i.e. code. This is acceptable for the synthetic benchmark tasks used here; running Harness² adaptation on private workspaces sends their code to the cloud and needs an explicit owner decision (it is not wired into normal orchestrator runs).
