# Research integrity and Orion rerun

Approved scope: the user's “давай реалізуємо це” accepts the five-part approach in the conversation. The audit in `docs/ORION1_Codex_Orchestrator_Audit_Research_Integrity_UA.md` supplies the detailed requirements. This document makes implementation boundaries explicit.

## Outcome

Preserve ORION-1 as historical evidence. Deliver an independently checked Orion experiment and an orchestrator which distinguishes artifact creation from behavioral/scientific verification. A valid negative scientific result is an accepted outcome. Do not claim novelty or real-world speaker validity from synthetic embeddings.

## Components

1. Reliability: resume records bind completed tasks to hashes of outputs and inputs; changed/missing files invalidate the task and its downstream consumers. Legacy snapshots require revalidation. Stable snapshots preserve completed verification evidence. Model and command deadlines are real wall-clock limits; termination cleans process trees. Context history is bounded while preserving the task and complete recent tool exchanges. Gauntlet reports model failure and never silently substitutes a reference patch.
2. Sandbox: live workspace shell runs in Docker with network disabled, no host credentials, a non-root UID, dropped capabilities, read-only root, CPU/RAM/PID limits and only a candidate workspace mount. Host execution is an explicit opt-out, visibly recorded. Trusted evaluator uses a separate read-only candidate snapshot and ephemeral output; it never mounts evaluator code, expected answers or evidence store into the candidate. Containers are removed on timeout/cancel. No Docker socket inside containers. Missing sandbox fails closed.
3. RIE: controller-owned versioned contracts, deterministic reference metric fixtures and randomized cases, behavior-based retrieval checks against exact neighbors, externally observed timing/resources, candidate and artifact hashes, evidence IDs, claim-to-evidence validation. Contract selection is operator/controller-owned, never trusted from planner output. Outcomes: ACCEPTED, REJECTED, INCONCLUSIVE, SKIPPED_RESOURCE_LIMIT. Shell exit zero alone never establishes scientific verification. Evidence is stored outside candidate workspace with owner-only access; hashes provide identity, not an unsupported claim of authenticity.
4. Orion: deterministic local RNG, explicit outlier labels/masks, controlled domain shifts, documented difficulty semantics; correct pairwise/B³/ARI/V-measure/coverage/FMR/FDP definitions and counts with null for undefined ratios; real unknown-K baselines, canonical clustering labels, mutual neighborhoods, abstention=-1; fair development and disjoint held-out seeds, frozen chosen parameters, paired ablations/sensitivity, saved predictions; real batched exact and approximate retrieval, trusted recall and resource measurements; parseable installable package; tests import actual source. Dense N×N operations must not be used for large-N scalability.
5. Rerun: regression suite, known-bad ORION-1 rejection, valid candidate acceptance, short live local-model run, full reproducible Orion experiment, independent evidence and a report tying claims to measurements. Record model IDs/digests, code revision, config, environment, raw events, interventions, exclusions and resource limits. Preserve old published artifacts; new artifacts go under a new run directory.

## Integration

`WorkspaceToolSuite` receives `sandbox_backend='host'|'docker'` and `sandbox_image`; existing library/mock calls keep host compatibility, CLI live R&D defaults to Docker. `SelfCorrectionLoop` receives an optional trusted gate, run after shell verification and before completion/checkpoint. A gate rejection supplies structured feedback. The coordinator owns the selected contract and evidence directory. New state records carry verification level and fingerprints. Ordinary coding without a scientific contract is labeled behavioral/structural, never scientific.

RIE tests use independent literal expectations and separate reference code. Candidate execution exchanges data through JSON/NumPy files with strict shape/range checks; hidden labels and reference outputs stay in the controller. Docker is an engineering security boundary with documented residual kernel/shared-hardware risks, not a formal proof.

## Acceptance

- Current fake scalability and placeholder metric implementations are rejected by executable checks.
- Correct metrics pass perfect/singleton/all-in-one/mixed/empty/all-abstained/partial-abstention and label/row permutation cases.
- Candidate cannot read a host canary or hidden controller files or reach the network; workspace writes work; timeouts kill descendants.
- Resume after source/output mutation invalidates dependent accepted results; unchanged verified tasks remain skippable.
- No reported empirical claim lacks matching trusted evidence; stale hashes and unsupported claims fail.
- Tests and clean package/container runs pass; fresh experiment includes measured neighbors, predictions and reproducibility information.
- No automatic merge/push, no modification of unrelated containers/services, no old artifact overwrite.

## Implementation rulings

- “Orin” is interpreted as Project Orion on this machine pending any correction; no remote Jetson changes are authorized by that ambiguity.
- Implement in `/tmp/antigravity-orion2`, branch `codex/research-integrity`, preserving the user's clean main checkout.
- Execute approved scope continuously; do not request repeated design approval.
