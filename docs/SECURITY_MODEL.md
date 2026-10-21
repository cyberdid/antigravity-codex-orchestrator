# Execution and disclosure boundaries

This describes implemented boundaries, not a claim of complete isolation or a
formal security proof. Source: `antigravity_hybrid_orchestrator/runtime/sandbox.py`, `privacy/`, `cli.py` and the
controller integrity/state modules.

## Assets and untrusted inputs

Candidate code, local-model actions, generated tests and candidate workspace files
are untrusted. Host credentials, hidden evaluations, oracle/reference code, trusted
contracts, controller policy and durable evidence must not be writable by a candidate.
The operator and host account remain trusted; a hostile host administrator is outside
this model.

## Live CLI execution

The default is Docker. Mock mode uses deterministic fixtures and a host tool backend.
The candidate container mounts the selected workspace writable. An explicitly
configured ephemeral output directory may also be mounted; neither mount is a place
for controller secrets or hidden answers.

The container command uses a non-root host UID/GID, `--network=none`, `--read-only`,
`--cap-drop=ALL`, `--security-opt=no-new-privileges`, 2 CPUs, 4 GiB RAM with matching
swap cap, 256 PIDs, a per-file output limit (16 MiB by default), and bounded tmpfs.
It does not mount the Docker socket or pass the host environment wholesale. Images
must exist locally (`--pull=never`). A missing image/daemon does not enable host
execution. Cleanup force-removes the owned container.

Docker shares the host kernel. The controls do not prove resistance to kernel
exploits, side channels, resource exhaustion outside the enforced limits or all
container escape techniques. A per-file size cap is not an aggregate disk quota;
writable bind mounts can still consume host disk. The operator must choose a
workspace whose contents the candidate may read and modify.

`--sandbox host` is an explicit execution policy that removes the container boundary.
Benchmark scripts and utilities have their own execution policies; the CLI default
does not apply to them automatically.

## Cloud disclosure

The standard AST view removes function bodies. Specification review can send
names, types, docstrings and dependent-module interfaces. These are useful metadata
and may themselves be sensitive. Credential-pattern/entropy checks are filters,
not formal non-disclosure guarantees. Optional `--cloud-assist spec` also sends
bounded failure information under its documented rules.

Free-form cloud calls use an empty temporary working directory and instructions
not to use tools. A changed cwd is not OS-level confinement of the cloud CLI.
The installed cloud clients, their account settings and service behaviour are
separate trust dependencies.

## Generated tests and scoring

The contract-test AST validator restricts imports/calls and rejects several unsafe
constructs. It is a static filter, not a Python sandbox or a proof that expectations
match the specification. Generated tests can over-constrain valid implementations.
Hidden post-run checks must remain inaccessible to both models. Passing them means
success under that evaluation, not proof of complete correctness.

## Recorded artifacts

Logs can contain local paths, source snippets and model-generated content. Public
research records are preserved as collected; no claim is made that absence of a
credential-pattern match proves absence of all confidential information. Historical
snapshots are immutable. Corrections/redactions, if ever needed, require an explicit
record rather than silent changes to measured outcomes.
