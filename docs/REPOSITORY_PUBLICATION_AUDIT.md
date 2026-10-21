# Repository publication readiness: revision 2, 2026-10-21

This records the bounded changes following the owner's GitHub audit. It is not
a comprehensive security audit, an independent model replication or legal review.
The owner confirmed licensing rights and departmental cooperation/PhD student status.

## Identity and preserved evidence

- Public project: **CloudEdge AI Orchestrator**.
- Repository: `cyberdid/cloud-edge-ai-orchestrator`.
- Distribution: `cloud-edge-ai-orchestrator`, version `0.1.0`.
- New CLI: `cloud-edge-ai`; both prior Codex aliases remain.
- Python namespace and historical Docker image name remain unchanged.
- Original publication tag object: `52dd891ee9a5cd755653c6eb025ba84119827b0f`.
- Original publication commit: `8e9ba05ebbbcaf7953a8f31287d935f01fa2d6be`.
- Revision-2 tag: `paper-i-preprint-2026-10-21-rev2`.
- Raw results, preregistrations and historical logs are not rewritten.

GitHub redirects old repository URLs after a rename. Keep the old name unused so
that those redirects remain available; see the [official rename documentation](https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository).
The existing local checkout directory may retain its old filesystem name.

## Changes

| Finding | Revision |
|---|---|
| No license | Apache-2.0 root license/NOTICE; CC BY 4.0 paper/data scope and legal text; author rights confirmation recorded |
| Stale package name in lock | Regenerated `uv.lock`; locked scientific/rendering extras; checked new and legacy commands |
| Missing Docker setup | Build command and live prerequisites added to Quick Start |
| Formal-sounding privacy claims | README now describes bounded disclosure and links the execution/threat model |
| Conflicting test counts | Removed fixed counts from the main README; pytest/CI report actual results |
| Mixed final/checkpoint files | Added `benchmarks/README.md`; preserve all observations and checkpoints |
| No CI | Three jobs: offline tests/install, real Docker acceptance, saved-data/PDF reproduction |
| Moving citation links | Active links use the renamed repository plus the new tag; original snapshot is explicit |
| Release/archive metadata | Versioned release notes, mixed-license scope, CFF and separate software/preprint Zenodo metadata |

## Local verification

- `uv lock --check`: succeeds after metadata alignment.
- Isolated installation: CPython 3.13.14 in `/tmp/cloudedge-install-20261021`; the
  research workspace's original `.venv` was not synchronised or pruned.
- New CLI and both compatibility aliases produce identical help.
- Existing offline tests: **995 passed, 12 skipped**, 121.88 s. Docker acceptance
  cases skipped here were run separately: **24 passed**, 7.81 s.
- Built wheel: includes package runtime requirement files and Apache license/NOTICE.
- Saved-data reproduction: numerical summaries and raw observations unchanged;
  English/Ukrainian and Markdown-mirror PDFs rebuilt without model calls.
- CFF validated against the official CFF 1.2 schema; workflow YAML parsed.
- Credential-pattern scan: 117 tracked text artifacts; no private-key/provider-token
  pattern matches. Nine files contain historical home-directory paths. These are
  retained provenance. This scan does not prove absence of every confidential value.

GitHub-hosted CI results are recorded on the repository's Actions/release pages;
the configuration alone is not evidence that a remote run passed.

## Archive status

`.zenodo.json` describes a software archive; `docs/paper/zenodo-preprint.json`
describes an article deposit. A GitHub Release is distinct from a Zenodo deposit.
No DOI is invented or claimed. An eventual DOI should be added in a later metadata
revision, preserving both existing scientific tags.
