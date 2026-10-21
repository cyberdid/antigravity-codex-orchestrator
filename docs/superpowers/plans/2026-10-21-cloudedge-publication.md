# CloudEdge publication readiness implementation plan

**Goal:** Rename the public platform to CloudEdge AI Orchestrator and publish a
licensed, installable and reproducible Paper I revision on `main`.

**Architecture:** Change public metadata, CLI branding and repository documentation;
retain `antigravity_hybrid_orchestrator` and the existing command aliases. Preserve
raw observations and the original publication tag. Add independent CI jobs for
existing offline tests and saved-data/PDF reproduction.

**Tech stack:** Python 3.12, uv, Docker, Tectonic, GitHub Actions and GitHub CLI.
**Specification:** Owner-provided GitHub publication audit and confirmation on 2026-10-21.

## Invariants

- Original tag `paper-i-preprint-2026-10-21` stays on `8e9ba05ebbbcaf7953a8f31287d935f01fa2d6be`.
- Repository becomes `cyberdid/cloud-edge-ai-orchestrator`; Python imports stay unchanged.
- Public command: `cloud-edge-ai`; previous Codex commands remain aliases.
- Owner confirmed licensing rights: Apache-2.0 code, CC BY 4.0 article and records.
- New publication tag: `paper-i-preprint-2026-10-21-rev2`; do not move old tags.
- No new model runs. Zenodo metadata/archive preparation is separate from a DOI deposit.
- Install verification uses `/tmp/cloudedge-install-20261021`, not the existing `.venv`.

## Tasks

- [ ] Add official license texts and scoped notices in `LICENSE`, `NOTICE`,
      `docs/paper/LICENSE`, `benchmarks/LICENSE`, and `docs/LICENSING.md`.
- [ ] Update `pyproject.toml` package name, URLs, license and console entry point;
      add optional `research` and `paper` dependencies matching the recorded runtime.
      Regenerate with `uv lock`; verify using `uv lock --check` and an isolated sync.
- [ ] Update CLI help branding and package display metadata; retain all imports and aliases.
- [ ] Replace root README with installation, Docker preparation, backend requirements,
      privacy boundaries, runtime limits, scientific artifacts, canonical/checkpoint
      distinctions, license scope and commands without fixed test counts.
- [ ] Add `docs/SECURITY_MODEL.md`, artifact index and release notes; retain historical
      logs unchanged. Scan tracked text logs for credential-shaped content without
      printing potential secrets.
- [ ] Add `.github/workflows/ci.yml`: existing test suite, optional Docker acceptance
      job, and a saved-data reproduction/PDF job with pinned action revisions and
      no local/cloud model calls.
- [ ] Update active paper/CFF links to the new repository and revision-2 tag; keep
      historical identifiers and original snapshot link explicit. Rebuild PDFs,
      visually review changed pages and regenerate `docs/paper/SHA256SUMS`.
- [ ] Verify numerical summaries and raw experiment hashes are unchanged; retain
      original tag object and target. Commit, fast-forward `main`, rename GitHub,
      update `origin`, push new commit/tag and create the versioned GitHub Release.
- [ ] Verify remote main/tag/release assets and report exact status, including CI
      results and the absence of an assigned Zenodo DOI.
