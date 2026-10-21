# Beyond Passing Tests: publication revision 2

**Platform:** CloudEdge AI Orchestrator (formerly Antigravity Codex Orchestrator).
**Author:** Oleksandr Pylypenko, ORCID 0009-0004-9610-9242.

This revision prepares the existing research for public reuse and citation. It
renames the public platform, adds Apache-2.0 software licensing and CC BY 4.0
article/data licensing, aligns package/lock/CLI metadata, documents Docker setup
and boundaries, and adds GitHub Actions for existing tests and saved-data/PDF
reproduction. The historical Python namespace and command aliases remain compatible.

## Scientific scope

No new model experiment was run. Raw observations and numerical outcomes are
unchanged: contract tests versus baseline give 99/150 → 131/150 measured verified
completions and 37/150 → 4/150 silent false successes in the pooled post-hoc analysis.
Campaign A's primary task-level test was not significant (`p=0.125`); equal totals
for tests and checklist do not establish equivalence. Shared specifications and
the 44-group sensitivity remain explicit. These are designed repair tasks and
model-generated bug variants, not a validation on independent real repositories.

## Fixed versions

- Original tag `paper-i-preprint-2026-10-21` remains on
  `8e9ba05ebbbcaf7953a8f31287d935f01fa2d6be`.
- Licensed/renamed revision: `paper-i-preprint-2026-10-21-rev2`.
- Frozen experimental-data snapshot: `9cd894f717a4235c7cb7bdc36563275ba36bdb0f`.

## Included assets

- English and Ukrainian article PDFs.
- SHA-256 manifest for distributed paper inputs/outputs and supporting metadata.
- Source/artifact archive of this revision, including final datasets, checkpoint
  provenance, scripts, figures, licenses and reproduction instructions.
- CFF and article-deposit metadata; neither asserts an assigned DOI.

From a checkout, verify before rebuilding:

```bash
sha256sum -c docs/paper/SHA256SUMS
uv sync --locked --extra paper
uv run --locked --extra paper bash scripts/paper/reproduce.sh
```

Install Tectonic 0.15.0 separately (Linux x86_64 helper:
`bash scripts/paper/install_tectonic.sh`). Rebuilding PDFs may change byte hashes
because of creation timestamps; numerical summaries and tables are compared separately.

Local verification: **995 passed, 12 skipped** in the existing offline suite and
**24 passed** in real Docker acceptance. GitHub-hosted results are visible in
[Actions](https://github.com/cyberdid/cloud-edge-ai-orchestrator/actions/workflows/ci.yml).
No independent GPU replication, formal peer review or Zenodo DOI deposit is claimed.
