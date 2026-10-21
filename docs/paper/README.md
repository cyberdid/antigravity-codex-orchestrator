# Beyond Passing Tests: publication and reproduction package

**Preprint revision: 2026-10-21.** Author: Oleksandr Pylypenko, ORCID
[0009-0004-9610-9242](https://orcid.org/0009-0004-9610-9242).

Independent Researcher in Artificial Intelligence and Machine Learning; PhD Student,
State University of Trade and Economics, Kyiv, Ukraine; in cooperation with the
Department of Software Engineering and Cybersecurity. Status and cooperation were
confirmed by the author. Experiments used the author's equipment and accounts.

## Read the paper

- [English publication PDF](article-en.pdf); source: [tex/en/main.tex](tex/en/main.tex).
- [Ukrainian publication PDF](article-uk.pdf); source: [tex/uk/main.tex](tex/uk/main.tex).
- [Markdown review mirror](manuscript.md) and its compatibility [PDF](manuscript.pdf).
- [Evidence register and source checks](sources.md).
- [Revision response](REVISION_NOTES.md).

The bilingual LaTeX articles are the publication versions. The Markdown mirror
retains evidence IDs and a simpler layout; it does not determine the bilingual
typesetting. Numerical table bodies are generated from saved benchmark JSON files.

## Fixed versions

The completed experimental-data snapshot, before this editorial revision, is
[`9cd894f717a4235c7cb7bdc36563275ba36bdb0f`](https://github.com/cyberdid/antigravity-codex-orchestrator/tree/9cd894f717a4235c7cb7bdc36563275ba36bdb0f).
The publication sources, added sensitivity analysis and revised PDFs are in `main`,
with publication tag `paper-i-preprint-2026-10-21`. For a fixed checkout use:

```bash
git clone https://github.com/cyberdid/antigravity-codex-orchestrator.git
cd antigravity-codex-orchestrator
git checkout paper-i-preprint-2026-10-21
git rev-parse HEAD
sha256sum -c docs/paper/SHA256SUMS
```

`SHA256SUMS` records the input datasets, relevant analysis and publication sources,
figures and PDFs. It excludes itself, this README and compiler intermediates.
The tag's commit SHA identifies the full revision. No Zenodo deposit, DOI,
ResearchGate upload, Google Scholar indexing or formal peer review is claimed here.

## Reproduce the published analysis and PDFs

Requirements: Python 3.12 or later, `matplotlib`, and Tectonic on `PATH` (built locally with Python 3.12.3, matplotlib 3.11.1 and Tectonic 0.15.0). The supplied virtual environment can be used
if already present. For a new environment:

```bash
python3 -m venv .paper-venv
.paper-venv/bin/python -m pip install -r docs/paper/requirements.txt
```

Install Tectonic separately. Its first compilation may download TeX resources;
subsequent builds can use its cache. Fonts are Computer Modern Unicode from the
TeX bundle. From the repository root:

```bash
PY=.paper-venv/bin/python bash scripts/paper/reproduce.sh
```

This single command:

1. Recomputes Campaign A/B and pooled summaries from the saved results.
2. Computes full campaign time, module-rejection frequency and shared-specification
   sensitivity in `benchmarks/publication_analysis.json`.
3. Regenerates all five figures and numerical LaTeX table bodies.
4. Builds English, Ukrainian and Markdown-mirror PDFs.

This is reproduction of the analysis and rendering from recorded data, **not an
independent replication of agent behaviour**. It makes no model calls, starts no
benchmark and requires no GPU. The original oracle-quality measurements are read
from their saved results; their mutation executions are not rerun. PDF creation
timestamps and library versions can change output byte hashes after rebuilding;
the distributed-file checksums apply before regeneration.

To compile only the bilingual articles:

```bash
bash docs/paper/tex/build.sh all
```

To recompute only the new descriptive/sensitivity analysis:

```bash
python3 -B scripts/analysis/publication.py
```

## Experimental reruns are separate

The original protocols are [Campaign A](PREREGISTRATION_CAMPAIGN_A.md) and
[Campaign B](PREREGISTRATION_CAMPAIGN_B.md). Their runner scripts call real local
and cloud models, consume quota and manage background timers; they are **not**
called by `reproduce.sh`. Model identifiers and settings describe the recorded
runs, not a guarantee that those endpoints will remain available.

Final `campaign_a.json`, `campaign_a_ablation.json` and `campaign_b.json` in
`benchmarks/` are authoritative. Files ending in `.partial.json` or
`.partial.backup.json` are retained interruption checkpoints and must not be
concatenated with final results. Campaign A has 225 primary runs plus 50 ablation
runs; Campaign B has 54 repair runs. The ablation's full-interface comparison
reuses 25 Campaign A runs rather than generating another 25 runs.

## Interpretation of the revised results

- Pooled figures (99/150 to 131/150 verified; 37/150 to 4/150 silent false successes)
  compare **contract tests** with baseline, not a mixture of both review forms.
- The pooled analysis is post hoc. The 50 task instances include six Campaign B
  variants of Campaign A specifications. A 44-group sensitivity is reported;
  it does not establish that all groups are independent.
- Campaign A's primary task-level hypothesis was not confirmed (`p=0.125`).
- Identical Campaign A totals for tests and checklist do not establish equivalence.
- Campaign A time overhead, including one generation per task reused across three
  seeds, is 53.65% for tests and 11.76% for checklist. This is not a measured
  one-off deployment, energy or monetary cost.
- No new causal-control or real-repository experiment was added in this revision.

## Publication handoff

The PDFs, sources, checksum manifest and `CITATION.cff` form the versioned preprint
package. A future archive should preserve this tag/commit and record its assigned
DOI in a later metadata update. The current package does not invent an archive
identifier or imply acceptance by a journal or conference.
