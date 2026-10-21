# Experimental artifact index

See [Paper I reproduction instructions](../docs/paper/README.md) and the
[evidence map](../docs/paper/sources.md). Do not combine checkpoints with final data.

| Final record | Meaning |
|---|---|
| `campaign_a.json` | Canonical Campaign A: 225 primary runs, 3 arms, 25 tasks, 3 seeds |
| `campaign_a_ablation.json` | 50 extra ablation runs; full-interface comparison reuses A |
| `campaign_b.json` | Canonical Campaign B: 54 repair runs, 6 model-generated bug variants |
| `campaign_a_analysis.txt`, `campaign_b_analysis.txt` | Summaries recomputed from the corresponding final JSON |
| `pooled_analysis.txt` | Post-hoc pooled comparisons of contract tests versus baseline |
| `publication_analysis.json` | Post-hoc cost, generation reliability and 44-group sensitivity |
| `*_test_grading.json`, `oracle_quality.json` | Saved test-quality/reference/mutation measurements |
| `spec_contract_*.json` without `partial` | Earlier specification-review experiments; interpret under their recorded protocol |
| `harness2_experiment.json`, `harness2_experiment/` | Harness adaptation experiment and its proposal/cache/progress evidence |
| `gemma_orion_phase*.json` | Earlier worker/control/ladder measurements, not all included in the pooled analysis |
| `orion3/` | Partial corpus and labels for the paused Orion-3 study; not completed experiment results |

## Checkpoints and diagnostics

- `*.partial.json` and `*.partial.backup.json` retain interruption/resume state.
- `*.log` record progress and diagnostics; they are not additional independent runs.
- The final A/B files supersede their checkpoints for analysis. Repeated records
  across these files are expected and must not inflate denominators.
- Paths and historical command/model names in raw records are provenance, not
  instructions to rename or re-execute them.

All raw observations are retained. The original Paper I tag stays on commit
`8e9ba05ebbbcaf7953a8f31287d935f01fa2d6be`; the licensed/renamed package uses
`paper-i-preprint-2026-10-21-rev2`. SHA-256 checksums for the article's inputs and
outputs are in `../docs/paper/SHA256SUMS`.

Experimental records/summaries are CC BY 4.0; executable source is Apache-2.0.
See [licensing scope](../docs/LICENSING.md). No independent GPU replication or
archival DOI is implied by distributing these records.
