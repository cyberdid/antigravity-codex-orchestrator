#!/usr/bin/env bash
# Recompute descriptive/statistical summaries and render the paper from SAVED data.
# No Ollama, cloud calls, worker runs, mutation execution or GPU use.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PY=${PY:-python3}
export PYTHONDONTWRITEBYTECODE=1
"$PY" scripts/analysis/campaign_a.py benchmarks/campaign_a.json benchmarks/campaign_a_ablation.json > benchmarks/campaign_a_analysis.txt
"$PY" scripts/analysis/campaign_a.py benchmarks/campaign_b.json > benchmarks/campaign_b_analysis.txt
"$PY" scripts/analysis/pooled.py > benchmarks/pooled_analysis.txt
"$PY" scripts/analysis/publication.py > benchmarks/publication_analysis.json
"$PY" scripts/analysis/paper_figures.py
"$PY" scripts/analysis/paper_diagram.py
bash docs/paper/tex/build.sh all
"$PY" scripts/paper/build_pdf.py
echo "Recomputed summaries, figures and all three PDFs from saved results."
