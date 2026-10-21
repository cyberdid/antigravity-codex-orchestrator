#!/usr/bin/env bash
# Build the LaTeX articles: regenerate table bodies from the benchmark files, then compile the
# English and Ukrainian PDFs with Tectonic (XeTeX). Output: docs/paper/article-en.pdf,
# docs/paper/article-uk.pdf. Usage: bash docs/paper/tex/build.sh [en|uk|all]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PAPER=$(cd "$HERE/.." && pwd)
TECTONIC=${TECTONIC:-$(command -v tectonic || echo "$HOME/.local/bin/tectonic")}
PY=${PY:-python3}
which=${1:-all}
PYTHONDONTWRITEBYTECODE=1 "$PY" "$HERE/gen_tables.py"
for lang in en uk; do
  [[ $which == all || $which == "$lang" ]] || continue
  [[ -f $HERE/$lang/main.tex ]] || { echo "skip $lang: no main.tex"; continue; }
  (cd "$HERE/$lang" && "$TECTONIC" --keep-logs --outdir "$HERE/$lang" main.tex)
  cp "$HERE/$lang/main.pdf" "$PAPER/article-$lang.pdf"
  echo "built $PAPER/article-$lang.pdf"
done
