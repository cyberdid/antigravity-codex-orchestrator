#!/usr/bin/env bash
# Resume the secondary confirmatory run (baseline vs SPECTESTS on H1-H13) after an interruption.
# Completed runs and the generated contract tests are taken from the saved partial file.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
LOG=benchmarks/spec_contract_confirm_all.resume.log
systemctl --user stop archive-index-sessions.timer archive-index.timer 2>/dev/null && echo "[$(date +%T)] paused archive-index timers" >> "$LOG"
echo "[$(date +%T)] resume at $(git log --oneline -1)" >> "$LOG"
.venv/bin/python3 scripts/benchmark_gemma_orion_phase2b.py \
  --policies P1_STAGE_ADAPTIVE,P1_STAGE_ADAPTIVE_SPECTESTS \
  --tasks H1,H2,H3,H4,H5,H6,H7,H8,H9,H10,H11,H12,H13 \
  --seeds 303,404,505 --attempts 2 \
  --resume-from benchmarks/spec_contract_confirm_all.partial.backup.json \
  --output benchmarks/spec_contract_confirm_all.json >> "$LOG" 2>&1
echo "[$(date +%T)] exited rc=$?" >> "$LOG"
systemctl --user start archive-index-sessions.timer archive-index.timer 2>/dev/null && echo "[$(date +%T)] restored archive-index timers" >> "$LOG"
echo "[$(date +%T)] DONE" >> "$LOG"
