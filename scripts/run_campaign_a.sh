#!/usr/bin/env bash
# Paper I Campaign A (docs/paper/PREREGISTRATION_CAMPAIGN_A.md): 225 primary runs, then 50 ablation runs.
# Re-running the script resumes each phase from its saved partial results.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
LOG=benchmarks/campaign_a.log
TASKS=$(seq -s, -f "P%g" 1 25)
COMMON=(--tasks "$TASKS" --attempts 2 --cloud-backend agy --cloud-model gemini-4-argon-medium)

run_phase() {  # output, policies, seeds
  local out=$1 policies=$2 seeds=$3 resume=()
  [ -f "$out" ] && { echo "[$(date +%T)] $out already complete" >> "$LOG"; return; }
  local partial="${out%.json}.partial.json"
  [ -f "$partial" ] && resume=(--resume-from "$partial")
  echo "[$(date +%T)] start $out ($policies, seeds $seeds) ${resume[*]}" >> "$LOG"
  .venv/bin/python3 scripts/benchmark_gemma_orion_phase2b.py --policies "$policies" --seeds "$seeds" \
    "${COMMON[@]}" "${resume[@]}" --output "$out" >> "$LOG" 2>&1
  echo "[$(date +%T)] $out exited rc=$?" >> "$LOG"
}

systemctl --user stop archive-index-sessions.timer archive-index.timer 2>/dev/null && echo "[$(date +%T)] paused archive-index timers" >> "$LOG"
echo "[$(date +%T)] campaign A at $(git log --oneline -1)" >> "$LOG"
run_phase benchmarks/campaign_a.json P1_STAGE_ADAPTIVE,P1_STAGE_ADAPTIVE_SPECTESTS,P1_STAGE_ADAPTIVE_CHECKLIST 303,404,505
run_phase benchmarks/campaign_a_ablation.json P1_STAGE_ADAPTIVE_SPECTESTS_SIG,P1_STAGE_ADAPTIVE_SPECTESTS_DOC 303
systemctl --user start archive-index-sessions.timer archive-index.timer 2>/dev/null && echo "[$(date +%T)] restored archive-index timers" >> "$LOG"
echo "[$(date +%T)] DONE" >> "$LOG"
