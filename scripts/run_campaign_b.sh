#!/usr/bin/env bash
# Paper I Campaign B (docs/paper/PREREGISTRATION_CAMPAIGN_B.md).
#   bash scripts/run_campaign_b.sh generate   -> writes scripts/paper_tasks/natural_tasks.json (commit it before running)
#   bash scripts/run_campaign_b.sh run        -> 3 arms x natural tasks x 3 seeds (resumable)
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
LOG=benchmarks/campaign_b.log
FROZEN=scripts/paper_tasks/natural_tasks.json

systemctl --user stop archive-index-sessions.timer archive-index.timer 2>/dev/null && echo "[$(date +%T)] paused archive-index timers" >> "$LOG"
case "${1:-}" in
  generate)
    [ -f "$FROZEN" ] && { echo "[$(date +%T)] $FROZEN exists; refusing to regenerate" >> "$LOG"; }
    [ -f "$FROZEN" ] || .venv/bin/python3 scripts/paper_tasks/natural.py >> "$LOG" 2>&1
    echo "[$(date +%T)] generate exited rc=$?" >> "$LOG"
    ;;
  run)
    if ! git ls-files --error-unmatch "$FROZEN" >/dev/null 2>&1 || ! git diff --quiet -- "$FROZEN"; then
      echo "[$(date +%T)] $FROZEN is not committed unchanged; freeze it first" >> "$LOG"
    else
      OUT=benchmarks/campaign_b.json
      TASKS=$(.venv/bin/python3 -c "import json; print(','.join(r['task_id'].split('_')[0] for r in json.load(open('$FROZEN'))['records'] if 'task_id' in r))")
      PARTIAL="${OUT%.json}.partial.json"; RESUME=()
      [ -f "$PARTIAL" ] && cp "$PARTIAL" "${PARTIAL%.json}.backup.json" && RESUME=(--resume-from "${PARTIAL%.json}.backup.json")
      echo "[$(date +%T)] campaign B at $(git log --oneline -1); tasks $TASKS ${RESUME[*]}" >> "$LOG"
      [ -f "$OUT" ] || .venv/bin/python3 scripts/benchmark_gemma_orion_phase2b.py \
        --policies P1_STAGE_ADAPTIVE,P1_STAGE_ADAPTIVE_SPECTESTS,P1_STAGE_ADAPTIVE_CHECKLIST --seeds 303,404,505 \
        --tasks "$TASKS" --attempts 2 --cloud-backend agy --cloud-model gemini-4-argon-medium \
        "${RESUME[@]}" --output "$OUT" >> "$LOG" 2>&1
      echo "[$(date +%T)] run exited rc=$?" >> "$LOG"
    fi
    ;;
  *) echo "usage: $0 generate|run" >&2 ;;
esac
systemctl --user start archive-index-sessions.timer archive-index.timer 2>/dev/null && echo "[$(date +%T)] restored archive-index timers" >> "$LOG"
echo "[$(date +%T)] DONE ${1:-}" >> "$LOG"
