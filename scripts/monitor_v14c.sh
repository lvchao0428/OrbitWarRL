#!/usr/bin/env bash
# monitor_v14c.sh — poll remote v14c training + eval_vs_v20 trend
# Usage: bash scripts/monitor_v14c.sh
#        bash scripts/monitor_v14c.sh --watch   # loop every 10m

set -euo pipefail
REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
ROOT="${REMOTE_ROOT:-~/project/OrbitWarRL}"
INTERVAL="${MONITOR_INTERVAL:-600}"

fetch() {
  ssh "$REMOTE" "cd $ROOT && {
    echo '=== training ==='
    tail -1 logs/v14c.log 2>/dev/null || echo '(no log)'
    echo '=== eval_vs_v20 ==='
    grep 'eval_vs_v20' logs/v14c.log 2>/dev/null | tail -8 || echo '(none yet)'
    echo '=== v13c baseline ==='
    head -1 logs/replay_analyze/v13c_final_vs_v20.summary.txt 2>/dev/null || echo '(no v13c summary)'
    echo '=== ckpt ==='
    ls -lt ckpt_multi_action_v14c/ckpt_*.pkl 2>/dev/null | head -2 || echo '(no ckpt yet)'
  }"
}

fetch

if [[ "${1:-}" == "--watch" ]]; then
  echo ""
  echo "[monitor] watching every ${INTERVAL}s (Ctrl+C to stop)"
  while true; do
    sleep "$INTERVAL"
    echo ""
    echo "--- $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
    fetch
  done
fi
