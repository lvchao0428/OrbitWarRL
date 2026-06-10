#!/usr/bin/env bash
# monitor_v14d.sh — poll v14d curriculum state + phase logs
set -euo pipefail
REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
ROOT="${REMOTE_ROOT:-~/project/OrbitWarRL}"

fetch() {
  ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 "$REMOTE" "cd $ROOT && {
    echo '=== search state ==='
    cat logs/v14d_search.state.json 2>/dev/null || echo '(no search state)'
    echo '=== search log (tail) ==='
    tail -3 logs/v14d_search.log 2>/dev/null || echo '(no search log)'
    echo '=== curriculum state ==='
    cat logs/v14d_curriculum.state.json 2>/dev/null || echo '(no curriculum state)'
    echo '=== process ==='
    pgrep -af 'v14d_curriculum_search|v14d_curriculum|multi_action_v14d' || echo '(no train)'
  }"
}

fetch

if [[ "${1:-}" == "--watch" ]]; then
  INTERVAL="${MONITOR_INTERVAL:-900}"
  echo "[monitor] every ${INTERVAL}s"
  while true; do
    sleep "$INTERVAL"
    echo "--- $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
    fetch
  done
fi
