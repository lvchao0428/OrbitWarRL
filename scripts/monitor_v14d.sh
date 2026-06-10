#!/usr/bin/env bash
# monitor_v14d.sh — poll v14d curriculum state + phase logs
set -euo pipefail
REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
ROOT="${REMOTE_ROOT:-~/project/OrbitWarRL}"

fetch() {
  ssh "$REMOTE" "cd $ROOT && {
    echo '=== state ==='
    cat logs/v14d_curriculum.state.json 2>/dev/null || echo '(no state)'
    echo '=== curriculum log (tail) ==='
    tail -3 logs/v14d_curriculum.log 2>/dev/null || echo '(no curriculum log)'
    for p in a b c; do
      echo \"=== phase \$p ===\"
      tail -1 logs/v14d_phase_\${p}.log 2>/dev/null || echo '(not started)'
      grep eval_vs_v20 logs/v14d_phase_\${p}.log 2>/dev/null | tail -1 || true
    done
    echo '=== process ==='
    pgrep -af 'v14d_curriculum|multi_action_v14d' || echo '(no train)'
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
