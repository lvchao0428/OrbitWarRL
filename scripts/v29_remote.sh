#!/usr/bin/env bash
# Remote control for v29 pipeline (no BC).
#
# Usage:
#   bash scripts/v29_remote.sh sync
#   bash scripts/v29_remote.sh smoke
#   bash scripts/v29_remote.sh pipeline [--updates N] [--skip-post]
#   bash scripts/v29_remote.sh status
#   bash scripts/v29_remote.sh tail
#   bash scripts/v29_remote.sh eval
#   bash scripts/v29_remote.sh stop
set -euo pipefail

REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
RDIR="${RDIR:-/home/charlie/project/OrbitWarRL}"
RPY="/home/charlie/anaconda3/bin/python"

cmd="${1:-status}"
shift || true

_ssh() { ssh -o ConnectTimeout=15 "$REMOTE" "$@"; }

V29_SCRIPTS=(
  scripts/v29_extend.sh
  scripts/v29_remote.sh
  scripts/v29_pipeline.sh
  scripts/v29_smoke.sh
  scripts/v29_post_train.sh
  scripts/v29_start_remote.sh
)

case "$cmd" in
  status)
    _ssh "cd '$RDIR' && echo '=== gpu ===' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo '(no gpu)' && echo '=== procs ===' && ps aux | grep -E 'v29_pipeline|v29_extend|multi_action_v29' | grep -v grep || echo '(none)' && echo '=== pipeline status ===' && cat logs/v29_pipeline/status.json 2>/dev/null || echo '(no status.json)'"
    ;;
  sync)
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
      --exclude='logs/' --exclude='ckpt_*/' --exclude='data/' --exclude='.DS_Store' \
      "$ROOT/" "$REMOTE:$RDIR/"
    chmod_cmd=""
    for s in "${V29_SCRIPTS[@]}"; do
      chmod_cmd+=" chmod +x '$RDIR/$s';"
    done
    _ssh "$chmod_cmd true"
    echo "synced"
    ;;
  smoke)
    _ssh "cd '$RDIR' && env PYTHON='$RPY' CHECK_REMOTE=0 bash scripts/v29_smoke.sh"
    ;;
  pipeline)
    EXTRA=()
    while [ $# -gt 0 ]; do EXTRA+=("$1"); shift; done
    _ssh "cd '$RDIR' && env PYTHON='$RPY' bash scripts/v29_start_remote.sh ${EXTRA[*]:-}"
    ;;
  wait)
    echo "Waiting for v29 pipeline to finish..."
    while _ssh "test -f '$RDIR/logs/v29_pipeline/pipeline.pid' && kill -0 \$(cat '$RDIR/logs/v29_pipeline/pipeline.pid') 2>/dev/null"; do
      _ssh "tail -3 '$RDIR/logs/v29_pipeline/pipeline.log' 2>/dev/null || true"
      sleep 120
    done
    _ssh "test -f '$RDIR/logs/v29_pipeline/status.done' && echo DONE || echo FAILED"
    ;;
  stop)
    _ssh "pkill -f 'v29_pipeline.sh' 2>/dev/null || true; pkill -f 'multi_action_v29_aim' 2>/dev/null || true; pkill -f 'v29_extend.sh' 2>/dev/null || true; sleep 2; echo stopped"
    ;;
  tail)
    _ssh "tail -15 \$(ls -t '$RDIR'/logs/v29_extend_*/train.log 2>/dev/null | head -1) 2>/dev/null || tail -15 '$RDIR/logs/v29_pipeline/pipeline.log'"
    ;;
  eval)
    _ssh "grep '^\[eval_vs_v20\]' \$(ls -t '$RDIR'/logs/v29_extend_*/train.log 2>/dev/null | head -1) | grep -v WARN | tail -20"
    ;;
  log)
    _ssh "tail -30 '$RDIR/logs/v29_pipeline/pipeline.log'"
    ;;
  *)
    echo "usage: $0 {status|sync|smoke|pipeline|wait|stop|tail|eval|log}" >&2
    exit 2
    ;;
esac
