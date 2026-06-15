#!/usr/bin/env bash
# Remote control for v26 one-click pipeline.
#
# Usage:
#   bash scripts/v26_remote.sh sync
#   bash scripts/v26_remote.sh smoke
#   bash scripts/v26_remote.sh pipeline [--updates N] [--skip-post]
#   bash scripts/v26_remote.sh status
#   bash scripts/v26_remote.sh tail
#   bash scripts/v26_remote.sh eval
#   bash scripts/v26_remote.sh wait
#   bash scripts/v26_remote.sh stop
set -euo pipefail

REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
RDIR="${RDIR:-/home/charlie/project/OrbitWarRL}"
RPY="/home/charlie/anaconda3/bin/python"

cmd="${1:-status}"
shift || true

_ssh() { ssh -o ConnectTimeout=15 "$REMOTE" "$@"; }

V26_SCRIPTS=(
  scripts/v26_extend.sh
  scripts/v26_remote.sh
  scripts/v26_pipeline.sh
  scripts/v26_smoke.sh
  scripts/v26_one_click.sh
  scripts/v26_post_train.sh
  scripts/test_v26_preflight.sh
  scripts/bc_collect_parallel.sh
)

case "$cmd" in
  status)
    _ssh "cd '$RDIR' && echo '=== gpu ===' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo '(no gpu)' && echo '=== procs ===' && ps aux | grep -E 'v26_pipeline|v26_extend|bc_collect|train_bc' | grep -v grep || echo '(none)' && echo '=== pipeline status ===' && cat logs/v26_pipeline/status.json 2>/dev/null || echo '(no status.json)'"
    ;;
  sync)
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
      --exclude='logs/' --exclude='ckpt_*/' --exclude='data/' --exclude='.DS_Store' \
      --exclude='_archive/' \
      "$ROOT/" "$REMOTE:$RDIR/"
    chmod_cmd=""
    for s in "${V26_SCRIPTS[@]}"; do
      chmod_cmd+=" chmod +x '$RDIR/$s';"
    done
    _ssh "$chmod_cmd true"
    echo "synced"
    ;;
  smoke)
    _ssh "cd '$RDIR' && env PYTHON='$RPY' CHECK_REMOTE=0 bash scripts/v26_smoke.sh"
    ;;
  pipeline)
    EXTRA=()
    while [ $# -gt 0 ]; do EXTRA+=("$1"); shift; done
    _ssh "cd '$RDIR' && mkdir -p logs/v26_pipeline && (nohup env PYTHON='$RPY' bash scripts/v26_pipeline.sh ${EXTRA[*]} > logs/v26_pipeline/nohup.log 2>&1 & echo \$! > logs/v26_pipeline/pipeline.pid) && sleep 2 && tail -5 logs/v26_pipeline/nohup.log"
    ;;
  wait)
    echo "Waiting for v26 pipeline to finish..."
    while _ssh "test -f '$RDIR/logs/v26_pipeline/pipeline.pid' && kill -0 \$(cat '$RDIR/logs/v26_pipeline/pipeline.pid') 2>/dev/null"; do
      _ssh "tail -3 '$RDIR/logs/v26_pipeline/pipeline.log' 2>/dev/null || true"
      sleep 120
    done
    _ssh "test -f '$RDIR/logs/v26_pipeline/status.done' && echo DONE || echo FAILED"
    ;;
  stop)
    _ssh "pkill -f 'v26_pipeline.sh' 2>/dev/null || true; pkill -f 'multi_action_v26_roi' 2>/dev/null || true; pkill -f 'v26_extend.sh' 2>/dev/null || true; sleep 2; echo stopped"
    ;;
  tail)
    _ssh "tail -15 \$(ls -t '$RDIR'/logs/v26_extend_*/train.log 2>/dev/null | head -1) 2>/dev/null || tail -15 '$RDIR/logs/v26_pipeline/pipeline.log'"
    ;;
  eval)
    _ssh "grep '^\[eval_vs_v20\]' \$(ls -t '$RDIR'/logs/v26_extend_*/train.log 2>/dev/null | head -1) | grep -v WARN | tail -20"
    ;;
  log)
    _ssh "tail -30 '$RDIR/logs/v26_pipeline/pipeline.log'"
    ;;
  *)
    echo "usage: $0 {status|sync|smoke|pipeline|wait|stop|tail|eval|log}" >&2
    exit 2
    ;;
esac
