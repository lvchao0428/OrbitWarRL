#!/usr/bin/env bash
# Remote control for v27 one-click pipeline.
#
# Usage:
#   bash scripts/v27_remote.sh sync
#   bash scripts/v27_remote.sh smoke
#   bash scripts/v27_remote.sh pipeline [--updates N] [--skip-post]
#   bash scripts/v27_remote.sh status
#   bash scripts/v27_remote.sh tail
#   bash scripts/v27_remote.sh eval
#   bash scripts/v27_remote.sh wait
#   bash scripts/v27_remote.sh stop
set -euo pipefail

REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
RDIR="${RDIR:-/home/charlie/project/OrbitWarRL}"
RPY="/home/charlie/anaconda3/bin/python"

cmd="${1:-status}"
shift || true

_ssh() { ssh -o ConnectTimeout=15 "$REMOTE" "$@"; }

V27_SCRIPTS=(
  scripts/v27_extend.sh
  scripts/v27_remote.sh
  scripts/v27_pipeline.sh
  scripts/v27_smoke.sh
  scripts/v27_one_click.sh
  scripts/v27_post_train.sh
  scripts/test_v27_preflight.sh
  scripts/bc_collect_parallel.sh
  scripts/bc_collect_v27_top10.sh
)

case "$cmd" in
  status)
    _ssh "cd '$RDIR' && echo '=== gpu ===' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo '(no gpu)' && echo '=== procs ===' && ps aux | grep -E 'v27_pipeline|v27_extend|bc_collect|train_bc' | grep -v grep || echo '(none)' && echo '=== pipeline status ===' && cat logs/v27_pipeline/status.json 2>/dev/null || echo '(no status.json)'"
    ;;
  sync)
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
      --exclude='logs/' --exclude='ckpt_*/' --exclude='data/' --exclude='.DS_Store' \
      "$ROOT/" "$REMOTE:$RDIR/"
    chmod_cmd=""
    for s in "${V27_SCRIPTS[@]}"; do
      chmod_cmd+=" chmod +x '$RDIR/$s';"
    done
    _ssh "$chmod_cmd true"
    echo "synced"
    ;;
  smoke)
    _ssh "cd '$RDIR' && env PYTHON='$RPY' CHECK_REMOTE=0 bash scripts/v27_smoke.sh"
    ;;
  pipeline)
    EXTRA=()
    while [ $# -gt 0 ]; do EXTRA+=("$1"); shift; done
    if [ ${#EXTRA[@]} -gt 0 ]; then
      _ssh "cd '$RDIR' && mkdir -p logs/v27_pipeline && (nohup env PYTHON='$RPY' bash scripts/v27_pipeline.sh ${EXTRA[@]} > logs/v27_pipeline/nohup.log 2>&1 & echo \$! > logs/v27_pipeline/pipeline.pid) && sleep 2 && tail -5 logs/v27_pipeline/nohup.log"
    else
      _ssh "cd '$RDIR' && mkdir -p logs/v27_pipeline && (nohup env PYTHON='$RPY' bash scripts/v27_pipeline.sh > logs/v27_pipeline/nohup.log 2>&1 & echo \$! > logs/v27_pipeline/pipeline.pid) && sleep 2 && tail -5 logs/v27_pipeline/nohup.log"
    fi
    ;;
  wait)
    echo "Waiting for v27 pipeline to finish..."
    while _ssh "test -f '$RDIR/logs/v27_pipeline/pipeline.pid' && kill -0 \$(cat '$RDIR/logs/v27_pipeline/pipeline.pid') 2>/dev/null"; do
      _ssh "tail -3 '$RDIR/logs/v27_pipeline/pipeline.log' 2>/dev/null || true"
      sleep 120
    done
    _ssh "test -f '$RDIR/logs/v27_pipeline/status.done' && echo DONE || echo FAILED"
    ;;
  stop)
    _ssh "pkill -f 'v27_pipeline.sh' 2>/dev/null || true; pkill -f 'multi_action_v27_frontier' 2>/dev/null || true; pkill -f 'v27_extend.sh' 2>/dev/null || true; sleep 2; echo stopped"
    ;;
  tail)
    _ssh "tail -15 \$(ls -t '$RDIR'/logs/v27_extend_*/train.log 2>/dev/null | head -1) 2>/dev/null || tail -15 '$RDIR/logs/v27_pipeline/pipeline.log'"
    ;;
  eval)
    _ssh "grep '^\[eval_vs_v20\]' \$(ls -t '$RDIR'/logs/v27_extend_*/train.log 2>/dev/null | head -1) | grep -v WARN | tail -20"
    ;;
  log)
    _ssh "tail -30 '$RDIR/logs/v27_pipeline/pipeline.log'"
    ;;
  *)
    echo "usage: $0 {status|sync|smoke|pipeline|wait|stop|tail|eval|log}" >&2
    exit 2
    ;;
esac
