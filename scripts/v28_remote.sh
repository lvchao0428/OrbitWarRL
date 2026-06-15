#!/usr/bin/env bash
# Remote control for v28 one-click pipeline.
#
# Usage:
#   bash scripts/v28_remote.sh sync
#   bash scripts/v28_remote.sh smoke
#   bash scripts/v28_remote.sh pipeline [--updates N] [--skip-post]
#   bash scripts/v28_remote.sh status
#   bash scripts/v28_remote.sh tail
#   bash scripts/v28_remote.sh eval
#   bash scripts/v28_remote.sh wait
#   bash scripts/v28_remote.sh stop
set -euo pipefail

REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
RDIR="${RDIR:-/home/charlie/project/OrbitWarRL}"
RPY="/home/charlie/anaconda3/bin/python"

cmd="${1:-status}"
shift || true

_ssh() { ssh -o ConnectTimeout=15 "$REMOTE" "$@"; }

V28_SCRIPTS=(
  scripts/v28_extend.sh
  scripts/v28_remote.sh
  scripts/v28_pipeline.sh
  scripts/v28_smoke.sh
  scripts/v28_one_click.sh
  scripts/v28_post_train.sh
  scripts/bc_collect_v27_top10.sh
)

case "$cmd" in
  status)
    _ssh "cd '$RDIR' && echo '=== gpu ===' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo '(no gpu)' && echo '=== procs ===' && ps aux | grep -E 'v28_pipeline|v28_extend|v27_pipeline|bc_collect|train_bc' | grep -v grep || echo '(none)' && echo '=== pipeline status ===' && cat logs/v28_pipeline/status.json 2>/dev/null || echo '(no status.json)'"
    ;;
  sync)
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
      --exclude='logs/' --exclude='ckpt_*/' --exclude='data/' --exclude='.DS_Store' \
      "$ROOT/" "$REMOTE:$RDIR/"
    chmod_cmd=""
    for s in "${V28_SCRIPTS[@]}"; do
      chmod_cmd+=" chmod +x '$RDIR/$s';"
    done
    _ssh "$chmod_cmd true"
    echo "synced"
    ;;
  smoke)
    _ssh "cd '$RDIR' && env PYTHON='$RPY' CHECK_REMOTE=0 bash scripts/v28_smoke.sh"
    ;;
  pipeline)
    EXTRA=()
    while [ $# -gt 0 ]; do EXTRA+=("$1"); shift; done
    PIPE_CMD="cd '$RDIR' && mkdir -p logs/v28_pipeline"
    PIPE_CMD+=" && pkill -f 'v27_pipeline.sh' 2>/dev/null || true"
    PIPE_CMD+=" && pkill -f 'multi_action_v27_frontier' 2>/dev/null || true"
    if [ ${#EXTRA[@]} -gt 0 ]; then
      PIPE_CMD+=" && nohup env PYTHON='$RPY' bash scripts/v28_pipeline.sh ${EXTRA[*]} >> logs/v28_pipeline/nohup.log 2>&1 &"
    else
      PIPE_CMD+=" && nohup env PYTHON='$RPY' bash scripts/v28_pipeline.sh >> logs/v28_pipeline/nohup.log 2>&1 &"
    fi
    PIPE_CMD+=" echo \$! > logs/v28_pipeline/pipeline.pid"
    PIPE_CMD+=" && sleep 3 && tail -8 logs/v28_pipeline/nohup.log"
    _ssh "$PIPE_CMD"
    ;;
  wait)
    echo "Waiting for v28 pipeline to finish..."
    while _ssh "test -f '$RDIR/logs/v28_pipeline/pipeline.pid' && kill -0 \$(cat '$RDIR/logs/v28_pipeline/pipeline.pid') 2>/dev/null"; do
      _ssh "tail -3 '$RDIR/logs/v28_pipeline/pipeline.log' 2>/dev/null || true"
      sleep 120
    done
    _ssh "test -f '$RDIR/logs/v28_pipeline/status.done' && echo DONE || echo FAILED"
    ;;
  stop)
    _ssh "pkill -f 'v28_pipeline.sh' 2>/dev/null || true; pkill -f 'multi_action_v28_roi' 2>/dev/null || true; pkill -f 'v28_extend.sh' 2>/dev/null || true; sleep 2; echo stopped"
    ;;
  tail)
    _ssh "tail -15 \$(ls -t '$RDIR'/logs/v28_extend_*/train.log 2>/dev/null | head -1) 2>/dev/null || tail -15 '$RDIR/logs/v28_pipeline/pipeline.log'"
    ;;
  eval)
    _ssh "grep '^\[eval_vs_v20\]' \$(ls -t '$RDIR'/logs/v28_extend_*/train.log 2>/dev/null | head -1) | grep -v WARN | tail -20"
    ;;
  log)
    _ssh "tail -30 '$RDIR/logs/v28_pipeline/pipeline.log'"
    ;;
  *)
    echo "usage: $0 {status|sync|smoke|pipeline|wait|stop|tail|eval|log}" >&2
    exit 2
    ;;
esac
