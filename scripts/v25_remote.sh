#!/usr/bin/env bash
# Remote control for v25 extend training.
#
# Usage:
#   bash scripts/v25_remote.sh status
#   bash scripts/v25_remote.sh sync
#   bash scripts/v25_remote.sh start [nupd]
#   bash scripts/v25_remote.sh stop
#   bash scripts/v25_remote.sh tail
#   bash scripts/v25_remote.sh eval
#   bash scripts/v25_remote.sh replay [ckpt] [seed]
set -euo pipefail

REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
RDIR="${RDIR:-/home/charlie/project/OrbitWarRL}"
RPY="/home/charlie/anaconda3/bin/python"

cmd="${1:-status}"
shift || true

_ssh() { ssh -o ConnectTimeout=15 "$REMOTE" "$@"; }

case "$cmd" in
  status)
    _ssh "cd '$RDIR' && echo '=== gpu ===' && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader && echo '=== procs ===' && ps aux | grep -E 'multi_action_v25_extend|v25_extend' | grep -v grep || echo '(none)'"
    ;;
  sync)
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
      --exclude='logs/' --exclude='ckpt_*/' --exclude='data/' --exclude='.DS_Store' \
      --exclude='_archive/' \
      "$ROOT/" "$REMOTE:$RDIR/"
    _ssh "chmod +x '$RDIR/scripts/v25_extend.sh' '$RDIR/scripts/v25_remote.sh'"
    echo "synced"
    ;;
  start)
    NUPD="${1:-11000}"
    _ssh "cd '$RDIR' && (nohup bash scripts/v25_extend.sh '$NUPD' > logs/v25_launch_\$(date +%Y%m%d_%H%M%S).log 2>&1 &) && sleep 3 && ps aux | grep 'v25_extend' | grep -v grep && ls -t logs/v25_extend_*/train.log 2>/dev/null | head -1"
    ;;
  stop)
    _ssh "pkill -f 'multi_action_v25_extend.yaml' 2>/dev/null || true; pkill -f 'v25_extend.sh' 2>/dev/null || true; sleep 2; ps aux | grep 'v25_extend' | grep python || echo 'v25 stopped'"
    ;;
  tail)
    _ssh "tail -12 \$(ls -t '$RDIR'/logs/v25_extend_*/train.log 2>/dev/null | head -1)"
    ;;
  eval)
    _ssh "grep '^\[eval_vs_v20\]' \$(ls -t '$RDIR'/logs/v25_extend_*/train.log 2>/dev/null | head -1) | grep -v WARN | tail -20"
    ;;
  replay)
    CKPT="${1:-ckpt_multi_action_v25_extend/ckpt_latest.pkl}"
    SEED="${2:-0}"
    _ssh "cd '$RDIR' && PYTHON='$RPY' bash scripts/v12_replay_html.sh '$CKPT' '$SEED'"
    ;;
  *)
    echo "usage: $0 {status|sync|start|stop|tail|eval|replay}" >&2
    exit 2
    ;;
esac
