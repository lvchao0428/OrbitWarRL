#!/usr/bin/env bash
# Remote control for v30 pipeline.
# Usage: bash scripts/v30_remote.sh {sync|smoke|pipeline|status|tail|eval|stop}
set -euo pipefail
REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
RDIR="${RDIR:-/home/charlie/project/OrbitWarRL}"
RPY="/home/charlie/anaconda3/bin/python"
cmd="${1:-status}"; shift || true
_ssh() { ssh -o ConnectTimeout=15 "$REMOTE" "$@"; }
case "$cmd" in
  status)
    _ssh "cd '$RDIR' && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null; ps aux | grep -E 'v30_pipeline|v30_extend|multi_action_v30' | grep -v grep || echo '(none)'; cat logs/v30_pipeline/status.done 2>/dev/null && echo DONE || true"
    ;;
  sync)
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
      --exclude='logs/' --exclude='ckpt_*/' --exclude='data/' "$ROOT/" "$REMOTE:$RDIR/"
    _ssh "chmod +x '$RDIR'/scripts/v30_*.sh"; echo synced ;;
  smoke) _ssh "cd '$RDIR' && env PYTHON='$RPY' bash scripts/v30_smoke.sh" ;;
  pipeline)
    EXTRA=("$@")
    _ssh "cd '$RDIR' && env PYTHON='$RPY' bash scripts/v30_start_remote.sh ${EXTRA[*]:-}" ;;
  tail) _ssh "tail -15 \$(ls -t '$RDIR'/logs/v30_extend_*/train.log 2>/dev/null | head -1)" ;;
  eval) _ssh "grep '^\[eval_vs_v20\]' \$(ls -t '$RDIR'/logs/v30_extend_*/train.log 2>/dev/null | head -1) | tail -15" ;;
  stop) _ssh "pkill -f v30_pipeline; pkill -f multi_action_v30; echo stopped" ;;
  *) echo "usage: $0 {sync|smoke|pipeline|status|tail|eval|stop}" >&2; exit 2 ;;
esac
