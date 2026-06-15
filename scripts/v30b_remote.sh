#!/usr/bin/env bash
# Remote control for v30b pipeline.
set -euo pipefail
REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
RDIR="${RDIR:-/home/charlie/project/OrbitWarRL}"
RPY="/home/charlie/anaconda3/bin/python"
cmd="${1:-status}"; shift || true
_ssh() { ssh -o ConnectTimeout=15 "$REMOTE" "$@"; }
case "$cmd" in
  status)
    _ssh "cd '$RDIR' && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null; ps aux | grep -E 'v30b_pipeline|v30b_extend|multi_action_v30b' | grep -v grep || echo '(none)'; cat logs/v30b_pipeline/status.done 2>/dev/null && echo DONE || true"
    ;;
  sync)
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
      --exclude='logs/' --exclude='ckpt_*/' --exclude='data/' "$ROOT/" "$REMOTE:$RDIR/"
    _ssh "chmod +x '$RDIR'/scripts/v30b_*.sh"; echo synced ;;
  smoke) _ssh "cd '$RDIR' && env PYTHON='$RPY' JAX_PLATFORMS=cpu bash scripts/v30b_smoke.sh" ;;
  pipeline)
    EXTRA=("$@")
    _ssh "cd '$RDIR' && env PYTHON='$RPY' bash scripts/v30b_start_remote.sh ${EXTRA[*]:-}" ;;
  tail) _ssh "tail -15 \$(ls -t '$RDIR'/logs/v30b_extend_*/train.log 2>/dev/null | head -1)" ;;
  eval) _ssh "grep '^\[eval_vs_v20\]' \$(ls -t '$RDIR'/logs/v30b_extend_*/train.log 2>/dev/null | head -1) | tail -15" ;;
  stop) _ssh "pkill -f v30b_pipeline; pkill -f multi_action_v30b; echo stopped" ;;
  *) echo "usage: $0 {sync|smoke|pipeline|status|tail|eval|stop}" >&2; exit 2 ;;
esac
