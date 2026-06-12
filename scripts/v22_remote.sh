#!/usr/bin/env bash
# Remote control for v22 training
#
# Usage:
#   bash scripts/v22_remote.sh status
#   bash scripts/v22_remote.sh sync
#   bash scripts/v22_remote.sh start [num_updates]
#   bash scripts/v22_remote.sh tail
#   bash scripts/v22_remote.sh eval
set -euo pipefail

REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
RDIR="${RDIR:-/home/charlie/project/OrbitWarRL}"

cmd="${1:-status}"
shift || true

_ssh() { ssh -o ConnectTimeout=15 "$REMOTE" "$@"; }

case "$cmd" in
  status)
    _ssh "cd '$RDIR' && echo '=== processes ===' && ps aux | grep -E 'orbit_wars_rl.scripts.train' | grep python | grep -v grep || echo '(none)' && echo && ls -t logs/v22_align_fix_*/train.log logs/v21_lux_align_*/train.log 2>/dev/null | head -2 && echo && for f in \$(ls -t logs/v22_align_fix_*/train.log 2>/dev/null | head -1); do echo \"--- tail \$f ---\"; tail -5 \"\$f\"; done"
    ;;
  sync)
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
      --exclude='logs/' --exclude='ckpt_*/' --exclude='data/' --exclude='.DS_Store' \
      --exclude='_archive/' \
      "$ROOT/" "$REMOTE:$RDIR/"
    _ssh "chmod +x '$RDIR/scripts/v22_align_fix.sh' '$RDIR/scripts/v22_remote.sh'"
    echo "synced"
    ;;
  start)
    NUPD="${1:-10000}"
    _ssh "cd '$RDIR' && nohup bash scripts/v22_align_fix.sh '$NUPD' > logs/v22_launch_\$(date +%Y%m%d_%H%M%S).log 2>&1 & sleep 3 && ps aux | grep v22_align | grep -v grep && ls -t logs/v22_align_fix_*/train.log 2>/dev/null | head -1"
    ;;
  tail)
    _ssh "tail -10 \$(ls -t '$RDIR'/logs/v22_align_fix_*/train.log 2>/dev/null | head -1)"
    ;;
  eval)
    _ssh "grep '^\[eval_vs_v20\]' \$(ls -t '$RDIR'/logs/v22_align_fix_*/train.log 2>/dev/null | head -1) | grep -v WARN | tail -15"
    ;;
  switch)
    "$0" sync
    "$0" start "${1:-10000}"
    ;;
  *)
    echo "usage: $0 {status|sync|start|tail|eval|switch}" >&2
    exit 2
    ;;
esac
