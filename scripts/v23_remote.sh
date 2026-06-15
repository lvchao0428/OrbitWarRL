#!/usr/bin/env bash
# Remote control for v23 Route A training
#
# Usage:
#   bash scripts/v23_remote.sh status
#   bash scripts/v23_remote.sh sync
#   bash scripts/v23_remote.sh stop-v22
#   bash scripts/v23_remote.sh start [num_updates]
#   bash scripts/v23_remote.sh tail
#   bash scripts/v23_remote.sh eval
#   bash scripts/v23_remote.sh switch [num_updates]
set -euo pipefail

REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
RDIR="${RDIR:-/home/charlie/project/OrbitWarRL}"

cmd="${1:-status}"
shift || true

_ssh() { ssh -o ConnectTimeout=15 "$REMOTE" "$@"; }

case "$cmd" in
  status)
    _ssh "cd '$RDIR' && echo '=== processes ===' && ps aux | grep -E 'orbit_wars_rl.scripts.train|v2[23]_' | grep python | grep -v grep || echo '(none)' && echo && ls -t logs/v23_route_a_*/train.log logs/v22_align_fix_*/train.log 2>/dev/null | head -3 && echo && for f in \$(ls -t logs/v23_route_a_*/train.log 2>/dev/null | head -1); do echo \"--- tail \$f ---\"; tail -5 \"\$f\"; done"
    ;;
  sync)
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
      --exclude='logs/' --exclude='ckpt_*/' --exclude='data/' --exclude='.DS_Store' \
      --exclude='_archive/' \
      "$ROOT/" "$REMOTE:$RDIR/"
    _ssh "chmod +x '$RDIR/scripts/v23_route_a.sh' '$RDIR/scripts/v23_remote.sh'"
    echo "synced"
    ;;
  stop-v22)
    # [x] brackets stop pkill -f from matching this ssh command line itself.
    _ssh "pkill -f 'multi_action_v22_align_fi[x].yaml' 2>/dev/null || true; pkill -f 'v22_align_fi[x].sh' 2>/dev/null || true; sleep 2; ps aux | grep 'v22_alig[n]' | grep python || echo 'v22 stopped'"
    ;;
  stop-v23)
    _ssh "pkill -f 'multi_action_v23_route_[a].yaml' 2>/dev/null || true; pkill -f 'v23_route_[a].sh' 2>/dev/null || true; sleep 2; ps aux | grep 'v23_route_[a]' | grep python || echo 'v23 stopped'"
    ;;
  start)
    NUPD="${1:-8000}"
    _ssh "cd '$RDIR' && nohup bash scripts/v23_route_a.sh '$NUPD' > logs/v23_launch_\$(date +%Y%m%d_%H%M%S).log 2>&1 & sleep 3 && ps aux | grep v23_route | grep python | grep -v grep && ls -t logs/v23_route_a_*/train.log 2>/dev/null | head -1"
    ;;
  tail)
    _ssh "tail -10 \$(ls -t '$RDIR'/logs/v23_route_a_*/train.log 2>/dev/null | head -1)"
    ;;
  eval)
    _ssh "grep '^\[eval_vs_v20\]' \$(ls -t '$RDIR'/logs/v23_route_a_*/train.log 2>/dev/null | head -1) | grep -v WARN | tail -15"
    ;;
  switch)
    "$0" sync
    "$0" stop-v22
    "$0" start "${1:-8000}"
    ;;
  *)
    echo "usage: $0 {status|sync|stop-v22|stop-v23|start|tail|eval|switch}" >&2
    exit 2
    ;;
esac
