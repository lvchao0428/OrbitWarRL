#!/usr/bin/env bash
# Remote control for v21 training on www.ultrapp.online
#
# Usage:
#   bash scripts/v21_remote.sh status
#   bash scripts/v21_remote.sh sync         # push code + configs
#   bash scripts/v21_remote.sh stop-v19     # stop previous training
#   bash scripts/v21_remote.sh start        # launch v21 from scratch
#   bash scripts/v21_remote.sh tail         # tail latest log
#   bash scripts/v21_remote.sh eval         # show eval_vs_v20 results
#   bash scripts/v21_remote.sh switch       # sync + stop-v19 + start
set -euo pipefail

REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
RDIR="${RDIR:-/home/charlie/project/OrbitWarRL}"

cmd="${1:-status}"
shift || true

_ssh() {
  ssh -o ConnectTimeout=15 "$REMOTE" "$@"
}

case "$cmd" in
  status)
    _ssh "cd '$RDIR' && echo '=== processes ===' && ps aux | grep -E 'orbit_wars_rl.scripts.train' | grep python | grep -v grep || echo '(none)' && echo && echo '=== latest logs ===' && ls -t logs/v21_lux_align_*/train.log logs/v19_v20_align_*/train.log 2>/dev/null | head -3 && echo && for f in \$(ls -t logs/v21_lux_align_*/train.log logs/v19_v20_align_*/train.log 2>/dev/null | head -1); do echo \"--- tail \$f ---\"; tail -5 \"\$f\"; done"
    ;;
  stop-v19)
    _ssh "pkill -f 'multi_action_v19_v20_align.yaml' 2>/dev/null || true; sleep 2; ps aux | grep v19_v20 | grep python | grep -v grep || echo 'v19 stopped'"
    ;;
  sync)
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    echo "syncing v21 code to $REMOTE:$RDIR ..."
    rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
      --exclude='logs/' --exclude='ckpt_*/' --exclude='data/' --exclude='.DS_Store' \
      --exclude='_archive/' \
      "$ROOT/" "$REMOTE:$RDIR/"
    _ssh "chmod +x '$RDIR/scripts/v21_lux_align.sh' '$RDIR/scripts/v21_remote.sh'"
    echo "synced full codebase"
    ;;
  start)
    NUPD="${1:-15000}"
    _ssh "cd '$RDIR' && nohup bash scripts/v21_lux_align.sh '$NUPD' > logs/v21_launch_\$(date +%Y%m%d_%H%M%S).log 2>&1 & sleep 3 && ps aux | grep 'v21_lux_align' | grep -v grep && ls -t logs/v21_lux_align_*/train.log 2>/dev/null | head -1 || echo 'waiting for log...'"
    ;;
  tail)
    _ssh "tail -10 \$(ls -t '$RDIR'/logs/v21_lux_align_*/train.log 2>/dev/null | head -1)"
    ;;
  eval)
    _ssh "grep '^\[eval_vs_v20\]' \$(ls -t '$RDIR'/logs/v21_lux_align_*/train.log 2>/dev/null | head -1) | grep -v WARN | tail -15"
    ;;
  switch)
    "$0" sync
    "$0" stop-v19
    sleep 2
    "$0" start "${1:-15000}"
    ;;
  *)
    echo "unknown cmd: $cmd" >&2
    echo "usage: $0 {status|sync|stop-v19|start|tail|eval|switch}" >&2
    exit 2
    ;;
esac
