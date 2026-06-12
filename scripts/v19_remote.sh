#!/usr/bin/env bash
# Remote control for v18/v19 training on www.ultrapp.online
#
# Usage:
#   bash scripts/v19_remote.sh status
#   bash scripts/v19_remote.sh stop-v18
#   bash scripts/v19_remote.sh sync
#   bash scripts/v19_remote.sh start-v19 [ckpt.pkl] [num_updates]
#   bash scripts/v19_remote.sh tail-v19
set -euo pipefail

REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
RDIR="${RDIR:-/home/charlie/project/OrbitWarRL}"
PY="${REMOTE_PY:-/home/charlie/anaconda3/bin/python}"

cmd="${1:-status}"
shift || true

_ssh() {
  ssh "$REMOTE" "$@"
}

case "$cmd" in
  status)
    _ssh "cd '$RDIR' && echo '=== processes ===' && ps aux | grep -E 'orbit_wars_rl.scripts.train|v1[89]_' | grep python | grep -v grep || echo '(none)' && echo && echo '=== latest logs ===' && ls -t logs/v18_multi_emit_*/train.log logs/v19_v20_align_*/train.log 2>/dev/null | head -3 && echo && for f in \$(ls -t logs/v19_v20_align_*/train.log logs/v18_multi_emit_*/train.log 2>/dev/null | head -1); do echo \"--- tail \$f ---\"; tail -3 \"\$f\"; done"
    ;;
  stop-v18)
    _ssh "pkill -f 'multi_action_v18_multi_emit.yaml' 2>/dev/null; true; sleep 2; ps aux | grep v18_multi | grep python | grep -v grep || echo 'v18 stopped'"
    ;;
  stop-v19)
    _ssh "pkill -f 'multi_action_v19_v20_align.yaml' 2>/dev/null; true; sleep 2; ps aux | grep v19_v20 | grep python | grep -v grep || echo 'v19 stopped'"
    ;;
  sync)
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    scp "$ROOT/orbit_wars_rl/configs/multi_action_v19_v20_align.yaml" \
      "$REMOTE:$RDIR/orbit_wars_rl/configs/multi_action_v19_v20_align.yaml"
    scp "$ROOT/scripts/v19_v20_align.sh" "$REMOTE:$RDIR/scripts/v19_v20_align.sh"
    scp "$ROOT/scripts/v19_remote.sh" "$REMOTE:$RDIR/scripts/v19_remote.sh"
    _ssh "chmod +x '$RDIR/scripts/v19_v20_align.sh' '$RDIR/scripts/v19_remote.sh'"
    echo "synced v19 config + scripts"
    ;;
  start-v19)
    RESUME="${1:-./ckpt_multi_action_v18_multi_emit/ckpt_001799.pkl}"
    NUPD="${2:-6000}"
    _ssh "cd '$RDIR' && nohup bash scripts/v19_v20_align.sh '$RESUME' '$NUPD' > logs/v19_launch_\$(date +%Y%m%d_%H%M%S).log 2>&1 & sleep 2 && ps aux | grep v19_v20 | grep python | grep -v grep && ls -t logs/v19_v20_align_*/train.log 2>/dev/null | head -1"
    ;;
  tail-v19)
    _ssh "tail -5 \$(ls -t '$RDIR'/logs/v19_v20_align_*/train.log 2>/dev/null | head -1)"
    ;;
  eval-v19)
    _ssh "grep '^\[eval_vs_v20\]' \$(ls -t '$RDIR'/logs/v19_v20_align_*/train.log 2>/dev/null | head -1) | grep -v WARN | tail -10"
    ;;
  switch)
    "$0" sync
    "$0" stop-v18
    RESUME="${1:-./ckpt_multi_action_v18_multi_emit/ckpt_001799.pkl}"
    NUPD="${2:-6000}"
    "$0" start-v19 "$RESUME" "$NUPD"
    ;;
  *)
    echo "unknown cmd: $cmd" >&2
    echo "usage: $0 {status|sync|stop-v18|stop-v19|start-v19|tail-v19|eval-v19|switch}" >&2
    exit 2
    ;;
esac
