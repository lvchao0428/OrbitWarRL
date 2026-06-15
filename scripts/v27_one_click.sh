#!/usr/bin/env bash
# v27 one-click: sync code -> remote Gate0 -> remote full pipeline.
#
# Usage:
#   bash scripts/v27_one_click.sh
#   bash scripts/v27_one_click.sh --skip-post
#   bash scripts/v27_one_click.sh --updates 12000
#   bash scripts/v27_one_click.sh --from gate3   # resume from PPO only
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SKIP_POST=0
UPDATES=4000
FROM_GATE=""
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-post) SKIP_POST=1; shift ;;
    --updates) UPDATES="$2"; shift 2 ;;
    --from) FROM_GATE="$2"; shift 2 ;;
    *) echo "unknown: $1" >&2; exit 2 ;;
  esac
done

PIPE_ARGS=(--updates "$UPDATES")
[ "$SKIP_POST" -eq 1 ] && PIPE_ARGS+=(--skip-post)
[ -n "$FROM_GATE" ] && PIPE_ARGS+=(--from "$FROM_GATE")

echo "═══════════════════════════════════════════════════"
echo " v27 one-click pipeline"
echo "═══════════════════════════════════════════════════"
echo " step 1/3: sync code to remote"
bash scripts/v27_remote.sh sync

echo ""
echo " step 2/3: Gate0 smoke (remote)"
bash scripts/v27_remote.sh smoke

echo ""
echo " step 3/3: start remote pipeline (BC 200g + top10 || PPO ~3.5h)"
bash scripts/v27_remote.sh pipeline "${PIPE_ARGS[@]}"

echo ""
echo "Pipeline started on remote (~3.5h total)."
echo "  bash scripts/v27_remote.sh status"
echo "  bash scripts/v27_remote.sh tail"
echo "  bash scripts/v27_remote.sh eval"
echo "  bash scripts/v27_remote.sh wait    # block until done"
