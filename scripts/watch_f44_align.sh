#!/usr/bin/env bash
# Quick status for f44_align training (local or after rsync log).
#
# Usage:
#   bash scripts/watch_f44_align.sh
#   bash scripts/watch_f44_align.sh logs/v11_f44_align.log

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${1:-$ROOT/logs/v11_f44_align.log}"

if [ ! -f "$LOG" ]; then
  echo "[watch] log not found: $LOG"
  echo "  rsync -avz charlie@www.ultrapp.online:/home/charlie/project/OrbitWarRL/logs/v11_f44_align.log logs/"
  exit 1
fi

echo "=== f44_align log: $LOG ==="
echo "--- last train line ---"
grep '^upd' "$LOG" | tail -1

echo ""
echo "--- eval_vs_v20 (inline gate) ---"
grep '^\[eval_vs_v20\]' "$LOG" || echo "(none yet — first at upd 49)"

echo ""
echo "--- ckpt saves ---"
grep '^\[ckpt\]' "$LOG" | tail -5

echo ""
echo "--- v20 metrics table (from eval lines) ---"
grep '^\[eval_vs_v20\]' "$LOG" | sed 's/.*tag=//' | awk -F'  ' '{print $0}' || true

PY="${PYTHON:-python3}"
if [ -x /home/charlie/anaconda3/bin/python ]; then PY=/home/charlie/anaconda3/bin/python; fi
if [ -f "$LOG" ] && grep -q '^upd.*opp ' "$LOG"; then
  echo ""
  echo "--- train metrics by opp (align proxy) ---"
  "$PY" "$ROOT/scripts/parse_train_log_by_opp.py" "$LOG" --align-only 2>/dev/null || true
fi

CKPT_DIR="$ROOT/ckpt_multi_action_v11_f44_align"
if [ -d "$CKPT_DIR" ]; then
  echo ""
  echo "--- latest .meta.json ---"
  ls -1t "$CKPT_DIR"/*.meta.json 2>/dev/null | head -3 | while read -r m; do
    echo "  $m"
    grep -E 'opp_tag|eval_vs_v20|align_' "$m" 2>/dev/null | head -8
  done
fi
