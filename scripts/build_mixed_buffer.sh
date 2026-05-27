#!/usr/bin/env bash
# Build balanced v20 + top10 state buffer for Plan B mix training.
#
# Usage (5090, from repo root):
#   bash scripts/build_mixed_buffer.sh
#   bash scripts/build_mixed_buffer.sh --smoke   # 10 JSON files only
#
# Outputs:
#   data/top10_winner_states.npz   (if not present or --rebuild-top10)
#   data/mixed_v20_top10.npz       (balanced 50/50 pool)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Prefer conda on 5090 (non-interactive ssh has no `python` on PATH).
if [ -z "${PYTHON:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PY=python
  elif [ -x /home/charlie/anaconda3/bin/python ]; then
    PY=/home/charlie/anaconda3/bin/python
  else
    PY=python3
  fi
else
  PY="$PYTHON"
fi

TOP10_DIR="${TOP10_DIR:-}"
if [ -z "$TOP10_DIR" ]; then
  for candidate in \
      top10_episodes_2026-05-04/episodes/episodes \
      episodes/episodes; do
    if [ -d "$candidate" ]; then
      TOP10_DIR="$candidate"
      break
    fi
  done
fi
V20_NPZ="${V20_NPZ:-data/v20_states_200g.npz}"
TOP10_NPZ="${TOP10_NPZ:-data/top10_winner_states.npz}"
MIXED_NPZ="${MIXED_NPZ:-data/mixed_v20_top10.npz}"
REBUILD_TOP10=0
SMOKE=0

for arg in "$@"; do
  case "$arg" in
    --rebuild-top10) REBUILD_TOP10=1 ;;
    --smoke) SMOKE=1 ;;
    *)
      echo "Unknown option: $arg (use --smoke or --rebuild-top10)" >&2
      exit 1
      ;;
  esac
done

if [ -z "$TOP10_DIR" ] || [ ! -d "$TOP10_DIR" ]; then
  echo "[build_mixed_buffer] ERROR: top10 replay dir not found." >&2
  echo "  Download: python download_top10_episodes_2026-05-04.py" >&2
  echo "  Or set TOP10_DIR=top10_episodes_2026-05-04/episodes/episodes" >&2
  exit 1
fi
echo "[build_mixed_buffer] TOP10_DIR=$TOP10_DIR"

if [ ! -f "$V20_NPZ" ]; then
  echo "[build_mixed_buffer] ERROR: missing $V20_NPZ" >&2
  echo "  Run collect_states first or set V20_NPZ." >&2
  exit 1
fi

COLLECT_ARGS=(--replay-dir "$TOP10_DIR" --winners-only --out "$TOP10_NPZ")
if [ "$SMOKE" -eq 1 ]; then
  TOP10_NPZ="data/top10_winner_states_smoke.npz"
  MIXED_NPZ="data/mixed_v20_top10_smoke.npz"
  COLLECT_ARGS=(--replay-dir "$TOP10_DIR" --winners-only --out "$TOP10_NPZ" --max-files 10)
fi

if [ "$REBUILD_TOP10" -eq 1 ] || [ ! -f "$TOP10_NPZ" ]; then
  echo "[build_mixed_buffer] collecting top10 states → $TOP10_NPZ"
  "$PY" -m orbit_wars_rl.bc.collect_states_from_json "${COLLECT_ARGS[@]}"
else
  echo "[build_mixed_buffer] reuse existing $TOP10_NPZ (pass --rebuild-top10 to refresh)"
fi

echo "[build_mixed_buffer] merging $V20_NPZ + $TOP10_NPZ → $MIXED_NPZ"
"$PY" -m orbit_wars_rl.bc.merge_state_buffers \
  --inputs "$V20_NPZ" "$TOP10_NPZ" \
  --out "$MIXED_NPZ"

echo "[build_mixed_buffer] done: $MIXED_NPZ"
