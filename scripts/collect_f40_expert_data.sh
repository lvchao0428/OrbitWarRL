#!/usr/bin/env bash
# Collect f40 expert artifacts:
#   - BC dataset from v20 self-play
#   - v20 state buffer
#   - optional mixed v20 + top10 state buffer if top10 replays are present
#
# Usage:
#   BC_GAMES=50 STATE_GAMES=50 bash scripts/collect_f40_expert_data.sh
#   BC_GAMES=2 STATE_GAMES=2 bash scripts/collect_f40_expert_data.sh  # smoke

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/charlie/anaconda3/bin/python ]; then
    PY=/home/charlie/anaconda3/bin/python
  elif command -v python3 >/dev/null 2>&1; then
    PY=python3
  else
    PY=python
  fi
else
  PY="$PYTHON"
fi

BC_GAMES="${BC_GAMES:-20}"
STATE_GAMES="${STATE_GAMES:-20}"
SEED="${SEED:-4000}"

BC_OUT="${BC_OUT:-data/bc_f40_v20_self.npz}"
V20_STATES="${V20_STATES:-data/f40_v20_states.npz}"
MIXED_OUT="${MIXED_OUT:-data/f40_mixed_states.npz}"
TOP10_STATES="${TOP10_STATES:-data/top10_winner_states.npz}"

mkdir -p data logs

echo "[f40-data] python=$PY"
echo "[f40-data] collect BC: games=$BC_GAMES seed=$SEED -> $BC_OUT"
"$PY" -m orbit_wars_rl.bc.collect_data \
  --num-games "$BC_GAMES" \
  --agent submission_v20_0513.py \
  --opponent submission_v20_0513.py \
  --seed "$SEED" \
  --out "$BC_OUT"

echo "[f40-data] collect state buffer: games=$STATE_GAMES seed=$SEED -> $V20_STATES"
"$PY" -m orbit_wars_rl.bc.collect_states \
  --num-games "$STATE_GAMES" \
  --agent submission_v20_0513.py \
  --opponent submission_v20_0513.py \
  --seed "$SEED" \
  --out "$V20_STATES"

if [ -f "$TOP10_STATES" ]; then
  echo "[f40-data] merge v20 + top10 -> $MIXED_OUT"
  "$PY" -m orbit_wars_rl.bc.merge_state_buffers \
    --inputs "$V20_STATES" "$TOP10_STATES" \
    --out "$MIXED_OUT"
else
  echo "[f40-data] no $TOP10_STATES; using v20 buffer as mixed buffer"
  cp "$V20_STATES" "$MIXED_OUT"
fi

echo "[f40-data] done"
