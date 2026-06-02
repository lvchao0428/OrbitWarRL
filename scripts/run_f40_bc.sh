#!/usr/bin/env bash
# Train an f40 BC seed from expert data and replay it vs v20.
#
# Usage:
#   bash scripts/run_f40_bc.sh
#   DATA=data/bc_f40_v20_self.npz EPOCHS=5 bash scripts/run_f40_bc.sh

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

DATA="${DATA:-data/bc_f40_v20_self.npz}"
OUT="${OUT:-ckpt_bc_f40/ckpt_final.pkl}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.0003}"
EMIT_POS_WEIGHT="${EMIT_POS_WEIGHT:-4.0}"
TAG="${TAG:-v11_f40_bc_seed}"

mkdir -p logs ckpt_bc_f40

echo "[f40-bc] data=$DATA out=$OUT epochs=$EPOCHS"
"$PY" -m orbit_wars_rl.bc.train_bc \
  --data "$DATA" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --emit-pos-weight "$EMIT_POS_WEIGHT" \
  --out "$OUT" \
  2>&1 | tee logs/f40_bc.log

echo "[f40-bc] replay gate: $OUT"
PYTHON="$PY" bash scripts/quick_replay.sh "$OUT" "$TAG"
