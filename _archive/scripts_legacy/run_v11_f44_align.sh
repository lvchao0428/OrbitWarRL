#!/usr/bin/env bash
# v11_f44_align — train/replay alignment (P0–P4). Resume f42 @49, f42 shaping.
#
# Key deltas vs f42:
#   eval_vs_v20: 3-game inline replay every eval_every
#   align_roll_window: 20 (strn+frzn rolling metrics in log)
#   buffer_rollout_ratio: 0.15 (was 0.40)
#   strong_ratio: 0.35 (was 0.25)
#   ckpt .meta.json with opp_tag
#
# Usage:
#   bash scripts/run_v11_f44_align.sh
#   FOREGROUND=1 bash scripts/run_v11_f44_align.sh
#   NUM_UPDATES=100 bash scripts/run_v11_f44_align.sh   # smoke

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

CFG="${CFG:-orbit_wars_rl/configs/multi_action_v11_f44_align.yaml}"
RESUME="${RESUME:-ckpt_multi_action_v11_f42/ckpt_000049.pkl}"
NUM_UPDATES="${NUM_UPDATES:-}"
JAX_FRAC="${JAX_FRAC:-0.85}"
LOG="${LOG:-logs/v11_f44_align.log}"
TB="${TB:-logs/v11_f44_align}"
FOREGROUND="${FOREGROUND:-0}"

mkdir -p logs

if [ ! -f "$RESUME" ]; then
  echo "[f44] ERROR: resume ckpt not found: $RESUME" >&2
  exit 1
fi
if [ ! -f "data/f40_mixed_states.npz" ]; then
  echo "[f44] ERROR: missing data/f40_mixed_states.npz" >&2
  exit 1
fi

ARGS=(--config "$CFG" --log-dir "$TB" --resume-from "$RESUME")
if [ -n "$NUM_UPDATES" ]; then
  ARGS+=(--num-updates "$NUM_UPDATES")
fi

TRAIN_ENV=(
  ORBITWARS_SHAPING_CAPTURE="${ORBITWARS_SHAPING_CAPTURE:-0.05}"
  ORBITWARS_SHAPING_CAPTURE_FLEET_SCALE="${ORBITWARS_SHAPING_CAPTURE_FLEET_SCALE:-0.10}"
  ORBITWARS_SHAPING_PROD_SHARE_DELTA="${ORBITWARS_SHAPING_PROD_SHARE_DELTA:-0.005}"
  ORBITWARS_SHAPING_ONE_SHIP_PENALTY="${ORBITWARS_SHAPING_ONE_SHIP_PENALTY:-0.005}"
  ORBITWARS_SHAPING_ONE_SHIP_THRESH="${ORBITWARS_SHAPING_ONE_SHIP_THRESH:-3.0}"
  ORBITWARS_SHAPING_HIGH_PROD_CAPTURE="${ORBITWARS_SHAPING_HIGH_PROD_CAPTURE:-0.02}"
  ORBITWARS_SHAPING_HIGH_PROD_THRESH="${ORBITWARS_SHAPING_HIGH_PROD_THRESH:-3.0}"
  ORBITWARS_SHAPING_EMIT_LOG=0.0
  ORBITWARS_SHAPING_EMIT_GATED=0
  ORBITWARS_SHAPING_PLANET_SHARE=0.0
  ORBITWARS_SHAPING_PROD_SHARE=0.0
  ORBITWARS_SHAPING_FLEET_LOG=0.0
  ORBITWARS_SHAPING_RELEASE=0.0
  ORBITWARS_SHAPING_RELEASE_K=20.0
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC"
)

echo "[f44] py=$PY cfg=$CFG resume=$RESUME log=$LOG"
echo "[f44] align: eval_vs_v20=3games buffer_rollout=0.15 strong=0.35"
echo "[f44] shaping: same as f42 (CAPTURE_FLEET_SCALE=0.10)"
if [ "$FOREGROUND" = "1" ]; then
  env "${TRAIN_ENV[@]}" \
    "$PY" -m orbit_wars_rl.scripts.train "${ARGS[@]}" \
    2>&1 | tee -a "$LOG"
else
  env "${TRAIN_ENV[@]}" \
    "$PY" -m orbit_wars_rl.scripts.train "${ARGS[@]}" \
    >> "$LOG" 2>&1 &
  echo "pid=$!"
  echo "tail -f $LOG"
  echo "# parse log by opp: $PY scripts/parse_train_log_by_opp.py $LOG --align-only"
fi
