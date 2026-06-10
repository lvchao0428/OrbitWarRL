#!/usr/bin/env bash
# v11_f33: f31 arch (5+4 pair, hard masks bin5) + top-player PPO hyperparams,
#          long train 2000 upd. Single delta vs f31: hyperparams + train volume.
#
# Hyperparams (vs f31): ent_coef 0.05 (single), lr 3e-5, ppo_epochs 1,
#   clip_eps 0.20, value_coef 0.5, num_updates 2000. arch/features FROZEN to f31.
#
# Usage:
#   bash scripts/run_v11_f33.sh
#   tail -f logs/v11_f33.log
#
# Monitor clip_frac (top-player's #1 warning sign):
#   bash scripts/check_training_health.sh logs/v11_f33.log
#   -> if clip_frac creeps past 0.25 monotonically, cut lr (2e-5/1e-5) or revert.
#
# Optional warm-start from f29 @599 (arch-compatible with f31):
#   RESUME=ckpt_multi_action_v11_f29/ckpt_000599.pkl bash scripts/run_v11_f33.sh
#
# After training (replay vs v20 gate):
#   bash scripts/quick_replay.sh ckpt_multi_action_v11_f33/ckpt_000600.pkl v11_f33_u600
#   bash scripts/quick_replay.sh ckpt_multi_action_v11_f33/ckpt_001200.pkl v11_f33_u1200
#   bash scripts/quick_replay.sh ckpt_multi_action_v11_f33/ckpt_002000.pkl v11_f33_u2000

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/charlie/anaconda3/bin/python ]; then
    PY=/home/charlie/anaconda3/bin/python
  elif command -v python3 >/dev/null 2>&1; then
    PY=python3
  elif command -v python >/dev/null 2>&1; then
    PY=python
  else
    PY=python3
  fi
else
  PY="$PYTHON"
fi

JAX_FRAC="${JAX_FRAC:-0.85}"
CFG="orbit_wars_rl/configs/multi_action_v11_f33.yaml"
LOG="logs/v11_f33.log"
TB="logs/v11_f33"
RESUME="${RESUME:-}"

# f31 shaping (NOT f32): FLEET_LOG=0.0, RELEASE=0.05. One delta at a time.
COMMON_SHAPING=(
  "ORBITWARS_SHAPING_EMIT_LOG=0.0"
  "ORBITWARS_SHAPING_EMIT_GATED=0"
  "ORBITWARS_SHAPING_PLANET_SHARE=0.005"
  "ORBITWARS_SHAPING_PROD_SHARE=0.0"
  "ORBITWARS_SHAPING_FLEET_LOG=0.0"
  "ORBITWARS_SHAPING_PROD_SHARE_DELTA=1.0"
  "ORBITWARS_SHAPING_RELEASE=0.05"
  "ORBITWARS_SHAPING_RELEASE_K=20.0"
  "ORBITWARS_SHAPING_CAPTURE=0.02"
)

mkdir -p logs

echo "[v11_f33] py=$PY  cfg=$CFG  log=$LOG"
echo "[v11_f33] f31 arch (5+4 pair, hard masks bin5) + top-player hparams, 2000 upd"
echo "[v11_f33] hparams: ent=0.05  lr=3e-5  ppo_epochs=1  clip_eps=0.20  vf=0.5"
if [ -n "$RESUME" ]; then
  echo "[v11_f33] warm-start from: $RESUME"
fi

echo "[v11_f33] parity smoke (fresh-init, f31 masks)"
if ! JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  "$PY" -m orbit_wars_rl.inference.test_parity \
    --num-states 16 --emit-hard-stop --flip-hard-mask; then
  echo "[v11_f33] parity FAILED — aborting" >&2
  exit 1
fi

RESUME_ARGS=()
if [ -n "$RESUME" ]; then
  RESUME_ARGS=(--resume-from "$RESUME")
fi

env "${COMMON_SHAPING[@]}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC" \
  "$PY" -m orbit_wars_rl.scripts.train \
    --config "$CFG" \
    --log-dir "$TB" \
    "${RESUME_ARGS[@]}" \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "tail -f $LOG"
echo "monitor: bash scripts/check_training_health.sh $LOG"
