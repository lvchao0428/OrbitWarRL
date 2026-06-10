#!/usr/bin/env bash
# Day11 f40 one-shot pipeline: BC collect -> BC train -> PPO buffer -> replay eval.
#
# Runs sequentially in the foreground; safe to nohup/disown on remote.
#
# Usage (remote, full Day11):
#   nohup bash scripts/run_day11_f40_pipeline.sh > logs/day11_f40_pipeline.nohup 2>&1 &
#
# Smoke (~30 min):
#   DAY11_SMOKE=1 bash scripts/run_day11_f40_pipeline.sh
#
# Skip stages (reuse existing artifacts):
#   SKIP_BC_COLLECT=1 SKIP_BUFFER_ENSURE=1 bash scripts/run_day11_f40_pipeline.sh
#
# Env overrides:
#   BC_GAMES=200 EPOCHS=20 NUM_UPDATES=500 CKPT_EVAL_UPDATES="99 199 299 499"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/charlie/anaconda3/bin/python ]; then
    export PYTHON=/home/charlie/anaconda3/bin/python
  elif command -v python3 >/dev/null 2>&1; then
    export PYTHON=python3
  else
    export PYTHON=python
  fi
fi

# --- defaults (Day11 plan) ---
if [ "${DAY11_SMOKE:-0}" = "1" ]; then
  BC_GAMES="${BC_GAMES:-5}"
  STATE_GAMES="${STATE_GAMES:-0}"
  EPOCHS="${EPOCHS:-5}"
  NUM_UPDATES="${NUM_UPDATES:-5}"
  CKPT_EVAL_UPDATES="${CKPT_EVAL_UPDATES:-4}"
  BC_BATCH_SIZE="${BC_BATCH_SIZE:-128}"
else
  BC_GAMES="${BC_GAMES:-200}"
  STATE_GAMES="${STATE_GAMES:-0}"   # mixed buffer usually already exists
  EPOCHS="${EPOCHS:-20}"
  NUM_UPDATES="${NUM_UPDATES:-500}"
  CKPT_EVAL_UPDATES="${CKPT_EVAL_UPDATES:-99 199 299 499}"
  BC_BATCH_SIZE="${BC_BATCH_SIZE:-256}"
fi

SEED="${SEED:-4200}"
EMIT_POS_WEIGHT="${EMIT_POS_WEIGHT:-4.0}"
JAX_FRAC="${JAX_FRAC:-0.85}"

BC_OUT="${BC_OUT:-data/bc_f40_v20_self.npz}"
BC_CKPT="${BC_CKPT:-ckpt_bc_f40/ckpt_final.pkl}"
MIXED_BUFFER="${MIXED_BUFFER:-data/f40_mixed_states.npz}"
PPO_CKPT_DIR="${PPO_CKPT_DIR:-ckpt_multi_action_v11_f40_buffer}"
PPO_LOG="${PPO_LOG:-logs/v11_f40_buffer.log}"
PPO_TB="${PPO_TB:-logs/v11_f40_buffer}"
PIPELINE_LOG="${PIPELINE_LOG:-logs/day11_f40_pipeline.log}"

SKIP_BC_COLLECT="${SKIP_BC_COLLECT:-0}"
SKIP_BUFFER_ENSURE="${SKIP_BUFFER_ENSURE:-0}"
SKIP_BC_TRAIN="${SKIP_BC_TRAIN:-0}"
SKIP_PPO="${SKIP_PPO:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

mkdir -p logs data ckpt_bc_f40 "$PPO_CKPT_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"
}

stage() {
  log "========== $* =========="
}

log "Day11 f40 pipeline start (smoke=${DAY11_SMOKE:-0})"
log "python=$PYTHON bc_games=$BC_GAMES epochs=$EPOCHS num_updates=$NUM_UPDATES"
log "eval_ckpts=$CKPT_EVAL_UPDATES"

# ------------------------------------------------------------------
# Stage 0: ensure mixed state buffer exists
# ------------------------------------------------------------------
if [ "$SKIP_BUFFER_ENSURE" != "1" ]; then
  stage "Stage 0: ensure mixed state buffer"
  if [ -f "$MIXED_BUFFER" ]; then
    log "mixed buffer exists: $MIXED_BUFFER"
  else
    log "building mixed buffer (STATE_GAMES=${STATE_GAMES:-20})"
    STATE_GAMES="${STATE_GAMES:-20}" \
      BC_GAMES=0 \
      bash scripts/collect_f40_expert_data.sh 2>&1 | tee -a "$PIPELINE_LOG"
  fi
else
  log "skip buffer ensure"
fi

if [ ! -f "$MIXED_BUFFER" ]; then
  log "ERROR: missing $MIXED_BUFFER"
  exit 1
fi

# ------------------------------------------------------------------
# Stage 1: BC data collection
# ------------------------------------------------------------------
if [ "$SKIP_BC_COLLECT" != "1" ]; then
  stage "Stage 1: BC collect ($BC_GAMES games)"
  "$PYTHON" -m orbit_wars_rl.bc.collect_data \
    --num-games "$BC_GAMES" \
    --agent submission_v20_0513.py \
    --opponent submission_v20_0513.py \
    --seed "$SEED" \
    --out "$BC_OUT" \
    2>&1 | tee -a "$PIPELINE_LOG"
else
  log "skip BC collect (use existing $BC_OUT)"
fi

if [ ! -f "$BC_OUT" ]; then
  log "ERROR: missing $BC_OUT"
  exit 1
fi

# ------------------------------------------------------------------
# Stage 2: BC train + replay gate
# ------------------------------------------------------------------
if [ "$SKIP_BC_TRAIN" != "1" ]; then
  stage "Stage 2: BC train + replay"
  DATA="$BC_OUT" \
    OUT="$BC_CKPT" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BC_BATCH_SIZE" \
    EMIT_POS_WEIGHT="$EMIT_POS_WEIGHT" \
    bash scripts/run_f40_bc.sh 2>&1 | tee -a "$PIPELINE_LOG"
else
  log "skip BC train"
fi

if [ ! -f "$BC_CKPT" ]; then
  log "ERROR: missing $BC_CKPT"
  exit 1
fi

# ------------------------------------------------------------------
# Stage 3: PPO buffer curriculum (foreground)
# ------------------------------------------------------------------
if [ "$SKIP_PPO" != "1" ]; then
  stage "Stage 3: PPO buffer ($NUM_UPDATES updates)"
  : > "$PPO_LOG"
  RESUME="$BC_CKPT" \
    NUM_UPDATES="$NUM_UPDATES" \
    CKPT_DIR="$PPO_CKPT_DIR" \
    CKPT_EVERY=100 \
    LOG="$PPO_LOG" \
    TB="$PPO_TB" \
    JAX_FRAC="$JAX_FRAC" \
    FOREGROUND=1 \
    bash scripts/run_v11_f40_buffer.sh 2>&1 | tee -a "$PIPELINE_LOG"
else
  log "skip PPO"
fi

# ------------------------------------------------------------------
# Stage 4: replay eval on key ckpts
# ------------------------------------------------------------------
if [ "$SKIP_EVAL" != "1" ]; then
  stage "Stage 4: replay eval"
  log "BC replay (refresh)"
  PYTHON="$PYTHON" bash scripts/quick_replay.sh "$BC_CKPT" "v11_f40_bc_seed" \
    2>&1 | tee -a "$PIPELINE_LOG"

  for U in $CKPT_EVAL_UPDATES; do
    CKPT="$PPO_CKPT_DIR/ckpt_$(printf '%06d' "$U").pkl"
    if [ ! -f "$CKPT" ]; then
      log "[skip] $CKPT not found"
      continue
    fi
    TAG="v11_f40_buffer_u${U}"
    log "replay $CKPT -> $TAG"
    PYTHON="$PYTHON" bash scripts/quick_replay.sh "$CKPT" "$TAG" \
      2>&1 | tee -a "$PIPELINE_LOG"
  done

  stage "Gate summaries"
  for TAG in v11_f40_bc_seed; do
    SUM="logs/replay_analyze/${TAG}_vs_v20.summary.txt"
    if [ -f "$SUM" ]; then
      head -1 "$SUM" | tee -a "$PIPELINE_LOG"
    fi
  done
  for U in $CKPT_EVAL_UPDATES; do
    SUM="logs/replay_analyze/v11_f40_buffer_u${U}_vs_v20.summary.txt"
    if [ -f "$SUM" ]; then
      head -1 "$SUM" | tee -a "$PIPELINE_LOG"
    fi
  done
else
  log "skip eval"
fi

stage "Pipeline complete"
log "artifacts:"
log "  BC data     : $BC_OUT"
log "  BC ckpt     : $BC_CKPT"
log "  PPO ckpts   : $PPO_CKPT_DIR/"
log "  PPO log     : $PPO_LOG"
log "  replay dir  : logs/replay_analyze/v11_f40_*"
log "  master log  : $PIPELINE_LOG"
log ""
log "Pull to local:"
log "  rsync -avz charlie@www.ultrapp.online:/home/charlie/project/OrbitWarRL/logs/replay_analyze/v11_f40_* logs/replay_analyze/"
log "  rsync -avz charlie@www.ultrapp.online:/home/charlie/project/OrbitWarRL/logs/day11_f40_pipeline.log logs/"
