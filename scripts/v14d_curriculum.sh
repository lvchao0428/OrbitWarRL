#!/usr/bin/env bash
# v14d — 三阶段 curriculum 从头训练（5090 / 本地 GPU）
#
# Phase A: 囤兵/耐心 (allow_hold, 无 force_emit_worth_it, HOLD shaping)
# Phase B: 占点/进攻 (resume A, force_emit_worth_it + CAPTURE)
# Phase C: 战术微调 (resume B, 降 shaping / 低 lr)
#
# Usage:
#   bash scripts/v14d_curriculum.sh
#   nohup bash scripts/v14d_curriculum.sh > logs/v14d_curriculum.log 2>&1 &
#
# Monitor:
#   tail -f logs/v14d_curriculum.log
#   bash scripts/monitor_v14d.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if [ -x /home/charlie/anaconda3/bin/python ]; then
  PY=/home/charlie/anaconda3/bin/python
fi

STATE_FILE="logs/v14d_curriculum.state.json"
mkdir -p logs ckpt_multi_action_v14d_a ckpt_multi_action_v14d_b ckpt_multi_action_v14d_c

log() { echo "[v14d $(date -u +%H:%M:%S)] $*"; }

latest_ckpt() {
  local dir="$1"
  ls -1 "$dir"/ckpt_*.pkl 2>/dev/null | sort | tail -1 || true
}

run_train() {
  local phase="$1"
  local config="$2"
  local logfile="$3"
  local resume="${4:-}"
  local extra_updates="${5:-0}"
  local base_updates
  base_updates=$("$PY" - <<PY
import yaml
with open("$config") as f:
    print(yaml.safe_load(f)["train"]["num_updates"])
PY
)
  local total=$((base_updates + extra_updates))
  local args=(--config "$config" --log-dir "logs/v14d_${phase}" --num-updates "$total")
  if [ -n "$resume" ] && [ -f "$resume" ]; then
    args+=(--resume-from "$resume")
    log "phase ${phase}: resume from $resume (${total} updates)"
  else
    log "phase ${phase}: scratch (${total} updates)"
  fi
  log "phase ${phase}: log -> $logfile"
  "$PY" -m orbit_wars_rl.scripts.train "${args[@]}" 2>&1 | tee -a "$logfile"
}

check_gate() {
  local phase="$1"
  local logfile="$2"
  "$PY" scripts/v14d_gate_check.py "$phase" --log "$logfile" --json-out "$STATE_FILE.gate_${phase}" || true
}

maybe_extend() {
  local phase="$1"
  local config="$2"
  local logfile="$3"
  local ckpt_dir="$4"
  local extend="${5:-400}"
  if "$PY" scripts/v14d_gate_check.py "$phase" --log "$logfile"; then
    return 0
  fi
  local ckpt
  ckpt=$(latest_ckpt "$ckpt_dir")
  if [ -z "$ckpt" ]; then
    log "phase ${phase}: gate FAIL and no ckpt to extend"
    return 1
  fi
  log "phase ${phase}: gate not met — extending +${extend} updates from $ckpt"
  run_train "$phase" "$config" "$logfile" "$ckpt" "$extend"
  "$PY" scripts/v14d_gate_check.py "$phase" --log "$logfile" || true
}

write_state() {
  local phase="$1"
  local status="$2"
  "$PY" - <<PY
import json, time
from pathlib import Path
p = Path("$STATE_FILE")
state = json.loads(p.read_text()) if p.is_file() else {}
state.update({
    "phase": "$phase",
    "status": "$status",
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
})
Path("$STATE_FILE").write_text(json.dumps(state, indent=2))
PY
}

# ── Phase A: hoard ─────────────────────────────────────────────────────────
write_state "a" "running"
export ORBITWARS_SHAPING_HOLD_BONUS=0.04
export ORBITWARS_SHAPING_CAPTURE=0.0
export ORBITWARS_SHAPING_RELEASE=0.03
export ORBITWARS_SHAPING_RELEASE_K=30
export ORBITWARS_SHAPING_EMIT_LOG=0.0
export ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.03
export ORBITWARS_SHAPING_ONE_SHIP_PENALTY=0.02
export ORBITWARS_SHAPING_ONE_SHIP_THRESH=3

LOG_A="logs/v14d_phase_a.log"
: > "$LOG_A"
run_train "a" orbit_wars_rl/configs/multi_action_v14d_phase_a.yaml "$LOG_A"
maybe_extend "a" orbit_wars_rl/configs/multi_action_v14d_phase_a.yaml "$LOG_A" ckpt_multi_action_v14d_a 500
CKPT_A=$(latest_ckpt ckpt_multi_action_v14d_a)
if [ -z "$CKPT_A" ]; then
  log "FATAL: no phase A checkpoint"
  write_state "a" "failed"
  exit 1
fi
check_gate a "$LOG_A"
write_state "a" "done"
log "Phase A complete: $CKPT_A"

# ── Phase B: capture ─────────────────────────────────────────────────────────
write_state "b" "running"
export ORBITWARS_SHAPING_HOLD_BONUS=0.01
export ORBITWARS_SHAPING_CAPTURE=0.10
export ORBITWARS_SHAPING_RELEASE=0.01
export ORBITWARS_SHAPING_RELEASE_K=15
export ORBITWARS_SHAPING_EMIT_LOG=0.0
export ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.05
export ORBITWARS_SHAPING_ONE_SHIP_PENALTY=0.03
export ORBITWARS_SHAPING_ONE_SHIP_THRESH=3

LOG_B="logs/v14d_phase_b.log"
: > "$LOG_B"
run_train "b" orbit_wars_rl/configs/multi_action_v14d_phase_b.yaml "$LOG_B" "$CKPT_A"
maybe_extend "b" orbit_wars_rl/configs/multi_action_v14d_phase_b.yaml "$LOG_B" ckpt_multi_action_v14d_b 600
CKPT_B=$(latest_ckpt ckpt_multi_action_v14d_b)
if [ -z "$CKPT_B" ]; then
  log "FATAL: no phase B checkpoint"
  write_state "b" "failed"
  exit 1
fi
check_gate b "$LOG_B"
write_state "b" "done"
log "Phase B complete: $CKPT_B"

# ── Phase C: refine ─────────────────────────────────────────────────────────
write_state "c" "running"
export ORBITWARS_SHAPING_HOLD_BONUS=0.0
export ORBITWARS_SHAPING_CAPTURE=0.05
export ORBITWARS_SHAPING_RELEASE=0.01
export ORBITWARS_SHAPING_RELEASE_K=15
export ORBITWARS_SHAPING_EMIT_LOG=0.0
export ORBITWARS_SHAPING_PROD_SHARE_DELTA=0.05
export ORBITWARS_SHAPING_ONE_SHIP_PENALTY=0.03
export ORBITWARS_SHAPING_ONE_SHIP_THRESH=3

LOG_C="logs/v14d_phase_c.log"
: > "$LOG_C"
run_train "c" orbit_wars_rl/configs/multi_action_v14d_phase_c.yaml "$LOG_C" "$CKPT_B"
maybe_extend "c" orbit_wars_rl/configs/multi_action_v14d_phase_c.yaml "$LOG_C" ckpt_multi_action_v14d_c 800
CKPT_C=$(latest_ckpt ckpt_multi_action_v14d_c)
check_gate c "$LOG_C"
write_state "c" "done"
log "Phase C complete: ${CKPT_C:-none}"

# ── Final eval ───────────────────────────────────────────────────────────────
if [ -n "${CKPT_C:-}" ] && [ -f "$CKPT_C" ]; then
  log "Running final 4-game eval vs v20 on $CKPT_C"
  PYTHON="$PY" NUM_GAMES=4 bash scripts/quick_replay.sh "$CKPT_C" v14d_final 2>&1 | tail -15 | tee -a logs/v14d_curriculum.log
fi

log "=== v14d curriculum finished ==="
write_state "done" "finished"
