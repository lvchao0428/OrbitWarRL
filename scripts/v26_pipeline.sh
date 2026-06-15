#!/usr/bin/env bash
# v26 remote pipeline: BC 800g -> epw30 -> PPO 9h -> post h2h/HTML.
#
# Usage:
#   bash scripts/v26_pipeline.sh [--updates N] [--skip-post] [--from gateN]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UPDATES=14200
SKIP_POST=0
FROM_GATE="gate1"
PIPE_DIR="logs/v26_pipeline"
STATUS_FILE="$PIPE_DIR/status.json"
LOG_FILE="$PIPE_DIR/pipeline.log"

RPY="${PYTHON:-/home/charlie/anaconda3/bin/python}"
if ! "$RPY" -c "import jax" >/dev/null 2>&1; then
  RPY="/home/charlie/anaconda3/bin/python"
fi
export PYTHON="$RPY"
export PYTHONPATH="$ROOT"
export JAX_PLATFORMS="${JAX_PLATFORMS:-}"
export ORBITWARS_SKIP_PARITY=1

while [ $# -gt 0 ]; do
  case "$1" in
    --updates) UPDATES="$2"; shift 2 ;;
    --skip-post) SKIP_POST=1; shift ;;
    --from) FROM_GATE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$PIPE_DIR" logs/bc_collect data

_write_status() {
  local stage="$1" exit_code="${2:-}" failed="${3:-}"
  local err_tail=""
  if [ -n "$failed" ] && [ -f "$LOG_FILE" ]; then
    err_tail="$(tail -20 "$LOG_FILE" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
  fi
  "$RPY" - "$STATUS_FILE" "$stage" "$exit_code" "$failed" "$err_tail" <<'PY'
import json, sys, datetime
path, stage, exit_code, failed, err_tail = sys.argv[1:6]
try:
    data = json.loads(open(path).read()) if __import__("os").path.isfile(path) else {}
except Exception:
    data = {}
data["stage"] = stage
data["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
if exit_code != "":
    data["exit_code"] = int(exit_code) if exit_code.lstrip("-").isdigit() else exit_code
if failed:
    data["failed_stage"] = failed
    data["error_tail"] = json.loads(err_tail) if err_tail else ""
gates = data.setdefault("gates", {})
if stage.startswith("gate"):
    gates[stage] = "running" if exit_code == "" else ("pass" if exit_code == "0" else "fail")
open(path, "w").write(json.dumps(data, indent=2))
PY
}

_gate_rank() {
  case "$1" in
    gate0) echo 0 ;;
    gate1) echo 1 ;;
    gate2) echo 2 ;;
    gate3) echo 3 ;;
    gate4) echo 4 ;;
    *) echo 0 ;;
  esac
}

_gate_should_run() {
  local gate="$1"
  if [ -z "$FROM_GATE" ]; then
    return 0
  fi
  [ "$(_gate_rank "$gate")" -ge "$(_gate_rank "$FROM_GATE")" ]
}

_gate0_smoke() {
  _write_status gate0_smoke
  echo "[pipeline] Gate0 smoke"
  bash scripts/v26_smoke.sh 2>&1 | tee -a "$LOG_FILE"
  _write_status gate0_smoke 0
}

_gate_bc_collect() {
  _write_status gate1_bc_collect
  echo "[pipeline] Gate1 BC collect 8x100 -> bc_v20_self_800g.npz"
  bash scripts/bc_collect_parallel.sh 8 100 data/bc_v20_self_800g.npz 2>&1 | tee -a "$LOG_FILE"
  "$RPY" - <<'PY' || { _write_status gate1_bc_collect 1 gate1_bc_collect; exit 1; }
import numpy as np
d = np.load("data/bc_v20_self_800g.npz")
n = d["src"].shape[0]
emits = d["emit"].sum(axis=1)
hold = (emits == 0).mean() * 100
e2p = (emits >= 2).mean() * 100
pf = d["planet_feats"].shape
gf = d["global_feats"].shape
print(f"samples={n} hold={hold:.1f}% e2+={e2p:.1f}% planet={pf} global={gf}")
assert n >= 200_000, f"too few samples: {n}"
assert pf[-1] == 63, pf
assert gf[-1] == 427, gf
assert 50 <= hold <= 70, hold
assert 15 <= e2p <= 30, e2p
print("Gate1 BC collect PASS")
PY
  _write_status gate1_bc_collect 0
}

_gate_bc_train() {
  _write_status gate2_bc_train
  echo "[pipeline] Gate2 BC train epw3.0"
  mkdir -p ckpt_bc_v20_epw30
  "$RPY" -m orbit_wars_rl.bc.train_bc \
    --data data/bc_v20_self_800g.npz \
    --epochs 10 --batch-size 256 --lr 3e-4 \
    --emit-pos-weight 3.0 \
    --out ckpt_bc_v20_epw30/ckpt_final.pkl \
    2>&1 | tee -a "$LOG_FILE"
  test -f ckpt_bc_v20_epw30/ckpt_final.pkl || { _write_status gate2_bc_train 1 gate2_bc_train; exit 1; }
  test -f ckpt_bc_v20_epw30/ckpt_final.meta.json || true
  echo "[pipeline] Gate2 quick_replay 5g sanity"
  NUM_GAMES=5 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    bash scripts/quick_replay.sh ckpt_bc_v20_epw30/ckpt_final.pkl v26_bc_epw30_gate \
    2>&1 | tee -a "$LOG_FILE"
  SUM="logs/replay_analyze/v26_bc_epw30_gate_vs_v20.summary.txt"
  if [ -f "$SUM" ]; then
    grep -qE 'flip|[0-9.]+%' "$SUM" || true
    cat "$SUM" | head -3
  fi
  _write_status gate2_bc_train 0
}

_gate_ppo_train() {
  _write_status gate3_ppo_train
  echo "[pipeline] Gate3 v26 PPO train ($UPDATES updates)"
  bash scripts/v26_extend.sh "$UPDATES" 2>&1 | tee -a "$LOG_FILE"
  TRAIN_LOG="$(ls -t logs/v26_extend_*/train.log 2>/dev/null | head -1)"
  grep -q '\[eval_vs_v20\] u199' "$TRAIN_LOG" || {
    echo "WARN: u199 eval line not found in $TRAIN_LOG"
  }
  _write_status gate3_ppo_train 0 "" "$TRAIN_LOG"
}

_gate_post_train() {
  if [ "$SKIP_POST" -eq 1 ]; then
    echo "[pipeline] Gate4 skipped (--skip-post)"
    _write_status gate4_post_train 0
    return 0
  fi
  _write_status gate4_post_train
  echo "[pipeline] Gate4 post-train h2h/HTML/DAY21"
  bash scripts/v26_post_train.sh 2>&1 | tee -a "$LOG_FILE"
  _write_status gate4_post_train 0
}

main() {
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo "=== v26 pipeline start $(date -Iseconds) updates=$UPDATES skip_post=$SKIP_POST from=$FROM_GATE ==="
  _write_status pipeline_start

  if _gate_should_run gate0; then _gate0_smoke; fi
  if _gate_should_run gate1; then _gate_bc_collect; fi
  if _gate_should_run gate2; then _gate_bc_train; fi
  if _gate_should_run gate3; then _gate_ppo_train; fi
  if _gate_should_run gate4; then _gate_post_train; fi

  touch "$PIPE_DIR/status.done"
  _write_status done 0
  echo "=== v26 pipeline DONE $(date -Iseconds) ==="
}

main "$@"
