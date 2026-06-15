#!/usr/bin/env bash
# v29 fast pipeline: smoke -> PPO 4000u -> post (no BC).
#
# Usage:
#   bash scripts/v29_pipeline.sh [--updates N] [--skip-post] [--from gateN]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UPDATES=4000
SKIP_POST=0
FROM_GATE="gate0"
PIPE_DIR="logs/v29_pipeline"
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

mkdir -p "$PIPE_DIR"

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
  bash scripts/v29_smoke.sh 2>&1 | tee -a "$LOG_FILE"
  _write_status gate0_smoke 0
}

_gate_ppo_train() {
  _write_status gate3_ppo_train
  echo "[pipeline] Gate3 v29 PPO train ($UPDATES updates)"
  bash scripts/v29_extend.sh "$UPDATES" 2>&1 | tee -a "$LOG_FILE"
  TRAIN_LOG="$(ls -t logs/v29_extend_*/train.log 2>/dev/null | head -1)"
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
  echo "[pipeline] Gate4 post-train h2h/HTML"
  bash scripts/v29_post_train.sh 2>&1 | tee -a "$LOG_FILE"
  _write_status gate4_post_train 0
}

main() {
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo "=== v29 pipeline start $(date -Iseconds) updates=$UPDATES skip_post=$SKIP_POST from=$FROM_GATE ==="
  _write_status pipeline_start

  if _gate_should_run gate0; then _gate0_smoke; fi
  if _gate_should_run gate3; then _gate_ppo_train; fi
  if _gate_should_run gate4; then _gate_post_train; fi

  touch "$PIPE_DIR/status.done"
  _write_status done 0
  echo "=== v29 pipeline DONE $(date -Iseconds) ==="
}

main "$@"
