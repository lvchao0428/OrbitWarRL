#!/usr/bin/env bash
# Phase C 长训 — 在 A/B 二分 confirm 通过后启动
#
# 读取 logs/v14d_search.state.json 中 pipeline.b 的 ckpt + 最优 shaping
# Usage:
#   bash scripts/v14d_phase_c_longtrain.sh
#   nohup bash scripts/v14d_phase_c_longtrain.sh >> logs/v14d_c_long.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if [ -x /home/charlie/anaconda3/bin/python ]; then
  PY=/home/charlie/anaconda3/bin/python
fi

STATE="${STATE:-logs/v14d_search.state.json}"
SPACE="${SPACE:-scripts/v14d_search_space.yaml}"
READY="logs/v14d_c_longtrain.ready.json"

read -r RESUME_CKPT NUM_UPDATES EXTEND MAX_EXT LOGFILE CKPT_DIR <<EOF
$("$PY" - <<PY
import json, yaml
from pathlib import Path

state = json.loads(Path("$STATE").read_text())
space = yaml.safe_load(open("$SPACE"))
lt = space.get("phase_c_longtrain", {})
b = state.get("pipeline", {}).get("b")
if not b or not b.get("ckpt"):
    raise SystemExit("pipeline.b missing — run A/B search confirm first")

Path("$READY").write_text(json.dumps({
    "resume_ckpt": b["ckpt"],
    "b_trial": b.get("trial_id"),
    "b_params": {**b.get("env_params", {}), **b.get("ppo_params", {})},
}, indent=2))

print(b["ckpt"])
print(lt.get("num_updates", 10000))
print(lt.get("extend_updates", 2000))
print(lt.get("max_extends", 3))
print(lt.get("log", "logs/v14d_phase_c_long.log"))
print(lt.get("ckpt_dir", "ckpt_multi_action_v14d_c_long"))
PY
)
EOF

echo "[c_long] resume from $RESUME_CKPT"
echo "[c_long] updates=$NUM_UPDATES extend=$EXTEND x$MAX_EXT"
echo "[c_long] ready file -> $READY"

# Export shaping: longtrain fixed + inherit key params from B confirm
while IFS='=' read -r k v; do
  export "$k=$v"
done < <("$PY" - <<PY
import json, yaml
from pathlib import Path
state = json.loads(Path("$STATE").read_text())
space = yaml.safe_load(open("$SPACE"))
lt = space["phase_c_longtrain"]
fixed = dict(lt.get("fixed", {}))
b = state["pipeline"]["b"]
for k in ("ORBITWARS_SHAPING_CAPTURE", "ORBITWARS_SHAPING_RELEASE",
          "ORBITWARS_SHAPING_RELEASE_K", "ORBITWARS_SHAPING_PROD_SHARE_DELTA"):
    if k in b.get("env_params", {}):
        fixed[k] = b["env_params"][k]
for k, v in fixed.items():
    print(f"{k}={v}")
PY
)

export ORBITWARS_SHAPING_SCALE=0.0
CONFIG="orbit_wars_rl/configs/multi_action_v14d_phase_c_long.yaml"
mkdir -p logs "$CKPT_DIR"

run_train() {
  local resume="$1"
  local updates="$2"
  local args=(
    -m orbit_wars_rl.scripts.train
    --config "$CONFIG"
    --log-dir logs/v14d_phase_c_long_tb
    --num-updates "$updates"
    --ckpt-dir "./$CKPT_DIR"
  )
  if [[ -n "$resume" ]]; then
    args+=(--resume-from "$resume")
  fi
  "$PY" "${args[@]}" 2>&1 | tee -a "$LOGFILE"
}

: > "$LOGFILE"
run_train "$RESUME_CKPT" "$NUM_UPDATES"

extends=0
while [[ $extends -lt $MAX_EXT ]]; do
  if "$PY" scripts/v14d_gate_check.py c --log "$LOGFILE"; then
    echo "[c_long] gate C PASS"
    break
  fi
  latest=$(ls -1 "$CKPT_DIR"/ckpt_*.pkl 2>/dev/null | sort | tail -1 || true)
  if [[ -z "$latest" ]]; then
    echo "[c_long] no ckpt to extend"
    exit 1
  fi
  extends=$((extends + 1))
  echo "[c_long] extend $extends/$MAX_EXT +$EXTEND from $latest"
  run_train "$latest" "$EXTEND"
done

echo "[c_long] final eval vs v20"
PYTHON="$PY" NUM_GAMES=4 bash scripts/quick_replay.sh \
  "$(ls -1 "$CKPT_DIR"/ckpt_*.pkl | sort | tail -1)" v14d_c_long 2>&1 | tail -15 | tee -a "$LOGFILE"

echo "[c_long] done"
