#!/usr/bin/env bash
# Serial overnight: f29 -> f29_buf -> f29b, each @299/@599 replay + summary table.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -x /home/charlie/anaconda3/bin/python ]; then
  export PYTHON=/home/charlie/anaconda3/bin/python
fi
PY="${PYTHON:-python3}"
JAX_FRAC="${JAX_FRAC:-0.85}"

SUMMARY="logs/overnight_f29_summary.txt"
MAIN_LOG="logs/overnight_f29.log"
mkdir -p logs logs/replay_analyze

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }

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

train_foreground() {
  local cfg="$1"
  local logfile="$2"
  local tb="$3"
  log "train start cfg=$cfg -> $logfile"
  env "${COMMON_SHAPING[@]}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC" \
    "$PY" -m orbit_wars_rl.scripts.train \
      --config "$cfg" \
      --log-dir "$tb" \
    2>&1 | tee "$logfile"
  log "train done cfg=$cfg"
}

replay_ckpt() {
  local ckpt="$1"
  local tag="$2"
  local template="${3:-}"
  if [ ! -f "$ckpt" ]; then
    log "WARN missing $ckpt — skip replay $tag"
    return 0
  fi
  log "replay $tag ($ckpt)"
  if [ -n "$template" ]; then
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
      "$PY" -m orbit_wars_rl.scripts.export_submission \
        --ckpt "$ckpt" --template "$template" --out "submission_rl_${tag}.py"
    "$PY" -m orbit_wars_rl.scripts.replay_analyze \
      --agent-a "submission_rl_${tag}.py" \
      --agent-b submission_v20_0513.py \
      --num-games 5 --seed-base 0 \
      --out "logs/replay_analyze/${tag}_vs_v20.json" \
      > "logs/replay_analyze/${tag}_vs_v20.summary.txt.full" 2>&1 || true
    "$PY" - <<PY > "logs/replay_analyze/${tag}_vs_v20.summary.txt"
import json
with open("logs/replay_analyze/${tag}_vs_v20.json") as f:
    data = json.load(f)
fb = data.get("aggregate_by_window", {}).get("first_80turns", {}).get("player_0", {})
spf = float(fb.get("mean_ships_per_fleet", 0) or 0)
garr = float(fb.get("mean_garrison_my", 0) or 0)
flip = float(fb.get("fleet_arrival_rate", 0) or 0) * 100
z0 = float(fb.get("zero_emit_rate", 0) or 0) * 100
emit_dist = fb.get("emit_count_distribution") or []
e8 = float(emit_dist[8]) * 100 if len(emit_dist) > 8 else 0
pct_dist = fb.get("pct_bin_distribution") or []
bin0 = float(pct_dist[0]) * 100 if pct_dist else 0
oc = fb.get("outcome") or {}
wld = f"{oc.get('win',0)}/{oc.get('loss',0)}/{oc.get('draw',0)}"
print(f"tag=${tag}  bin0={bin0:.1f}%  spf={spf:.2f}  garr={garr:.1f}  flip={flip:.2f}%  e8={e8:.1f}%  z0={z0:.1f}%  WLD={wld}")
PY
  else
    bash scripts/quick_replay.sh "$ckpt" "$tag" || true
  fi
}

run_track() {
  local cfg="$1"
  local ckpt_dir="$2"
  local name="$3"
  local logfile="$4"
  local tb="$5"
  local template="${6:-submission_rl_v11_f29.py}"

  log "=== TRACK $name ==="
  train_foreground "$cfg" "$logfile" "$tb"
  for upd in 299 599; do
    local ckpt="${ckpt_dir}/ckpt_$(printf '%06d' "$upd").pkl"
    local tag="v11_${name}_u$(printf '%03d' "$upd")"
    if [ "$template" = "submission_rl_v11_f29b.py" ]; then
      replay_ckpt "$ckpt" "$tag" "$template"
    else
      replay_ckpt "$ckpt" "$tag" ""
    fi
  done
}

log "parity smoke (fresh-init)"
if ! JAX_PLATFORMS=cpu "$PY" -m orbit_wars_rl.inference.test_parity --num-states 16 | tee -a "$MAIN_LOG"; then
  log "parity FAILED — aborting overnight"
  exit 1
fi

pkill -f 'multi_action_v11_f29' 2>/dev/null || true
sleep 2

run_track orbit_wars_rl/configs/multi_action_v11_f29.yaml \
  ckpt_multi_action_v11_f29 f29 logs/v11_f29.log logs/v11_f29

run_track orbit_wars_rl/configs/multi_action_v11_f29_buf.yaml \
  ckpt_multi_action_v11_f29_buf f29_buf logs/v11_f29_buf.log logs/v11_f29_buf

run_track orbit_wars_rl/configs/multi_action_v11_f29b.yaml \
  ckpt_multi_action_v11_f29b f29b logs/v11_f29b.log logs/v11_f29b \
  submission_rl_v11_f29b.py

log "writing summary -> $SUMMARY"
"$PY" - <<'PY' | tee "$SUMMARY"
import json
from pathlib import Path

tracks = [
    ("f29", "v11_f29_u299", "v11_f29_u599"),
    ("f29_buf", "v11_f29_buf_u299", "v11_f29_buf_u599"),
    ("f29b", "v11_f29b_u299", "v11_f29b_u599"),
]
baselines = [("f27@799", "v11_f27_u799"), ("f28@099", "v11_f28_u099")]

def load_metrics(tag):
    p = Path(f"logs/replay_analyze/{tag}_vs_v20.json")
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    fb = data.get("aggregate_by_window", {}).get("first_80turns", {}).get("player_0", {})
    if not fb:
        return None
    spf = float(fb.get("mean_ships_per_fleet", 0) or 0)
    garr = float(fb.get("mean_garrison_my", 0) or 0)
    flip = float(fb.get("fleet_arrival_rate", 0) or 0) * 100
    z0 = float(fb.get("zero_emit_rate", 0) or 0) * 100
    emit_dist = fb.get("emit_count_distribution") or []
    e8 = float(emit_dist[8]) * 100 if len(emit_dist) > 8 else 0.0
    pct_dist = fb.get("pct_bin_distribution") or []
    bin0 = float(pct_dist[0]) * 100 if pct_dist else 0.0
    oc = fb.get("outcome") or {}
    wld = f"{oc.get('win',0)}/{oc.get('loss',0)}/{oc.get('draw',0)}"
    return dict(tag=tag, spf=spf, garr=garr, flip=flip, z0=z0, e8=e8, bin0=bin0, wld=wld)

rows = []
for name, t299, t599 in tracks:
    for t in (t299, t599):
        m = load_metrics(t)
        if m:
            m["track"] = name
            rows.append(m)

print("f29 overnight replay gate (first-80 vs v20, player_0)")
print(f"{'track':<10} {'tag':<18} {'bin0':>6} {'spf':>6} {'garr':>6} {'flip':>6} {'e8':>6} {'z0':>6} {'WLD':>7}")
print("-" * 78)
for r in rows:
    print(f"{r['track']:<10} {r['tag']:<18} {r['bin0']:5.1f}% {r['spf']:6.2f} {r['garr']:6.1f} "
          f"{r['flip']:5.2f}% {r['e8']:5.1f}% {r['z0']:5.1f}% {r['wld']:>7}")

print("\nBaselines (if present):")
for name, tag in baselines:
    m = load_metrics(tag)
    if m:
        print(f"  {name}: bin0={m['bin0']:.1f}% spf={m['spf']:.2f} flip={m['flip']:.2f}% e8={m['e8']:.1f}%")

print("\nMorning decision hints:")
print("  * f29 (no buffer) bin0 down alone -> signals work; f30 = f29_buf long run")
print("  * only buf/f29b bin0 down -> need buffer / distribution")
print("  * f29b e8 down but flip ~1% -> hard stop fixes spam not combat")
print("  * all bin0 >30% -> need turn-start pct or planet-level signals")
print("  * any flip >3% and spf >8 -> promote that track to 800 upd")
PY

log "overnight complete"
