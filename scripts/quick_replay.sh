#!/usr/bin/env bash
# quick_replay.sh — export a v11 ckpt + run replay vs v20.
#
# Usage:
#   bash scripts/quick_replay.sh <ckpt_path> <tag>
# Example:
#   bash scripts/quick_replay.sh \
#     ckpt_multi_action_v11_g1_scratch/ckpt_000799.pkl \
#     v11_g1_u799
#
# Outputs:
#   submission_rl_<tag>.py
#   logs/replay_analyze/<tag>_vs_v20.json
#   logs/replay_analyze/<tag>_vs_v20.summary.txt   (first-80 metrics one-liner)
#
# Runs entirely on CPU (parity + replay are light); does NOT compete with
# training on the GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CKPT="${1:?Usage: $0 <ckpt> <tag>}"
TAG="${2:?Usage: $0 <ckpt> <tag>}"
NUM_GAMES="${NUM_GAMES:-5}"
SEED_BASE="${SEED_BASE:-0}"
PY="${PYTHON:-python}"

mkdir -p logs/replay_analyze

SUB="submission_rl_${TAG}.py"
JSON="logs/replay_analyze/${TAG}_vs_v20.json"
SUMMARY="logs/replay_analyze/${TAG}_vs_v20.summary.txt"

# Auto-detect template from ckpt arch (planet_feat_dim and K).
ARCH_INFO=$(JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$PY" - <<PY
from orbit_wars_rl.inference.weights import load_flat_params, infer_arch_from_flat
a = infer_arch_from_flat(load_flat_params("$CKPT"))
print(a["planet_feat_dim"], a["max_fleets_per_turn"])
PY
)
PLANET_DIM=$(echo "$ARCH_INFO" | awk '{print $1}')
CKPT_K=$(echo "$ARCH_INFO" | awk '{print $2}')

if [ "$PLANET_DIM" = "19" ]; then
  TEMPLATE="submission_rl_v4.py"
elif [ "$CKPT_K" = "8" ]; then
  TEMPLATE="submission_rl_v11_k8.py"
else
  TEMPLATE="submission_rl_v11.py"
fi

echo "[quick_replay] ckpt=$CKPT  template=$TEMPLATE  out=$SUB  json=$JSON"

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  "$PY" -m orbit_wars_rl.scripts.export_submission \
  --ckpt "$CKPT" \
  --template "$TEMPLATE" \
  --out "$SUB"

# replay_analyze writes JSON + prints first-80 + full-game tables
"$PY" -m orbit_wars_rl.scripts.replay_analyze \
  --agent-a "$SUB" \
  --agent-b submission_v20_0513.py \
  --num-games "$NUM_GAMES" \
  --seed-base "$SEED_BASE" \
  --out "$JSON" 2>&1 | tee "${SUMMARY}.full"

# Extract first-80 numbers from JSON for a one-line gate
"$PY" - <<PY > "$SUMMARY"
import json
with open("$JSON") as f:
    data = json.load(f)

# replay_analyze.py JSON layout (verified from source):
#   data["aggregate_by_window"]["first_80turns"]["player_0"]
fb = data.get("aggregate_by_window", {}).get("first_80turns", {}).get("player_0", {})
spf = float(fb.get("mean_ships_per_fleet", 0.0) or 0.0)
garr = float(fb.get("mean_garrison_my", 0.0) or 0.0)
flip = float(fb.get("fleet_arrival_rate", 0.0) or 0.0) * 100.0
z0 = float(fb.get("zero_emit_rate", 0.0) or 0.0) * 100.0
emit_dist = fb.get("emit_count_distribution") or []  # list, index = n_emits
e2_plus = sum(float(p) for i, p in enumerate(emit_dist) if i >= 2)
oc = fb.get("outcome", {}) or {}
wld = f"{oc.get('win', 0)}/{oc.get('loss', 0)}/{oc.get('draw', 0)}"

print(f"tag=$TAG  spf={spf:.2f}  garr={garr:.2f}  flip={flip:.2f}%  "
      f"z0={z0:.1f}%  e2+={e2_plus*100:.1f}%  WLD={wld}")
print()
print("Day5 first-80 gate (Top10 calibrated):")
print(f"  spf  > 10    actual {spf:.2f}    {'OK' if spf > 10 else 'FAIL'}")
print(f"  garr > 60    actual {garr:.2f}   {'OK' if garr > 60 else 'FAIL'}")
print(f"  flip > 6%    actual {flip:.2f}%  {'OK' if flip > 6 else 'FAIL'}")
print(f"  e2+  > 5%    actual {e2_plus*100:.1f}% {'OK' if e2_plus > 0.05 else 'FAIL'}")
PY

echo ""
echo "================================================================"
cat "$SUMMARY"
echo "================================================================"
