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
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python
fi

mkdir -p logs/replay_analyze

SUB="submission_rl_${TAG}.py"
JSON="logs/replay_analyze/${TAG}_vs_v20.json"
SUMMARY="logs/replay_analyze/${TAG}_vs_v20.summary.txt"

# Auto-detect template from ckpt arch (planet_feat_dim, K, has_pair, f29 marker).
ARCH_INFO=$(JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$PY" - <<PY
from orbit_wars_rl.inference.weights import load_flat_params, infer_arch_from_flat
a = infer_arch_from_flat(load_flat_params("$CKPT"))
print(a["planet_feat_dim"], a["max_fleets_per_turn"], int(a.get("has_pair", False)), int(a.get("dst_pair_dim", 0)), a.get("global_feat_dim", 17), a.get("emit_pair_dim", 0))
PY
)
PLANET_DIM=$(echo "$ARCH_INFO" | awk '{print $1}')
CKPT_K=$(echo "$ARCH_INFO" | awk '{print $2}')
HAS_PAIR=$(echo "$ARCH_INFO" | awk '{print $3}')
DST_PAIR_DIM=$(echo "$ARCH_INFO" | awk '{print $4}')
GLOBAL_DIM=$(echo "$ARCH_INFO" | awk '{print $5}')
EMIT_PAIR_DIM=$(echo "$ARCH_INFO" | awk '{print $6}')

if [ "$PLANET_DIM" = "19" ]; then
  TEMPLATE="submission_rl_v4.py"
elif [ "$PLANET_DIM" = "63" ] && [ "$HAS_PAIR" = "1" ] && [ "$GLOBAL_DIM" = "427" ]; then
  if [ "$EMIT_PAIR_DIM" = "12" ] || [ "$DST_PAIR_DIM" = "7" ]; then
    TEMPLATE="submission_rl_v30c.py"
  else
    TEMPLATE="submission_rl_v21.py"
  fi
elif [ "$PLANET_DIM" = "41" ] && [ "$HAS_PAIR" = "1" ] && [ "$GLOBAL_DIM" = "427" ]; then
  TEMPLATE="submission_rl_v17.py"
elif [ "$PLANET_DIM" = "39" ] && [ "$HAS_PAIR" = "1" ] && [ "$GLOBAL_DIM" = "427" ]; then
  TEMPLATE="submission_rl_v17.py"
elif [ "$PLANET_DIM" = "39" ] && [ "$HAS_PAIR" = "1" ] && [ "$GLOBAL_DIM" = "27" ]; then
  TEMPLATE="submission_rl_v15.py"
elif [ "$PLANET_DIM" = "39" ] && [ "$HAS_PAIR" = "1" ] && [ "$GLOBAL_DIM" = "24" ]; then
  TEMPLATE="submission_rl_v13c.py"
elif [ "$PLANET_DIM" = "33" ] && [ "$HAS_PAIR" = "1" ] && [ "$GLOBAL_DIM" = "18" ]; then
  if echo "$TAG" | grep -qE 'f38|f40'; then
    TEMPLATE="submission_rl_v11_f38.py"
  else
    TEMPLATE="submission_rl_v11_f37.py"
  fi
elif [ "$PLANET_DIM" = "33" ] && [ "$HAS_PAIR" = "1" ]; then
  # f35: f33 arch + 5 src-quality / v20-target-score planet feats (dst_pair=5).
  TEMPLATE="submission_rl_v11_f35.py"
elif [ "$PLANET_DIM" = "28" ] && [ "$HAS_PAIR" = "1" ]; then
  if [ "$DST_PAIR_DIM" = "7" ]; then
    TEMPLATE="submission_rl_v11_f32.py"
  elif [ "$DST_PAIR_DIM" = "5" ]; then
    if [ -f submission_rl_v11_f33.py ] && echo "$TAG" | grep -qE 'f33|f34|f36a|f36b'; then
      TEMPLATE="submission_rl_v11_f33.py"
    elif [ -f submission_rl_v11_f31.py ] && echo "$TAG" | grep -q 'f31'; then
      TEMPLATE="submission_rl_v11_f31.py"
    elif grep -q '^EMIT_HARD_STOP\s*=\s*1' orbit_wars_rl/configs/multi_action_v11_f29b.yaml 2>/dev/null \
       && echo "$CKPT" | grep -q 'f29b'; then
      TEMPLATE="submission_rl_v11_f29b.py"
    elif [ -f submission_rl_v11_f29b.py ] && echo "$TAG" | grep -q 'f29b'; then
      TEMPLATE="submission_rl_v11_f29b.py"
    else
      TEMPLATE="submission_rl_v11_f29.py"
    fi
  elif [ "$DST_PAIR_DIM" = "4" ]; then
    TEMPLATE="submission_rl_v11_f26.py"
  elif grep -q '^MIN_BIN_PCT_FEAT\s*=' submission_rl_v11_f27.py 2>/dev/null; then
    TEMPLATE="submission_rl_v11_f27.py"
  else
    TEMPLATE="submission_rl_v11_f26.py"
  fi
elif [ "$PLANET_DIM" = "28" ]; then
  TEMPLATE="submission_rl_v11_f25.py"
elif [ "$CKPT_K" = "8" ]; then
  TEMPLATE="submission_rl_v11_k8.py"
else
  TEMPLATE="submission_rl_v11.py"
fi

echo "[quick_replay] ckpt=$CKPT  template=$TEMPLATE  out=$SUB  json=$JSON"

EXPORT_ARGS=()
if [ -n "${ORBITWARS_SKIP_PARITY:-}" ]; then
  EXPORT_ARGS+=(--skip-parity)
fi
if [ -n "${EMIT_HARD_STOP_MIN_STEP:-}" ]; then
  EXPORT_ARGS+=(--emit-hard-stop-min-step "$EMIT_HARD_STOP_MIN_STEP")
fi
if echo "$TAG" | grep -q 'f40_bc'; then
  EXPORT_ARGS+=(--emit-hard-stop 0 --flip-hard-mask 0)
fi

# Export flags from ckpt sidecar (.meta.json) or env overrides.
META="${CKPT%.pkl}.meta.json"
if [ -f "$META" ] || [ -n "${EXPORT_ALLOW_HOLD:-}" ]; then
  EXPORT_FLAGS=$("$PY" - <<PY
import json, os
meta = {}
mp = "$META"
if os.path.isfile(mp):
    with open(mp) as f:
        meta = json.load(f)
exp = meta.get("export") or {}
def _flag(name, env_key, cast=int):
    env = os.environ.get(env_key)
    if env is not None and env != "":
        return cast(env)
    v = exp.get(name)
    return cast(v) if v is not None else None
allow = _flag("allow_hold", "EXPORT_ALLOW_HOLD", int)
worth = _flag("force_emit_worth_it", "EXPORT_FORCE_EMIT_WORTH_IT", int)
minb = _flag("min_pct_bin", "EXPORT_MIN_PCT_BIN", int)
emit = _flag("emit_hard_stop", "EXPORT_EMIT_HARD_STOP", int)
flip = _flag("flip_hard_mask", "EXPORT_FLIP_HARD_MASK", int)
emit_min = _flag("emit_hard_stop_min_step", "EXPORT_EMIT_HARD_STOP_MIN_STEP", int)
parts = []
if allow is not None: parts.append(f"--allow-hold {allow}")
if worth is not None: parts.append(f"--force-emit-worth-it {worth}")
if minb is not None: parts.append(f"--min-pct-bin {minb}")
if emit is not None: parts.append(f"--emit-hard-stop {emit}")
if flip is not None: parts.append(f"--flip-hard-mask {flip}")
if emit_min is not None: parts.append(f"--emit-hard-stop-min-step {emit_min}")
print(" ".join(parts))
PY
)
  if [ -n "$EXPORT_FLAGS" ]; then
    # shellcheck disable=SC2206
    EXPORT_ARGS+=($EXPORT_FLAGS)
    echo "[quick_replay] export flags: $EXPORT_FLAGS"
  fi
fi

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  "$PY" -m orbit_wars_rl.scripts.export_submission \
  --ckpt "$CKPT" \
  --template "$TEMPLATE" \
  --out "$SUB" \
  ${EXPORT_ARGS+"${EXPORT_ARGS[@]}"}

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
e8 = float(emit_dist[8]) * 100.0 if len(emit_dist) > 8 else 0.0
bin_dist = fb.get("pct_bin_distribution") or []
bin0 = float(bin_dist[0]) * 100.0 if bin_dist else 0.0
oc = fb.get("outcome", {}) or {}
wld = f"{oc.get('win', 0)}/{oc.get('loss', 0)}/{oc.get('draw', 0)}"

print(f"tag=$TAG  bin0={bin0:.1f}%  spf={spf:.2f}  garr={garr:.2f}  flip={flip:.2f}%  "
      f"z0={z0:.1f}%  e8={e8:.1f}%  e2+={e2_plus*100:.1f}%  WLD={wld}")
print()
print("Day5 first-80 gate (Top10 calibrated):")
print(f"  spf  > 10    actual {spf:.2f}    {'OK' if spf > 10 else 'FAIL'}")
print(f"  garr > 60    actual {garr:.2f}   {'OK' if garr > 60 else 'FAIL'}")
print(f"  flip > 6%    actual {flip:.2f}%  {'OK' if flip > 6 else 'FAIL'}")
print(f"  e2+  > 5%    actual {e2_plus*100:.1f}% {'OK' if e2_plus > 0.05 else 'FAIL'}")

# f31 anti-spam metrics from per_game traces (first 80 turns)
per_game = data.get("per_game") or []
game_spfs = []
all_launches = []
home_le1_emits = 0
total_emits = 0
for g in per_game:
    tr = g.get("player_0") or {}
    spf_turns = [
        v for i, v in enumerate(tr.get("ships_per_fleet") or [])
        if v is not None and i < 80
    ]
    if spf_turns:
        game_spfs.append(sum(spf_turns) / len(spf_turns))
    all_launches.extend(
        s for s, turn in zip(tr.get("launch_ships") or [], tr.get("launch_turns") or [])
        if turn < 80
    )
    emit_turns = tr.get("emit_turns") or []
    emit_hg = tr.get("emit_home_garrison") or []
    for turn_idx, hg in zip(emit_turns, emit_hg):
        if turn_idx >= 80:
            continue
        total_emits += 1
        if hg <= 1:
            home_le1_emits += 1

min_game_spf = min(game_spfs) if game_spfs else 0.0
one_ship_rate = (
    100.0 * sum(1 for s in all_launches if s == 1) / len(all_launches)
    if all_launches else 0.0
)
home_le1_emit_rate = (
    100.0 * home_le1_emits / total_emits if total_emits else 0.0
)

print()
print("f31/f32 gate (aggregate first-80):")
print(f"  bin0 < 15%     actual {bin0:.1f}%   {'OK' if bin0 < 15 else 'FAIL'}")
print(f"  e8   < 5%      actual {e8:.1f}%    {'OK' if e8 < 5 else 'FAIL'}")
print(f"  z0   > 8%      actual {z0:.1f}%    {'OK' if z0 > 8 else 'FAIL'}  (f32 target)")
print(f"  z0   > 10%     actual {z0:.1f}%    {'OK' if z0 > 10 else 'FAIL'}  (f31 target)")
print()
print("f31 anti-spam metrics (player_0, first 80 / all launches):")
print(f"  min_game_spf       = {min_game_spf:.2f}   (gate > 5)")
print(f"  one_ship_rate      = {one_ship_rate:.1f}%  (gate < 50%)")
print(f"  home_le1_emit_rate = {home_le1_emit_rate:.1f}%  (lower is better)")
PY

echo ""
echo "================================================================"
cat "$SUMMARY"
echo "================================================================"
