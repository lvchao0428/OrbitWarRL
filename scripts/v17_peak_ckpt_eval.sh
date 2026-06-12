#!/usr/bin/env bash
# Offline vs-v20 eval + HTML replay for v17 peak flip ckpts (u1399 / u2199).
#
# Usage:
#   bash scripts/v17_peak_ckpt_eval.sh
#   bash scripts/v17_peak_ckpt_eval.sh ckpt_002199.pkl v17_u2199
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-/home/charlie/anaconda3/bin/python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python3
fi
export PYTHON="$PY"

CKPT_DIR="${CKPT_DIR:-./ckpt_multi_action_v17_frog_hist50}"
NUM_GAMES="${NUM_GAMES:-10}"
HTML_SEEDS="${HTML_SEEDS:-0 1}"
EVAL_SEEDS="${EVAL_SEEDS:-0 1 42}"

run_one() {
  local ckpt_path="$1"
  local base_tag="$2"
  local ckpt_name
  ckpt_name="$(basename "$ckpt_path" .pkl)"

  echo ""
  echo "========== $ckpt_name ($base_tag) =========="

  for seed in $EVAL_SEEDS; do
    local tag="${base_tag}_eval_s${seed}"
    echo "[eval] $tag  games=$NUM_GAMES  seed_base=$seed"
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
      NUM_GAMES="$NUM_GAMES" SEED_BASE="$seed" \
      bash scripts/quick_replay.sh "$ckpt_path" "$tag"
  done

  # Reuse submission exported by first eval pass.
  local sub=""
  for seed in $EVAL_SEEDS; do
    local cand="submission_rl_${base_tag}_eval_s${seed}.py"
    if [ -f "$cand" ]; then
      sub="$cand"
      break
    fi
  done
  if [ -z "$sub" ]; then
    echo "ERR: no submission export for $base_tag" >&2
    exit 1
  fi

  for seed in $HTML_SEEDS; do
    local out="logs/replay_html/${base_tag}_vs_v20_seed${seed}"
    mkdir -p "$out"
    echo "[html] seed=$seed -> $out/replay.html"
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
      "$PY" -m orbit_wars_rl.scripts.replay_html \
        --agent-a "$sub" \
        --agent-b submission_v20_0513.py \
        --seed "$seed" \
        --out-dir "$out" \
        --tag replay
  done

  # If any eval JSON has a win, also HTML that seed.
  "$PY" - <<PY
import json, glob, subprocess, os, re
base = "${base_tag}"
sub = "${sub}"
py = "${PY}"
for path in sorted(glob.glob(f"logs/replay_analyze/{base}_eval_s*_vs_v20.json")):
    m = re.search(r"_eval_s(\d+)", path)
    seed_base = int(m.group(1)) if m else 0
    data = json.load(open(path))
    for i, g in enumerate(data.get("per_game") or []):
        if (g.get("player_0") or {}).get("outcome") != "win":
            continue
        seed = seed_base + i
        out = f"logs/replay_html/{base}_vs_v20_seed{seed}_win"
        if os.path.isfile(f"{out}/replay.html"):
            print(f"[html] win replay exists: {out}/replay.html")
            continue
        print(f"[html] WIN seed={seed} -> {out}/replay.html")
        subprocess.run([
            py, "-m", "orbit_wars_rl.scripts.replay_html",
            "--agent-a", sub,
            "--agent-b", "submission_v20_0513.py",
            "--seed", str(seed),
            "--out-dir", out,
            "--tag", "replay",
        ], check=True, env={**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""})
PY
}

# Default: peak flip ckpts from overnight run (restart counter u1399 / u2199).
if [ "$#" -ge 2 ]; then
  run_one "$1" "$2"
elif [ "$#" -eq 0 ]; then
  run_one "$CKPT_DIR/ckpt_001399.pkl" "v17_u1399"
  run_one "$CKPT_DIR/ckpt_002199.pkl" "v17_u2199"
else
  echo "Usage: $0 [ckpt.pkl tag]  OR  $0" >&2
  exit 1
fi

echo ""
echo "========== SUMMARY =========="
for f in logs/replay_analyze/v17_u1399_eval_s*_vs_v20.summary.txt \
         logs/replay_analyze/v17_u2199_eval_s*_vs_v20.summary.txt; do
  [ -f "$f" ] || continue
  echo "--- $(basename "$f") ---"
  head -1 "$f"
done
echo ""
echo "HTML replays:"
find logs/replay_html -path '*v17_u1399*' -name replay.html 2>/dev/null
find logs/replay_html -path '*v17_u2199*' -name replay.html 2>/dev/null
