#!/usr/bin/env bash
# v25: find all inline-eval win seeds per ckpt, export HTML replays, 20-game h2h.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHON="${PYTHON:-/home/charlie/anaconda3/bin/python}"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" ORBITWARS_SKIP_PARITY=1

# Nearest saved ckpt for each eval point that reported a win (10-game inline eval).
CKPTS=(399 4399 5599 6399 6799 7999 8399 8799 9599 9999 10799)

mkdir -p logs/replay_analyze logs/replay_html logs/v25_h2h logs/v25_win_replays

echo "=== Phase 1: replay_analyze (10 games, seed 0-9) ==="
for U in "${CKPTS[@]}"; do
  CKPT="ckpt_multi_action_v25_extend/ckpt_$(printf '%06d' "$U").pkl"
  TAG="v25_u${U}"
  JSON="logs/replay_analyze/${TAG}_vs_v20.json"
  if [ ! -f "$JSON" ]; then
    echo "[analyze] $TAG ..."
    bash scripts/quick_replay.sh "$CKPT" "$TAG" >/tmp/qr_v25_${U}.log 2>&1 || {
      echo "WARN quick_replay failed $TAG"; cat /tmp/qr_v25_${U}.log | tail -5
    }
  else
    echo "[analyze] skip existing $JSON"
  fi
done

echo ""
echo "=== Phase 2: HTML replay for every winning seed ==="
WINS_FILE="logs/v25_win_replays/win_seeds.txt"
: > "$WINS_FILE"

"$PYTHON" - <<'PY'
import json
from pathlib import Path

ckpts = [399, 4399, 5599, 6399, 6799, 7999, 8399, 8799, 9599, 9999, 10799]
wins = []
for u in ckpts:
    tag = f"v25_u{u}"
    p = Path(f"logs/replay_analyze/{tag}_vs_v20.json")
    if not p.exists():
        print(f"MISSING {p}")
        continue
    d = json.loads(p.read_text())
    sb = d["seed_base"]
    for i, g in enumerate(d["per_game"]):
        if g["player_0"].get("outcome") == "win":
            seed = sb + i
            wins.append((u, seed))
            print(f"  WIN ckpt_u{u} seed={seed} (game{i})")

out = Path("logs/v25_win_replays/win_seeds.txt")
with out.open("w") as f:
    for u, seed in wins:
        f.write(f"{u} {seed}\n")
print(f"\nTotal wins: {len(wins)} unique pairs -> {out}")
PY

while read -r U SEED; do
  [ -z "$U" ] && continue
  CKPT="ckpt_multi_action_v25_extend/ckpt_$(printf '%06d' "$U").pkl"
  TAG="v25_u${U}_s${SEED}"
  SUB="submission_rl_v25_u${U}.py"
  OUT="logs/replay_html/${TAG}"
  if [ -f "$OUT/replay.html" ]; then
    echo "[html] skip $TAG"
    continue
  fi
  echo "[html] $TAG ..."
  "$PYTHON" -m orbit_wars_rl.scripts.replay_html \
    --agent-a "$SUB" \
    --agent-b submission_v20_0513.py \
    --seed "$SEED" \
    --out-dir "$OUT"
done < "$WINS_FILE"

echo ""
echo "=== Phase 3: 20-game h2h (formal acceptance) ==="
H2H_CKPTS=(10799 9599 8399 4399 9999)
for U in "${H2H_CKPTS[@]}"; do
  CKPT="ckpt_multi_action_v25_extend/ckpt_$(printf '%06d' "$U").pkl"
  TAG="v25_u${U}_h2h20"
  echo "[h2h20] $TAG ..."
  NUM_GAMES=20 SEED_BASE=0 \
    bash scripts/quick_replay.sh "$CKPT" "$TAG" \
    2>&1 | tee "logs/v25_h2h/${TAG}.log" | tail -12
done

echo ""
echo "=== Done ==="
echo "HTML: logs/replay_html/v25_u*_s*/replay.html"
echo "H2H:  logs/replay_analyze/v25_u*_h2h20_vs_v20.summary.txt"
