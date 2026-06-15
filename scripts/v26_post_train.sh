#!/usr/bin/env bash
# v26 post-train: h2h20, win HTML, DAY21 progress stub.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHON="${PYTHON:-/home/charlie/anaconda3/bin/python}"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" ORBITWARS_SKIP_PARITY=1

CKPT_DIR="ckpt_multi_action_v26_roi"
mkdir -p logs/replay_analyze logs/replay_html logs/v26_h2h logs/v26_win_replays docs

echo "=== Phase 1: discover ckpts ==="
CKPTS=()
while IFS= read -r line; do
  [ -n "$line" ] && CKPTS+=("$line")
done < <("$PYTHON" - <<'PY'
import glob, re
from pathlib import Path

updates = []
for p in glob.glob("ckpt_multi_action_v26_roi/ckpt_*.pkl"):
    m = re.search(r"(\d{6})", Path(p).name)
    if m:
        updates.append(int(m.group(1)))
for p in glob.glob("ckpt_multi_action_v26_roi/eval/ckpt_eval_u*.pkl"):
    m = re.search(r"u(\d{6})", Path(p).name)
    if m:
        updates.append(int(m.group(1)))
for u in sorted(set(updates))[-8:]:
    print(u)
PY
)

if [ ${#CKPTS[@]} -eq 0 ]; then
  CKPTS=(199 399 799 1199 3999 7999 11999 14199)
fi

echo "ckpt updates: ${CKPTS[*]}"

echo ""
echo "=== Phase 2: quick_replay analyze (10g) ==="
for U in "${CKPTS[@]}"; do
  CKPT="$CKPT_DIR/ckpt_$(printf '%06d' "$U").pkl"
  if [ ! -f "$CKPT" ]; then
    CKPT="$CKPT_DIR/eval/ckpt_eval_u$(printf '%06d' "$U").pkl"
  fi
  [ -f "$CKPT" ] || continue
  TAG="v26_u${U}"
  JSON="logs/replay_analyze/${TAG}_vs_v20.json"
  if [ ! -f "$JSON" ]; then
    echo "[analyze] $TAG ..."
    bash scripts/quick_replay.sh "$CKPT" "$TAG" >/tmp/qr_v26_${U}.log 2>&1 || {
      echo "WARN quick_replay failed $TAG"; tail -5 /tmp/qr_v26_${U}.log
    }
  fi
done

echo ""
echo "=== Phase 3: 20-game h2h for top ckpts ==="
H2H_LOG="logs/v26_h2h/h2h20_summary.txt"
: > "$H2H_LOG"
for U in "${CKPTS[@]: -3}"; do
  CKPT="$CKPT_DIR/ckpt_$(printf '%06d' "$U").pkl"
  [ -f "$CKPT" ] || CKPT="$CKPT_DIR/eval/ckpt_eval_u$(printf '%06d' "$U").pkl"
  [ -f "$CKPT" ] || continue
  TAG="v26_h2h_u${U}"
  echo "[h2h20] u${U} ..."
  NUM_GAMES=20 bash scripts/quick_replay.sh "$CKPT" "$TAG" >> "$H2H_LOG" 2>&1 || true
  cat "logs/replay_analyze/${TAG}_vs_v20.summary.txt" 2>/dev/null >> "$H2H_LOG" || true
done

echo ""
echo "=== Phase 4: win HTML ==="
WINS_FILE="logs/v26_win_replays/win_seeds.txt"
: > "$WINS_FILE"
"$PYTHON" - <<'PY'
import json, glob
from pathlib import Path

wins = []
for p in sorted(glob.glob("logs/replay_analyze/v26_*_vs_v20.json")):
    d = json.loads(Path(p).read_text())
    tag = p.split("/")[-1].replace("_vs_v20.json", "")
    u = tag.replace("v26_u", "").replace("v26_h2h_u", "")
    sb = d.get("seed_base", 0)
    for i, g in enumerate(d.get("per_game", [])):
        if g.get("player_0", {}).get("outcome") == "win":
            wins.append((u, sb + i, tag))
            print(f"  WIN {tag} seed={sb+i}")
Path("logs/v26_win_replays/win_seeds.txt").write_text(
    "\n".join(f"{u} {seed} {tag}" for u, seed, tag in wins)
)
print(f"Total wins: {len(wins)}")
PY

while read -r U SEED TAG; do
  [ -z "$U" ] && continue
  CKPT="$CKPT_DIR/ckpt_$(printf '%06d' "$U").pkl"
  [ -f "$CKPT" ] || CKPT="$CKPT_DIR/eval/ckpt_eval_u$(printf '%06d' "$U").pkl"
  OUT="logs/replay_html/v26_u${U}_s${SEED}"
  [ -f "$OUT/replay.html" ] && continue
  SUB="submission_rl_${TAG}.py"
  [ -f "$SUB" ] || continue
  echo "[html] u${U} seed=${SEED} ..."
  "$PYTHON" -m orbit_wars_rl.scripts.replay_html \
    --agent-a "$SUB" \
    --agent-b submission_v20_0513.py \
    --seed "$SEED" \
    --out-dir "$OUT" || true
done < "$WINS_FILE"

echo ""
echo "=== Phase 5: DAY21 progress doc ==="
DOC="docs/DAY21_PROGRESS.zh.md"
TRAIN_LOG="$(ls -t logs/v26_extend_*/train.log 2>/dev/null | head -1)"
{
  echo "# DAY21 进展 — v26 ROI"
  echo ""
  echo "生成时间: $(date -Iseconds)"
  echo ""
  echo "## 训练"
  echo "- log: \`${TRAIN_LOG:-logs/v26_extend_*/train.log}\`"
  echo "- ckpt_dir: \`ckpt_multi_action_v26_roi/\`"
  echo ""
  echo "## vs-v20 inline eval (末 15 行)"
  echo '```'
  grep '^\[eval_vs_v20\]' "$TRAIN_LOG" 2>/dev/null | grep -v WARN | tail -15 || echo "(no eval lines yet)"
  echo '```'
  echo ""
  echo "## h2h20 摘要"
  echo '```'
  cat "$H2H_LOG" 2>/dev/null || echo "(no h2h yet)"
  echo '```'
  echo ""
  echo "## 验收目标"
  echo "| 指标 | 目标 |"
  echo "|------|------|"
  echo "| 20局 h2h WLD | ≥ 4/16/0 (20%) |"
  echo "| flip + e2+ | flip≥8% 且 e2+≥12% |"
  echo "| vs-random | ≥ 95% |"
} > "$DOC"
echo "wrote $DOC"
echo "=== v26 post-train done ==="
