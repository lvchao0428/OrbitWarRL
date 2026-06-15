#!/usr/bin/env bash
# v28 post-train: h2h20, win HTML, progress doc.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHON="${PYTHON:-/home/charlie/anaconda3/bin/python}"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" ORBITWARS_SKIP_PARITY=1

CKPT_DIR="ckpt_multi_action_v28_roi"
mkdir -p logs/replay_analyze logs/replay_html logs/v28_h2h logs/v28_win_replays docs

echo "=== Phase 1: discover ckpts ==="
CKPTS=()
while IFS= read -r line; do
  [ -n "$line" ] && CKPTS+=("$line")
done < <("$PYTHON" - <<'PY'
import glob, re
from pathlib import Path

updates = []
for p in glob.glob("ckpt_multi_action_v28_roi/ckpt_*.pkl"):
    m = re.search(r"(\d{6})", Path(p).name)
    if m:
        updates.append(int(m.group(1)))
for p in glob.glob("ckpt_multi_action_v28_roi/eval/ckpt_eval_u*.pkl"):
    m = re.search(r"u(\d{6})", Path(p).name)
    if m:
        updates.append(int(m.group(1)))
for u in sorted(set(updates))[-8:]:
    print(u)
PY
)

if [ ${#CKPTS[@]} -eq 0 ]; then
  CKPTS=(199 399 799 1199 2399 3999)
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
  TAG="v28_u${U}"
  JSON="logs/replay_analyze/${TAG}_vs_v20.json"
  if [ ! -f "$JSON" ]; then
    echo "[analyze] $TAG ..."
    bash scripts/quick_replay.sh "$CKPT" "$TAG" >/tmp/qr_v28_${U}.log 2>&1 || {
      echo "WARN quick_replay failed $TAG"; tail -5 /tmp/qr_v28_${U}.log
    }
  fi
done

echo ""
echo "=== Phase 3: 20-game h2h for top ckpts ==="
H2H_LOG="logs/v28_h2h/h2h20_summary.txt"
: > "$H2H_LOG"
for U in "${CKPTS[@]: -3}"; do
  CKPT="$CKPT_DIR/ckpt_$(printf '%06d' "$U").pkl"
  [ -f "$CKPT" ] || CKPT="$CKPT_DIR/eval/ckpt_eval_u$(printf '%06d' "$U").pkl"
  [ -f "$CKPT" ] || continue
  TAG="v28_h2h_u${U}"
  echo "[h2h20] u${U} ..."
  NUM_GAMES=20 bash scripts/quick_replay.sh "$CKPT" "$TAG" >> "$H2H_LOG" 2>&1 || true
  cat "logs/replay_analyze/${TAG}_vs_v20.summary.txt" 2>/dev/null >> "$H2H_LOG" || true
done

echo ""
echo "=== Phase 4: seed=0 replay (u3999 calibration check) ==="
U3999=3999
CKPT3999="$CKPT_DIR/ckpt_$(printf '%06d' "$U3999").pkl"
[ -f "$CKPT3999" ] || CKPT3999="$CKPT_DIR/eval/ckpt_eval_u$(printf '%06d' "$U3999").pkl"
if [ -f "$CKPT3999" ]; then
  TAG="v28_u3999"
  SUB="submission_rl_${TAG}.py"
  if [ ! -f "$SUB" ]; then
    bash scripts/quick_replay.sh "$CKPT3999" "$TAG" >/tmp/qr_v28_seed.log 2>&1 || true
  fi
  OUT="logs/replay_html/v28_u3999_s0"
  if [ -f "$SUB" ] && [ ! -f "$OUT/replay.html" ]; then
    "$PYTHON" -m orbit_wars_rl.scripts.replay_html \
      --agent-a "$SUB" \
      --agent-b submission_v20_0513.py \
      --seed 0 \
      --out-dir "$OUT" || true
  fi
fi

echo ""
echo "=== Phase 5: progress doc ==="
DOC="docs/DAY22_PROGRESS.zh.md"
TRAIN_LOG="$(ls -t logs/v28_extend_*/train.log 2>/dev/null | head -1)"
{
  echo "# DAY22 进展 — v28 payback ROI"
  echo ""
  echo "生成时间: $(date -Iseconds)"
  echo ""
  echo "## 训练"
  echo "- log: \`${TRAIN_LOG:-logs/v28_extend_*/train.log}\`"
  echo "- ckpt_dir: \`ckpt_multi_action_v28_roi/\`"
  echo "- init: v25 u9599; ROI: prod/(payback+eta); calib seed u3999 id=20"
  echo ""
  echo "## vs-v20 inline eval"
  echo '```'
  grep '^\[eval_vs_v20\]' "$TRAIN_LOG" 2>/dev/null | grep -v WARN | tail -15 || echo "(no eval lines yet)"
  echo '```'
  echo ""
  echo "## h2h20"
  echo '```'
  cat "$H2H_LOG" 2>/dev/null || echo "(no h2h yet)"
  echo '```'
} > "$DOC"
echo "wrote $DOC"
echo "=== v28 post-train done ==="
