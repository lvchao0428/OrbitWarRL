#!/usr/bin/env bash
# Full replay eval for f44 (optional; training already runs mini v20 every 50 upd).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -z "${PYTHON:-}" ] && [ -x /home/charlie/anaconda3/bin/python ]; then
  export PYTHON=/home/charlie/anaconda3/bin/python
fi
for U in 49 99 149 199 249 299 349 399 449 499; do
  CKPT="ckpt_multi_action_v11_f44_align/ckpt_$(printf '%06d' "$U").pkl"
  [ -f "$CKPT" ] || continue
  bash scripts/quick_replay.sh "$CKPT" "v11_f44_u${U}"
done
echo "=== prefer ckpts with meta opp_tag=strn/frzn and best eval_vs_v20 in log ==="
for U in 49 99 149 199 249 299 349 399 449 499; do
  META="ckpt_multi_action_v11_f44_align/ckpt_$(printf '%06d' "$U").meta.json"
  SUM="logs/replay_analyze/v11_f44_u${U}_vs_v20.summary.txt"
  [ -f "$META" ] && echo -n "u${U} meta: " && head -3 "$META"
  [ -f "$SUM" ] && head -1 "$SUM"
done
