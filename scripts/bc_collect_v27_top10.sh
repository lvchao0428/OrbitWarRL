#!/usr/bin/env bash
# Collect top10 winner states for v27 PPO buffer curriculum.
#
# Usage:
#   bash scripts/bc_collect_v27_top10.sh [replay_dir] [out_npz]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REPLAY_DIR="${1:-_archive/top10_episodes_2026-05-04/episodes/episodes}"
OUT="${2:-data/top10_winner_states.npz}"

PY="${PYTHON:-python3}"
if ! "$PY" -c "import numpy" >/dev/null 2>&1; then
  PY="/home/charlie/anaconda3/bin/python"
fi

mkdir -p data logs/bc_collect

if [ ! -d "$REPLAY_DIR" ]; then
  echo "[top10] WARN replay dir missing: $REPLAY_DIR"
  echo "[top10] skip — create empty placeholder or sync _archive from remote"
  exit 0
fi

echo "[top10] collect winners from $REPLAY_DIR -> $OUT"
PYTHONPATH="$ROOT" JAX_PLATFORMS=cpu "$PY" -m orbit_wars_rl.bc.collect_states_from_json \
  --replay-dir "$REPLAY_DIR" \
  --winners-only \
  --out "$OUT" \
  2>&1 | tee "logs/bc_collect/top10_winner_$(date +%Y%m%d_%H%M%S).log"

test -f "$OUT" || { echo "[top10] FAIL no output"; exit 1; }
"$PY" - <<PY
import numpy as np
d = np.load("$OUT")
n = d["step"].shape[0]
print(f"[top10] states={n} planet={d['planet_feats'].shape} global={d['global_feats'].shape}")
assert n >= 1000, f"too few top10 states: {n}"
assert d["planet_feats"].shape[-1] == 63
print("[top10] PASS")
PY
