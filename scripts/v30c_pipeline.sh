#!/usr/bin/env bash
# v30c pipeline: smoke -> PPO 800u -> post (no BC warmstart)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
UPDATES=800
SKIP_POST=0
PIPE_DIR="logs/v30c_pipeline"
LOG_FILE="$PIPE_DIR/pipeline.log"
RPY="${PYTHON:-python3}"
export PYTHON="$RPY" PYTHONPATH="$ROOT" ORBITWARS_SKIP_PARITY=1
while [ $# -gt 0 ]; do
  case "$1" in
    --updates) UPDATES="$2"; shift 2 ;;
    --skip-post) SKIP_POST=1; shift ;;
    *) echo "unknown: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$PIPE_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== v30c pipeline $(date -Iseconds) updates=$UPDATES ==="
bash scripts/v30c_smoke.sh
bash scripts/v30c_extend.sh "$UPDATES"
if [ "$SKIP_POST" -eq 0 ]; then bash scripts/v30c_post_train.sh; fi
touch "$PIPE_DIR/status.done"
echo "=== v30c pipeline DONE ==="
