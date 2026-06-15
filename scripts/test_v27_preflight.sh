#!/usr/bin/env bash
# v27 preflight: python/jax, resume ckpt, submission template, v27 scripts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RDIR="${RDIR:-$ROOT}"
REMOTE_RDIR="${REMOTE_RDIR:-/home/charlie/project/OrbitWarRL}"
REMOTE="${REMOTE:-charlie@www.ultrapp.online}"
RPY="${RPY:-/home/charlie/anaconda3/bin/python}"
CHECK_REMOTE="${CHECK_REMOTE:-1}"

_fail() { echo "[preflight FAIL] $*" >&2; exit 1; }
_ok() { echo "[preflight OK] $*"; }

if [ "$CHECK_REMOTE" = "1" ]; then
  if ssh -o ConnectTimeout=10 "$REMOTE" "test -x '$RPY'" 2>/dev/null; then
    ssh -o ConnectTimeout=15 "$REMOTE" "'$RPY' -c 'import jax, numpy'" \
      || _fail "remote python missing jax/numpy"
    _ok "remote python/jax"
    ssh -o ConnectTimeout=15 "$REMOTE" \
      "(test -f '$REMOTE_RDIR/ckpt_multi_action_v26_roi/ckpt_latest.pkl' || test -f '$REMOTE_RDIR/ckpt_multi_action_v25_extend/ckpt_009599.pkl')" \
      || _fail "missing remote v26 latest or v25 u9599 ckpt"
    _ok "remote resume ckpt"
    ssh -o ConnectTimeout=15 "$REMOTE" \
      "test -f '$REMOTE_RDIR/submission_v20_0513.py'" \
      || _fail "missing remote submission_v20_0513.py"
    _ok "remote submission_v20_0513.py"
  else
    echo "[preflight WARN] cannot ssh $REMOTE — falling back to local checks"
    CHECK_REMOTE=0
  fi
fi

if [ "$CHECK_REMOTE" = "0" ]; then
  PY="${PYTHON:-python3}"
  if ! "$PY" -c "import jax, numpy" 2>/dev/null; then
    PY="/home/charlie/anaconda3/bin/python"
  fi
  "$PY" -c "import jax, numpy" || _fail "local python missing jax/numpy"
  _ok "local python/jax"
fi

for f in \
  orbit_wars_rl/configs/multi_action_v27_frontier.yaml \
  scripts/v27_extend.sh \
  scripts/v27_pipeline.sh \
  scripts/v27_smoke.sh \
  scripts/v27_remote.sh \
  scripts/v27_one_click.sh; do
  [ -f "$ROOT/$f" ] || _fail "missing $f"
done
_ok "v27 config/scripts present"

for f in scripts/v27_extend.sh scripts/v27_pipeline.sh scripts/v27_smoke.sh \
  scripts/v27_remote.sh scripts/v27_one_click.sh scripts/v27_post_train.sh \
  scripts/test_v27_preflight.sh scripts/bc_collect_v27_top10.sh; do
  [ -x "$ROOT/$f" ] || chmod +x "$ROOT/$f" 2>/dev/null || true
done

echo "[preflight PASS]"
