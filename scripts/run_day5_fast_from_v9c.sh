#!/usr/bin/env bash
# Day5 one-shot: kill v9c/v9d 4000-upd runs, resume from latest v9c ckpt,
# then serially probe R1 / R2 / R4 / combo with early gate stop.
#
# Run on the 5090 box (same machine as v9c/d training):
#
#   bash scripts/run_day5_fast_from_v9c.sh
#
# Subset / overrides:
#   bash scripts/run_day5_fast_from_v9c.sh r1_only r2_release
#   RESUME_FROM=ckpt_multi_action_v9c/ckpt_003199.pkl MIN_UPD=400 bash scripts/run_day5_fast_from_v9c.sh
#   SKIP_KILL=1 bash scripts/run_day5_fast_from_v9c.sh   # v9c/d already stopped
#
# Notes:
#   * --resume-from loads v9c *weights* only; the FAST run restarts upd counter
#     at 0 with new shaping env vars (expected clip spike in first ~300 upd).
#   * Gate baseline defaults to v9c@~3200: spf=30, garr=44.
#   * v9c ckpt dir is left intact; only the training processes are killed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
SKIP_KILL="${SKIP_KILL:-0}"
CKPT_DIR="${CKPT_DIR:-ckpt_multi_action_v9c}"

_latest_ckpt() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    return 1
  fi
  # shellcheck disable=SC2012
  ls -1 "$dir"/ckpt_*.pkl 2>/dev/null | sort | tail -1
}

_kill_v9_runs() {
  echo "[kill] stopping v9c/v9d training processes ..."
  pkill -f 'orbit_wars_rl.scripts.train.*multi_action_v9c.yaml' 2>/dev/null || true
  pkill -f 'orbit_wars_rl.scripts.train.*multi_action_v9d.yaml' 2>/dev/null || true
  sleep 3
  if pgrep -f 'multi_action_v9[cd].yaml' >/dev/null 2>&1; then
    echo "[kill] still alive -- SIGKILL"
    pkill -9 -f 'multi_action_v9c.yaml' 2>/dev/null || true
    pkill -9 -f 'multi_action_v9d.yaml' 2>/dev/null || true
    sleep 2
  fi
  if pgrep -f 'multi_action_v9[cd].yaml' >/dev/null 2>&1; then
    echo "[ERR] v9c/d still running after kill; abort." >&2
    pgrep -af 'multi_action_v9[cd].yaml' >&2 || true
    exit 2
  fi
  echo "[kill] v9c/v9d stopped."
}

if [ "$SKIP_KILL" != "1" ]; then
  _kill_v9_runs
else
  echo "[kill] SKIP_KILL=1 -- not touching v9c/d processes"
fi

if [ -n "${RESUME_FROM:-}" ]; then
  if [ ! -f "$RESUME_FROM" ]; then
    echo "[ERR] RESUME_FROM not found: $RESUME_FROM" >&2
    exit 2
  fi
else
  RESUME_FROM="$(_latest_ckpt "$CKPT_DIR" || true)"
  if [ -z "$RESUME_FROM" ] || [ ! -f "$RESUME_FROM" ]; then
    echo "[ERR] no checkpoint in $CKPT_DIR (expected ckpt_XXXXXX.pkl from v9c run)" >&2
    exit 2
  fi
fi

# Optional: read last upd from v9c log for a sharper baseline (falls back to u3200).
if [ -z "${BASELINE_SPF:-}" ] || [ -z "${BASELINE_GARR:-}" ]; then
  v9c_log="logs/multi_action_v9c.log"
  if [ -f "$v9c_log" ]; then
    snap="$("$PY" -m orbit_wars_rl.scripts.monitor_train --once "$v9c_log" 2>/dev/null \
      | awk '/^multi_action_v9c/{print $2, $7, $9}' || true)"
    if [ -n "$snap" ]; then
      read -r _upd _spf _garr <<< "$snap"
      BASELINE_SPF="${BASELINE_SPF:-$_spf}"
      BASELINE_GARR="${BASELINE_GARR:-$_garr}"
      echo "[baseline] from $v9c_log: spf=$BASELINE_SPF garr=$BASELINE_GARR (upd $_upd)"
    fi
  fi
fi

export RESUME_FROM
export BASELINE_SPF="${BASELINE_SPF:-30.0}"
export BASELINE_GARR="${BASELINE_GARR:-44.0}"
export BASELINE_PDELTA="${BASELINE_PDELTA:-0.0}"
export JAX_FRAC="${JAX_FRAC:-0.85}"
export POLL_SEC="${POLL_SEC:-120}"
export MIN_UPD="${MIN_UPD:-300}"

echo ""
echo "[day5] resume_from=$RESUME_FROM"
echo "[day5] gate baseline spf=$BASELINE_SPF garr=$BASELINE_GARR min_upd=$MIN_UPD"
echo "[day5] queue: ${*:-r1_only r2_release r4_emit r1_r2_r4 (default)}"
echo ""

exec bash "$ROOT/scripts/run_fast_serial.sh" "$@"
