#!/usr/bin/env bash
# Day5 FAST serial launcher.
#
# Runs new-reward variants (R1 / R2 / R4 / combo) ONE AT A TIME, polls
# orbit_wars_rl.scripts.check_fast_gate every POLL_SEC seconds, and stops
# the current run as soon as the gate says PROMOTE (clear win) or KILL
# (clear loss). Avoids waiting the full 4000 updates per recipe.
#
# Verdict flow per recipe:
#   PROMOTE  -> record winner in logs/fast_serial_summary.tsv, move on
#   KILL     -> sigterm the training proc, record loss, move on
#   CONTINUE -> sleep POLL_SEC, re-check
#   (reach config num_updates without verdict) -> record INCONCLUSIVE
#
# Usage:
#   bash scripts/run_fast_serial.sh                       # run default queue
#   bash scripts/run_fast_serial.sh r1 r2_release         # subset, in order
#   POLL_SEC=180 MIN_UPD=300 bash scripts/run_fast_serial.sh
#   bash scripts/run_day5_fast_from_v9c.sh              # kill v9c/d + resume v9c + serial
#   RESUME_FROM=ckpt_multi_action_v9c/ckpt_003199.pkl bash scripts/run_fast_serial.sh r1_only
#
# Env knobs:
#   POLL_SEC      seconds between gate checks      (default 120)
#   WINDOW        gate window (lines)              (default 50)
#   MIN_UPD       min upd before gate can decide   (default 300)
#   JAX_FRAC      per-process VRAM cap             (default 0.85, serial)
#   RESUME_FROM   ckpt path to warm-start from     (default none = scratch)
#   BASELINE_SPF  frozen-base train-log spf        (default 30.0, v9c@~3200)
#   BASELINE_GARR frozen-base train-log garr       (default 44.0, v9c@~3200)
#
# Notes:
#   * Runs are SERIAL by design -- the gpu gets all of VRAM, training is
#     fastest, and one human can watch one log at a time.
#   * Each variant gets its own log + ckpt dir under logs/ and ckpt_*/ so
#     promoted candidates can be exported for h2h later without conflict.
#   * The gate is intentionally simple (vote of 3 metrics + hard EV/clip
#     kill); see check_fast_gate.py for the exact rules.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
POLL_SEC="${POLL_SEC:-120}"
WINDOW="${WINDOW:-50}"
MIN_UPD="${MIN_UPD:-300}"
JAX_FRAC="${JAX_FRAC:-0.85}"
RESUME_FROM="${RESUME_FROM:-}"
BASELINE_SPF="${BASELINE_SPF:-30.0}"
BASELINE_GARR="${BASELINE_GARR:-44.0}"
BASELINE_PDELTA="${BASELINE_PDELTA:-0.0}"

# v9c full v2 family -- all FAST variants resume from v9c ckpt and keep
# planet_share; R1/R2 swap prod_share / fleet_log for v3 terms.
V9C_BASE="ORBITWARS_SHAPING_PLANET_SHARE=0.005"

CFG="orbit_wars_rl/configs/multi_action_v10_fast.yaml"
SUMMARY="logs/fast_serial_summary.tsv"

mkdir -p logs

# Variant -> ENV_VARS for ORBITWARS_SHAPING_*.
# Resume weights from v9c; each variant swaps in ONE (or combo) v3 term.
# R1: prod_share level -> delta (mutually exclusive).
# R2: fleet_log -> release_bonus (mutually exclusive per DAY5_TRAINING_ACTIONS).
declare -A LAUNCH=(
  [r1_only]="${V9C_BASE} ORBITWARS_SHAPING_PROD_SHARE=0.0 ORBITWARS_SHAPING_FLEET_LOG=0.002 ORBITWARS_SHAPING_PROD_SHARE_DELTA=1.0"
  [r2_release]="${V9C_BASE} ORBITWARS_SHAPING_PROD_SHARE=0.01 ORBITWARS_SHAPING_FLEET_LOG=0.0 ORBITWARS_SHAPING_RELEASE=0.05 ORBITWARS_SHAPING_RELEASE_K=20.0"
  [r4_emit]="${V9C_BASE} ORBITWARS_SHAPING_PROD_SHARE=0.01 ORBITWARS_SHAPING_FLEET_LOG=0.002 ORBITWARS_SHAPING_EMIT_LOG=0.01"
  [r1_r2_r4]="${V9C_BASE} ORBITWARS_SHAPING_PROD_SHARE=0.0 ORBITWARS_SHAPING_FLEET_LOG=0.0 ORBITWARS_SHAPING_PROD_SHARE_DELTA=1.0 ORBITWARS_SHAPING_RELEASE=0.05 ORBITWARS_SHAPING_RELEASE_K=20.0 ORBITWARS_SHAPING_EMIT_LOG=0.01"
)

VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
  VARIANTS=(r1_only r2_release r4_emit r1_r2_r4)
fi

# Validate all names up front so a typo on variant #4 doesn't waste 3 hours.
for v in "${VARIANTS[@]}"; do
  if [ -z "${LAUNCH[$v]+x}" ]; then
    echo "[ERR] unknown variant '$v'. Known: ${!LAUNCH[*]}" >&2
    exit 2
  fi
done

# Ensure summary header exists.
if [ ! -f "$SUMMARY" ]; then
  printf "ts\tvariant\tverdict\tupd\tspf\tgarr\tpdelta\tev\tclip\tnotes\n" > "$SUMMARY"
fi

echo "[config] serial mode  POLL_SEC=$POLL_SEC  MIN_UPD=$MIN_UPD  JAX_FRAC=$JAX_FRAC"
echo "[config] resume_from=${RESUME_FROM:-<scratch>}"
echo "[config] queue: ${VARIANTS[*]}"
echo ""

for v in "${VARIANTS[@]}"; do
  envs="${LAUNCH[$v]}"
  tag="v10_${v}"
  log="logs/multi_action_${tag}.log"
  tb="logs/multi_action_${tag}"
  ckpt="ckpt_multi_action_${tag}"

  echo "================================================================"
  echo "[serial] starting variant=$v  log=$log  shaping=$envs"
  echo "================================================================"

  resume_arg=()
  if [ -n "$RESUME_FROM" ]; then
    resume_arg=(--resume-from "$RESUME_FROM")
  fi

  # Each variant overrides ckpt_dir via a temp YAML so promoted candidates
  # don't all dump into the same ckpt_multi_action_v10_fast/ dir.
  tmp_cfg="$(mktemp -t "fast_${v}_XXXX").yaml"
  # shellcheck disable=SC2016
  sed -E "s|ckpt_multi_action_v10_fast|${ckpt}|g" "$CFG" > "$tmp_cfg"

  # shellcheck disable=SC2086
  env $envs \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_FRAC" \
      nohup "$PY" -m orbit_wars_rl.scripts.train \
      --config "$tmp_cfg" \
      --log-dir "$tb" \
      "${resume_arg[@]}" \
      > "$log" 2>&1 &
  pid=$!
  echo "[serial] pid=$pid  config(tmp)=$tmp_cfg"

  verdict="INCONCLUSIVE"
  last_summary=""

  # Poll loop. We rely on check_fast_gate to read the tail of $log and
  # decide. The script's process exit code is the verdict.
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_SEC"
    set +e
    out="$("$PY" -m orbit_wars_rl.scripts.check_fast_gate \
        --log "$log" \
        --window "$WINDOW" \
        --min-upd "$MIN_UPD" \
        --baseline-spf "$BASELINE_SPF" \
        --baseline-garr "$BASELINE_GARR" \
        --baseline-pdelta "$BASELINE_PDELTA" 2>&1)"
    rc=$?
    set -e
    echo "[$(date +%H:%M:%S)] [$v] $out"
    last_summary="$out"

    case "$rc" in
      0)
        verdict="PROMOTE"
        echo "[serial] $v PROMOTE -- stopping training proc $pid"
        kill "$pid" 2>/dev/null || true
        sleep 5
        kill -9 "$pid" 2>/dev/null || true
        break
        ;;
      2)
        verdict="KILL"
        echo "[serial] $v KILL -- stopping training proc $pid"
        kill "$pid" 2>/dev/null || true
        sleep 5
        kill -9 "$pid" 2>/dev/null || true
        break
        ;;
      1)
        ;;  # CONTINUE
      3)
        ;;  # ERROR -- usually "no parseable rows yet"; keep polling
      *)
        echo "[serial] gate returned unexpected rc=$rc; continuing"
        ;;
    esac
  done

  # Reaped or killed; collect a final reading either way.
  set +e
  final="$("$PY" -m orbit_wars_rl.scripts.check_fast_gate \
      --log "$log" \
      --window "$WINDOW" \
      --min-upd "$MIN_UPD" \
      --baseline-spf "$BASELINE_SPF" \
      --baseline-garr "$BASELINE_GARR" \
      --baseline-pdelta "$BASELINE_PDELTA" 2>&1)"
  set -e

  # Parse the second line (upd=... ev=... ...) into TSV fields. We use
  # awk so this works whether or not optional metrics are present.
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  metrics_line="$(echo "$final" | grep '^\[gate\] upd=' || true)"
  upd="$(echo "$metrics_line"  | sed -nE 's/.*upd=([^ ]+).*/\1/p')"
  ev="$(echo "$metrics_line"   | sed -nE 's/.*ev=([^ ]+).*/\1/p')"
  clip="$(echo "$metrics_line" | sed -nE 's/.*clip=([^ ]+).*/\1/p')"
  spf="$(echo "$metrics_line"  | sed -nE 's/.*spf=([^ ]+).*/\1/p')"
  garr="$(echo "$metrics_line" | sed -nE 's/.*garr=([^ ]+).*/\1/p')"
  pdel="$(echo "$metrics_line" | sed -nE 's/.*pdelta=([^ ]+).*/\1/p')"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$ts" "$v" "$verdict" "${upd:-?}" "${spf:-?}" "${garr:-?}" \
    "${pdel:-?}" "${ev:-?}" "${clip:-?}" "shaping=$envs" >> "$SUMMARY"

  rm -f "$tmp_cfg"
  echo "[serial] $v done. verdict=$verdict  summary appended -> $SUMMARY"
  echo ""

  # Polite gap so the user can ^C between variants if they want.
  sleep 5
done

echo "================================================================"
echo "[serial] all variants finished. Summary:"
column -t -s $'\t' "$SUMMARY" | tail -n $((${#VARIANTS[@]} + 1))
echo ""
echo "Next steps for any PROMOTE'd variant:"
echo "  1. Export the latest ckpt:"
echo "       python -m orbit_wars_rl.scripts.export_submission \\"
echo "           --ckpt ckpt_multi_action_v10_<variant>/ckpt_000XXX.pkl \\"
echo "           --out ckpt_multi_action_v10_<variant>/submission.py"
echo "  2. h2h vs v20 and frozen v9c -- see docs/H2H_EVAL_RUNBOOK.md"
echo "  3. If h2h beats v9c, run the OVERNIGHT 4000-update version"
echo "     by re-launching with num_updates=4000 (edit the v10 fast yaml,"
echo "     or use the existing run_v9_ablation.sh harness)."
