"""Read the tail of a training log, compute a recent-window mean of key
metrics, and decide PROMOTE / KILL / CONTINUE against Day5 FAST thresholds.

Used by ``scripts/run_fast_serial.sh`` to gate one FAST run after another
without running for the full 4000 updates: as soon as a window-mean clearly
clears or fails the bar, we stop early.

Returns exit codes the shell script understands:

    0  PROMOTE   (gate passed -- caller should mark winner)
    1  CONTINUE  (still in-flight, not enough updates yet)
    2  KILL      (gate failed sustainedly -- caller should kill the run)
    3  ERROR     (log unreadable / no parseable lines)

Decisions only kick in once we have at least ``--min-upd`` parseable update
rows. The window is the LAST ``--window`` lines.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from orbit_wars_rl.scripts.monitor_train import parse_one_line


def _tail_lines(path: Path, n_max: int = 5000) -> List[str]:
    try:
        with path.open("r", errors="replace") as f:
            data = f.readlines()
    except OSError:
        return []
    return data[-n_max:]


def _window(path: Path, window: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in _tail_lines(path):
        d = parse_one_line(line)
        if d is not None:
            rows.append(d)
    return rows[-window:]


def _mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
    return statistics.mean(vals) if vals else None


def evaluate(
    rows: List[Dict[str, Any]],
    *,
    baseline_pdelta: float,
    baseline_spf: float,
    baseline_garr: float,
    min_upd: int,
    ev_min: float,
    clip_max: float,
) -> Dict[str, Any]:
    """Compute the decision dict; see module docstring for exit-code semantics."""
    if not rows:
        return dict(verdict="ERROR", reason="no parseable rows", metrics={})

    last_upd = rows[-1].get("upd", 0)
    metrics = dict(
        upd=last_upd,
        ev=_mean(rows, "ev"),
        clip=_mean(rows, "clip"),
        kl=_mean(rows, "kl"),
        spf=_mean(rows, "spf"),
        garr=_mean(rows, "garr"),
        pdelta=_mean(rows, "pdelta"),
        pS=_mean(rows, "pS"),
    )

    if last_upd < min_upd:
        return dict(verdict="CONTINUE", reason=f"upd {last_upd} < {min_upd}",
                    metrics=metrics)

    # KILL conditions (must be sustained across the window).
    if metrics["ev"] is not None and metrics["ev"] < ev_min:
        return dict(verdict="KILL", reason=f"ev {metrics['ev']:.2f} < {ev_min}",
                    metrics=metrics)
    if metrics["clip"] is not None and metrics["clip"] > clip_max:
        return dict(verdict="KILL", reason=f"clip {metrics['clip']:.2f} > {clip_max}",
                    metrics=metrics)

    # PROMOTE conditions. We use a 2-of-3 vote on signal metrics so any
    # one noisy metric doesn't decide alone:
    #   (a) spf above the baseline by a clear margin
    #   (b) garr above the baseline by a clear margin
    #   (c) prod_share_delta positive (territory growing) above baseline
    votes = 0
    if metrics["spf"] is not None and metrics["spf"] > baseline_spf + 2.0:
        votes += 1
    if metrics["garr"] is not None and metrics["garr"] > baseline_garr + 5.0:
        votes += 1
    if (metrics["pdelta"] is not None
            and metrics["pdelta"] > baseline_pdelta + 0.02):
        votes += 1

    if votes >= 2:
        return dict(verdict="PROMOTE", reason=f"votes={votes}/3", metrics=metrics)

    return dict(verdict="CONTINUE", reason=f"votes={votes}/3, need 2", metrics=metrics)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--window", type=int, default=50,
                    help="number of most-recent parsed lines to average over")
    ap.add_argument("--min-upd", type=int, default=200,
                    help="only decide PROMOTE/KILL after upd >= this")
    ap.add_argument("--baseline-spf", type=float, default=21.0,
                    help="frozen base train-log spf (v9b @3999 = 21.2)")
    ap.add_argument("--baseline-garr", type=float, default=50.0,
                    help="frozen base train-log garr (v9b @3999 = 49.9)")
    ap.add_argument("--baseline-pdelta", type=float, default=0.0,
                    help="frozen base prod_share_delta (level-form base = 0)")
    ap.add_argument("--ev-min", type=float, default=0.30,
                    help="hard kill if window-mean ev < this")
    ap.add_argument("--clip-max", type=float, default=0.35,
                    help="hard kill if window-mean clip_frac > this")
    args = ap.parse_args(argv)

    rows = _window(args.log, args.window)
    verdict = evaluate(
        rows,
        baseline_pdelta=args.baseline_pdelta,
        baseline_spf=args.baseline_spf,
        baseline_garr=args.baseline_garr,
        min_upd=args.min_upd,
        ev_min=args.ev_min,
        clip_max=args.clip_max,
    )
    print(f"[gate] {verdict['verdict']:<8} {verdict['reason']}")
    m = verdict["metrics"]
    if m:
        print(f"[gate] upd={m.get('upd')}  ev={m.get('ev')}  "
              f"clip={m.get('clip')}  spf={m.get('spf')}  "
              f"garr={m.get('garr')}  pdelta={m.get('pdelta')}  pS={m.get('pS')}")

    return {"PROMOTE": 0, "CONTINUE": 1, "KILL": 2, "ERROR": 3}[verdict["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
