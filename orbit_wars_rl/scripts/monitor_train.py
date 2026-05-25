"""Multi-log training monitor — tail one or more v9.*.log files and surface
warnings when key health metrics drift out of band.

Day 3 §13 deliverable; required before launching the v9 ablation so 4
parallel runs are observable without a human staring at 4 terminals.

Thresholds (top1 §307 + §102, DAY3/DAY4 audit):
  * explained_variance:  WARN < 0.5,  ALERT < 0.3,  RECOVER > 0.7
  * clip_frac:           WARN > 0.25, ALERT > 0.35
  * approx_kl:           WARN > 0.05, ALERT > 0.10  (PPO update too aggressive)
  * loss:                ALERT |loss| > 5.0       (numerics broken)
  * spf  (mean ships/fleet): info only, but flag if stuck < 3 after upd 500
  * z0   (zero-emit rate):  info only

Usage:
    python -m orbit_wars_rl.scripts.monitor_train logs/multi_action_v9*.log
    python -m orbit_wars_rl.scripts.monitor_train --once logs/*.log   # one-shot summary
    python -m orbit_wars_rl.scripts.monitor_train --interval 60 logs/*.log
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


LINE_RE = re.compile(
    r"upd\s+(?P<upd>\d+)\s+steps\s+(?P<steps>\d+)\s+sps\s+(?P<sps>[\d.]+)\s+"
    r"opp\s+(?P<opp>\S+)\s+"
    r"loss\s+(?P<loss>[+-]?[\d.]+)\s+"
    r"pg\s+(?P<pg>[+-]?[\d.]+)\s+"
    r"v\s+(?P<v>[+-]?[\d.]+)\s+"
    r"ev\s+(?P<ev>[+-]?[\d.]+)\s+"
    r"adv_std\s+(?P<adv_std>[+-]?[\d.]+)\s+"
    r"tR\s+(?P<tR>[+-]?[\d.]+)\s+"
    r"ent\[s/d/p/e\]\s+(?P<ent_s>[\d.]+)/(?P<ent_d>[\d.]+)/(?P<ent_p>[\d.]+)/(?P<ent_e>[\d.]+)\s+"
    r"emits\s+(?P<emits>[\d.]+)\s+"
    r"spf\s+(?P<spf>[\d.]+)\s+"
    r"z0\s+(?P<z0>[\d.]+)\s+"
    r"garr\s+(?P<garr>[\d.]+)\s+"
    r"(?:pS\s+(?P<pS>[\d.]+)\s+ptS\s+(?P<ptS>[\d.]+)\s+fLog\s+(?P<fLog>[\d.]+)\s+)?"
    r"(?:pd\u0394\s+(?P<pdelta>[+-]?[\d.]+)\s+pkR\s+(?P<pkR>[\d.]+)\s+)?"
    r"clip\s+(?P<clip>[\d.]+)\s+"
    r"kl\s+(?P<kl>[+-]?[\d.]+)"
)


@dataclass
class Thresholds:
    ev_warn: float = 0.5
    ev_alert: float = 0.3
    clip_warn: float = 0.25
    clip_alert: float = 0.35
    kl_warn: float = 0.05
    kl_alert: float = 0.10
    loss_alert: float = 5.0
    spf_stuck_after_upd: int = 500
    spf_stuck_value: float = 3.0


@dataclass
class RunState:
    path: str
    last_upd: int = -1
    last_metrics: Optional[Dict[str, float]] = None
    # Last 50 updates' EV / clip / kl for trend detection.
    recent_ev: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    recent_clip: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    recent_kl: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    # Per-warning latch so we don't spam (warn at most once every 200 updates).
    last_warn_at: Dict[str, int] = field(default_factory=dict)
    finished: bool = False  # set when process appears to have exited


def parse_one_line(line: str) -> Optional[Dict[str, float]]:
    m = LINE_RE.search(line)
    if not m:
        return None
    d = m.groupdict()
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        try:
            out[k] = float(v) if k not in ("upd", "steps") else int(v)
        except ValueError:
            out[k] = v
    return out


def tail_file_lines(path: str, start_offset: int) -> Tuple[List[str], int]:
    """Read new lines since ``start_offset``. Returns (new_lines, new_offset)."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return [], start_offset
    if size < start_offset:
        # Log was rotated / truncated -- restart.
        start_offset = 0
    if size == start_offset:
        return [], start_offset
    with open(path, "r", errors="replace") as f:
        f.seek(start_offset)
        chunk = f.read()
    new_offset = start_offset + len(chunk.encode("utf-8", errors="replace"))
    return chunk.splitlines(), new_offset


def check_warnings(name: str, st: RunState, t: Thresholds) -> List[str]:
    """Return list of (severity, message) strings for any band-violation."""
    out: List[str] = []
    m = st.last_metrics
    if m is None:
        return out
    upd = int(m.get("upd", 0))

    def _maybe(tag: str, severity: str, msg: str, cooldown: int = 200) -> None:
        last = st.last_warn_at.get(tag, -10**9)
        if upd - last >= cooldown:
            out.append(f"[{severity}] {name} upd {upd}: {msg}")
            st.last_warn_at[tag] = upd

    ev = m.get("ev")
    if ev is not None:
        if ev < t.ev_alert:
            _maybe("ev_alert", "ALERT", f"explained_variance {ev:.2f} < {t.ev_alert}")
        elif ev < t.ev_warn:
            _maybe("ev_warn", "WARN", f"explained_variance {ev:.2f} < {t.ev_warn}")

    clip = m.get("clip")
    if clip is not None:
        if clip > t.clip_alert:
            _maybe("clip_alert", "ALERT", f"clip_frac {clip:.2f} > {t.clip_alert}")
        elif clip > t.clip_warn:
            _maybe("clip_warn", "WARN", f"clip_frac {clip:.2f} > {t.clip_warn}")

    kl = m.get("kl")
    if kl is not None:
        if abs(kl) > t.kl_alert:
            _maybe("kl_alert", "ALERT", f"|approx_kl| {kl:+.3f} > {t.kl_alert}")
        elif abs(kl) > t.kl_warn:
            _maybe("kl_warn", "WARN", f"|approx_kl| {kl:+.3f} > {t.kl_warn}")

    loss = m.get("loss")
    if loss is not None and abs(loss) > t.loss_alert:
        _maybe("loss_alert", "ALERT", f"|loss| {loss:+.2f} > {t.loss_alert} (numerics?)")

    spf = m.get("spf")
    if (
        spf is not None
        and upd >= t.spf_stuck_after_upd
        and spf < t.spf_stuck_value
    ):
        _maybe(
            "spf_stuck",
            "WARN",
            f"spf {spf:.1f} < {t.spf_stuck_value} after upd {t.spf_stuck_after_upd} "
            f"(policy collapsed to micro-fleets?)",
            cooldown=500,
        )

    return out


def render_summary(states: Dict[str, RunState]) -> None:
    if not states:
        print("(no logs to monitor)")
        return
    header = (
        f"{'name':<24} {'upd':>5} {'steps':>8} {'ev':>5} {'clip':>5} {'kl':>6} "
        f"{'spf':>5} {'z0':>4} {'garr':>5} {'pS':>4} {'ptS':>4} {'fLog':>4} "
        f"{'WRr':>4} {'status':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, st in sorted(states.items()):
        m = st.last_metrics or {}
        upd = m.get("upd", "-")
        steps = m.get("steps", "-")
        def _f(k: str, w: int = 5, d: int = 2, default: str = " -") -> str:
            v = m.get(k)
            if v is None:
                return f"{default:>{w}}"
            return f"{v:>{w}.{d}f}"
        # WRr (win rate vs random) is not parsed by LINE_RE (it's optional and
        # appears at end of some lines). Skip for now.
        status = "DONE" if st.finished else "live"
        print(
            f"{name:<24} {upd:>5} {steps:>8} "
            f"{_f('ev', 5, 2)} {_f('clip', 5, 2)} {_f('kl', 6, 3)} "
            f"{_f('spf', 5, 1)} {_f('z0', 4, 2)} {_f('garr', 5, 1)} "
            f"{_f('pS', 4, 2)} {_f('ptS', 4, 2)} {_f('fLog', 4, 2)} "
            f"{'-':>4} "
            f"{status:>8}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="log files (globs allowed)")
    p.add_argument("--once", action="store_true",
                   help="render summary once and exit")
    p.add_argument("--interval", type=float, default=30.0,
                   help="seconds between updates (default 30)")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-update warnings; only show table")
    args = p.parse_args(argv)

    expanded: List[str] = []
    for pat in args.logs:
        for g in sorted(glob.glob(pat)):
            expanded.append(g)
    # de-dup preserving order
    expanded = list(dict.fromkeys(expanded))
    if not expanded:
        print("ERR: no log files matched", file=sys.stderr)
        return 2

    states: Dict[str, RunState] = {}
    offsets: Dict[str, int] = {}
    for path in expanded:
        name = os.path.basename(path).replace(".log", "")
        states[name] = RunState(path=path)
        offsets[path] = 0

    t = Thresholds()
    print(f"Monitoring {len(expanded)} log file(s):")
    for path in expanded:
        print(f"  - {path}")
    print()

    while True:
        for name, st in states.items():
            new_lines, new_offset = tail_file_lines(st.path, offsets[st.path])
            offsets[st.path] = new_offset
            for line in new_lines:
                m = parse_one_line(line)
                if m is None:
                    continue
                st.last_metrics = m
                st.last_upd = int(m.get("upd", st.last_upd))
                if "ev" in m:
                    st.recent_ev.append(float(m["ev"]))
                if "clip" in m:
                    st.recent_clip.append(float(m["clip"]))
                if "kl" in m:
                    st.recent_kl.append(float(m["kl"]))
                if not args.quiet:
                    for w in check_warnings(name, st, t):
                        print(w, flush=True)

        print()
        render_summary(states)
        sys.stdout.flush()

        if args.once:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nstopped by user")
            return 0


if __name__ == "__main__":
    sys.exit(main())
