"""Replay analyzer — extract behaviour signatures from kaggle env games.

Day 4 Track 1 deliverable. Distinct from ``replay_dump.py`` (which prints
turn-by-turn human-readable rows). This script runs **N games** between two
agents, then for each agent / each turn computes the 6 behaviour metrics we
care about for reward-shaping diagnosis:

    1.  ships_per_fleet (mean ships in each launched fleet)
    2.  pct_bin_hist    (histogram over 8 pct bins of ships_sent / src_garrison)
    3.  zero_emit_rate  (fraction of turns with 0 launches)
    4.  garrison_my     (total ships on my planets, turn-start)
    5.  fleet_sun_loss  (fleets that disappear into the sun, counted via diff)
    6.  capture_success (planet ownership flips from neutral/enemy → mine)

The point isn't to learn from v20 — it's to compare v8-vs-v8 and v20-vs-v20
side-by-side so we can SEE which behaviour the v8 reward landscape is
silent about. This drives the Track 3 reward-shaping ablation choice.

Usage:
    python -m orbit_wars_rl.scripts.replay_analyze \
        --agent-a submission_rl_v7_u499.py \
        --agent-b submission_v20_0513.py \
        --num-games 10 \
        --out logs/replay_analyze_v7_vs_v20.json

Each game seed is ``seed_base + i``. JSON output contains per-game
per-player time-series + aggregate stats.

Pure stdlib + kaggle_environments. No JAX. Runs on any machine.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kaggle_environments import make


# Must match orbit_wars_rl.env.constants.PCT_BIN_VALUES (kept inline so this
# script has no dependency on the JAX env package -- can run anywhere).
PCT_BIN_VALUES: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00)


# -------------------------- per-turn extraction --------------------------


def _planet_ships(obs: Dict[str, Any], owner: int) -> int:
    return sum(int(p[5]) for p in obs.get("planets") or [] if int(p[1]) == owner)


def _planet_owners(obs: Dict[str, Any]) -> Dict[int, int]:
    return {int(p[0]): int(p[1]) for p in obs.get("planets") or []}


def _planet_garrison_map(obs: Dict[str, Any]) -> Dict[int, int]:
    return {int(p[0]): int(p[5]) for p in obs.get("planets") or []}


def _fleet_ids(obs: Dict[str, Any], owner: int) -> set:
    return {int(f[0]) for f in obs.get("fleets") or [] if int(f[1]) == owner}


def _classify_pct_bin(ships_sent: int, src_garrison: int) -> int:
    """Return the bin index in PCT_BIN_VALUES whose value is *closest* to
    ships_sent / src_garrison. Used to reconstruct what fraction an agent
    "intended" — we never see the policy directly, only the (src, ships)
    they actually launched.
    """
    if src_garrison <= 0:
        return 0
    frac = ships_sent / float(src_garrison)
    best_b = 0
    best_d = abs(PCT_BIN_VALUES[0] - frac)
    for b, v in enumerate(PCT_BIN_VALUES):
        d = abs(v - frac)
        if d < best_d:
            best_d = d
            best_b = b
    return best_b


# -------------------------- per-game traces --------------------------


def _empty_trace() -> Dict[str, list]:
    return {
        "ships_per_fleet": [],   # per turn: mean of ships in launches this turn (None if no launches)
        "n_emits": [],           # per turn: # of launches
        "garrison_my": [],       # per turn: total ships on my planets (turn-start)
        "pct_bin_per_launch": [],  # all launches across game: list of (turn, bin_idx)
        "captures": 0,           # how many planets owner-flipped TO me during the game
        "fleets_launched": 0,
        "fleets_arrived": 0,     # fleets that left + later disappeared but a capture occurred
        "fleet_disappear_no_capture": 0,  # likely sun-loss / out-of-bounds (proxy for sun_loss)
        "outcome": None,         # 'win' / 'loss' / 'draw' / 'error'
    }


def analyse_one_game(env_steps: List[List[Any]]) -> Tuple[Dict[str, list], Dict[str, list]]:
    """Walk kaggle env trajectory; return (trace_p0, trace_p1).

    env_steps[t][p] has keys ``observation`` and ``action``. ``action`` is
    the action *just submitted* on this step (so ``env_steps[t][p]["action"]``
    is the move that produced the state at t+1).
    """
    trace_p0 = _empty_trace()
    trace_p1 = _empty_trace()

    prev_owners: Optional[Dict[int, int]] = None
    prev_fleet_ids = {0: set(), 1: set()}

    for t, step in enumerate(env_steps):
        obs_p0 = step[0].get("observation", {}) or {}
        # In kaggle 2p mode planets/fleets are the same shared schema; obs
        # differs only in 'player' field. Use obs_p0 for ownership state.
        obs_shared = obs_p0
        owners = _planet_owners(obs_shared)
        garrison_map = _planet_garrison_map(obs_shared)

        # 1. garrison_my @ this turn (turn-start state)
        my_ships_p0 = sum(g for pid, g in garrison_map.items() if owners.get(pid, -1) == 0)
        my_ships_p1 = sum(g for pid, g in garrison_map.items() if owners.get(pid, -1) == 1)
        trace_p0["garrison_my"].append(my_ships_p0)
        trace_p1["garrison_my"].append(my_ships_p1)

        # 2. actions submitted at this turn → ships per fleet, pct_bin
        for player_idx, trace in ((0, trace_p0), (1, trace_p1)):
            act = step[player_idx].get("action")
            if not isinstance(act, list):
                trace["ships_per_fleet"].append(None)
                trace["n_emits"].append(0)
                continue
            n = 0
            ships_sum = 0
            for m in act:
                if not (isinstance(m, list) and len(m) == 3):
                    continue
                try:
                    src_id = int(m[0])
                    ships = int(m[2])
                except (TypeError, ValueError):
                    continue
                if ships <= 0:
                    continue
                n += 1
                ships_sum += ships
                src_g = garrison_map.get(src_id, 0)
                bin_idx = _classify_pct_bin(ships, src_g)
                trace["pct_bin_per_launch"].append((t, bin_idx))
                trace["fleets_launched"] += 1
            if n == 0:
                trace["ships_per_fleet"].append(None)
                trace["n_emits"].append(0)
            else:
                trace["ships_per_fleet"].append(ships_sum / float(n))
                trace["n_emits"].append(n)

        # 3. captures: count planets that flipped TO us during this turn
        if prev_owners is not None:
            for pid, prev_o in prev_owners.items():
                new_o = owners.get(pid, prev_o)
                if new_o != prev_o:
                    if new_o == 0:
                        trace_p0["captures"] += 1
                    elif new_o == 1:
                        trace_p1["captures"] += 1

        # 4. fleet vanish accounting (CAVEAT: this is a CRUDE PROXY).
        #    A fleet "vanishes" when it was in obs[t-1] but not in obs[t].
        #    In orbit_wars that can mean MANY things: sun loss, out-of-bounds,
        #    arrival-with-capture, arrival-without-capture (combat absorbed
        #    against a stronger garrison or enemy fleet), or being swept by
        #    a rotating planet. We only distinguish "arrival caused a flip"
        #    from "everything else". So `fleet_loss_rate` here is really
        #    `non-flip outcome rate` — it WILL be high even for v20 because
        #    most fleets reinforce already-owned planets or get absorbed in
        #    combat. Use it for delta comparison between agents, NOT as a
        #    true "sun loss" indicator.
        curr_fleet_ids = {p: _fleet_ids(obs_shared, p) for p in (0, 1)}
        for player_idx, trace in ((0, trace_p0), (1, trace_p1)):
            vanished = prev_fleet_ids[player_idx] - curr_fleet_ids[player_idx]
            if not vanished:
                continue
            # rough attribution: any planet owner-flip this turn benefitting
            # us = an arrival; the rest of the vanished count = sun/oob/absorbed
            arrivals = 0
            if prev_owners is not None:
                for pid, prev_o in prev_owners.items():
                    new_o = owners.get(pid, prev_o)
                    if new_o != prev_o and new_o == player_idx:
                        arrivals += 1
            arrivals = min(arrivals, len(vanished))
            trace["fleets_arrived"] += arrivals
            trace["fleet_disappear_no_capture"] += len(vanished) - arrivals
        prev_owners = owners
        prev_fleet_ids = curr_fleet_ids

    # outcome
    final = env_steps[-1]
    try:
        r0 = float(final[0].get("reward") or 0)
        r1 = float(final[1].get("reward") or 0)
    except (TypeError, ValueError):
        r0 = r1 = 0
    if r0 > r1:
        trace_p0["outcome"] = "win"
        trace_p1["outcome"] = "loss"
    elif r1 > r0:
        trace_p0["outcome"] = "loss"
        trace_p1["outcome"] = "win"
    else:
        trace_p0["outcome"] = "draw"
        trace_p1["outcome"] = "draw"

    return trace_p0, trace_p1


# -------------------------- aggregate across games --------------------------


def _safe_mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def aggregate(traces: List[Dict[str, list]], turn_lo: int = 0, turn_hi: Optional[int] = None) -> Dict[str, Any]:
    """Roll up per-game traces into one summary row.

    ``turn_lo`` / ``turn_hi`` slice the per-turn series. Default = full game.
    Use ``turn_hi=80`` to look ONLY at the first 80 turns (matches our
    training ``episode_steps=80`` -- this is what the v8 policy actually
    sees and gets rewarded for).
    """
    n = len(traces)
    if n == 0:
        return {}

    def _slice(series: List[Any]) -> List[Any]:
        hi = turn_hi if turn_hi is not None else len(series)
        return series[turn_lo:hi]

    # Flatten per-turn series across games for histograms.
    all_emits = [v for t in traces for v in _slice(t["n_emits"])]
    all_ships_per_fleet = [
        v for t in traces for v in _slice(t["ships_per_fleet"]) if v is not None
    ]
    all_garrison_my = [v for t in traces for v in _slice(t["garrison_my"])]
    all_bins = [
        b for t in traces for (turn, b) in t["pct_bin_per_launch"]
        if turn_lo <= turn < (turn_hi if turn_hi is not None else 10**9)
    ]

    n_turns_total = sum(len(t["n_emits"]) for t in traces)
    n_zero_emit_turns = sum(1 for v in all_emits if v == 0)

    bin_hist = [0] * len(PCT_BIN_VALUES)
    for b in all_bins:
        bin_hist[b] += 1
    bin_total = sum(bin_hist) or 1
    bin_dist = [b / bin_total for b in bin_hist]

    # emit count histogram (0..8+)
    emit_hist = [0] * 9
    for v in all_emits:
        v_clip = min(v, 8)
        emit_hist[v_clip] += 1
    emit_total = sum(emit_hist) or 1
    emit_dist = [c / emit_total for c in emit_hist]

    # game-level outcomes
    wins = sum(1 for t in traces if t["outcome"] == "win")
    losses = sum(1 for t in traces if t["outcome"] == "loss")
    draws = sum(1 for t in traces if t["outcome"] == "draw")

    fleets_launched = sum(t["fleets_launched"] for t in traces)
    fleets_arrived = sum(t["fleets_arrived"] for t in traces)
    fleets_disappear_no_capture = sum(t["fleet_disappear_no_capture"] for t in traces)
    captures = sum(t["captures"] for t in traces)

    return {
        "n_games": n,
        "n_turns_total": n_turns_total,
        "turn_window": [turn_lo, turn_hi],
        "outcome": {"win": wins, "loss": losses, "draw": draws},
        # per-turn rates
        "zero_emit_rate": n_zero_emit_turns / n_turns_total,
        "mean_emits_per_turn": _safe_mean(all_emits) or 0.0,
        # per-launch averages
        "mean_ships_per_fleet": _safe_mean(all_ships_per_fleet) or 0.0,
        "pct_bin_distribution": bin_dist,
        "pct_bin_values": list(PCT_BIN_VALUES),
        # state averages
        "mean_garrison_my": _safe_mean(all_garrison_my) or 0.0,
        "max_garrison_my": max(all_garrison_my) if all_garrison_my else 0,
        "p95_garrison_my": (
            sorted(all_garrison_my)[int(0.95 * len(all_garrison_my))]
            if all_garrison_my
            else 0
        ),
        # event counts (across all games)
        "fleets_launched": fleets_launched,
        "fleets_arrived": fleets_arrived,
        "fleets_disappear_no_capture": fleets_disappear_no_capture,
        "captures": captures,
        "fleet_arrival_rate": fleets_arrived / max(1, fleets_launched),
        "fleet_loss_rate": fleets_disappear_no_capture / max(1, fleets_launched),
        # emit-per-turn histogram (mostly for v20 64% 0-emit comparison)
        "emit_count_distribution": emit_dist,
    }


def print_summary(name: str, agg: Dict[str, Any]) -> None:
    if not agg:
        print(f"{name}: no data")
        return
    out = agg["outcome"]
    print(f"=== {name} ({agg['n_games']} games, {agg['n_turns_total']} turns) ===")
    print(f"  outcome              W/L/D = {out['win']}/{out['loss']}/{out['draw']}")
    print(f"  mean_emits_per_turn  = {agg['mean_emits_per_turn']:.2f}")
    print(f"  zero_emit_rate       = {agg['zero_emit_rate']:.2%}")
    print(f"  mean_ships_per_fleet = {agg['mean_ships_per_fleet']:.2f}")
    print(f"  mean_garrison_my     = {agg['mean_garrison_my']:.2f}")
    print(f"  p95_garrison_my      = {agg['p95_garrison_my']}")
    print(f"  fleets_launched      = {agg['fleets_launched']}")
    print(f"  fleet_flip_rate      = {agg['fleet_arrival_rate']:.2%}   (fleets whose arrival flipped a planet)")
    print(f"  fleet_nonflip_rate   = {agg['fleet_loss_rate']:.2%}   (crude proxy; see source for caveat)")
    print(f"  captures             = {agg['captures']}")
    print("  pct_bin_distribution (bin@val: pct):")
    for b, v in enumerate(PCT_BIN_VALUES):
        print(f"      bin{b} @ {v:.2f}: {agg['pct_bin_distribution'][b]:.1%}")
    print("  emit_count_distribution (n_emits per turn: pct):")
    for n, p in enumerate(agg["emit_count_distribution"]):
        print(f"      {n}: {p:.1%}")
    print()


# -------------------------- main --------------------------


def run_games(
    agent_a: str,
    agent_b: str,
    num_games: int,
    seed_base: int,
    verbose: bool,
) -> List[Tuple[Dict[str, list], Dict[str, list]]]:
    out: List[Tuple[Dict[str, list], Dict[str, list]]] = []
    for i in range(num_games):
        seed = seed_base + i
        cfg = {"seed": seed}
        env = make("orbit_wars", configuration=cfg, debug=False)
        t0 = time.time()
        env.run([agent_a, agent_b])
        elapsed = time.time() - t0
        tr0, tr1 = analyse_one_game(env.steps)
        if verbose:
            n = len(env.steps)
            print(
                f"  [game {i+1}/{num_games}] seed={seed} steps={n} "
                f"p0_outcome={tr0['outcome']} elapsed={elapsed:.1f}s"
            )
        out.append((tr0, tr1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-a", type=str, required=True, help="player 0 submission .py")
    ap.add_argument("--agent-b", type=str, required=True, help="player 1 submission .py")
    ap.add_argument("--num-games", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--out", type=str, default=None,
                    help="optional JSON path to dump full per-game traces + aggregate")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-game progress lines")
    args = ap.parse_args()

    for f in (args.agent_a, args.agent_b):
        if not Path(f).exists():
            print(f"file not found: {f}", file=sys.stderr)
            return 2

    print(f"agent A (player 0): {args.agent_a}")
    print(f"agent B (player 1): {args.agent_b}")
    print(f"running {args.num_games} games, seed_base={args.seed_base}...")
    print()
    games = run_games(
        args.agent_a, args.agent_b, args.num_games, args.seed_base,
        verbose=not args.quiet,
    )

    traces_p0 = [g[0] for g in games]
    traces_p1 = [g[1] for g in games]

    # We aggregate over TWO windows:
    #   * first 80 turns -- matches our training episode_steps=80; this is the
    #     time horizon the RL policy is actually rewarded over.
    #   * full game -- what the leaderboard scores (500 turns).
    # Comparing the two columns side-by-side tells us whether the "v20 is
    # stockpiling" intuition holds inside the first 80 turns or only kicks
    # in later (which would mean our train horizon is too short to ever
    # learn that pattern).
    windows = [("first_80turns", 0, 80), ("full_game", 0, None)]
    aggs = {}
    for label, lo, hi in windows:
        aggs[label] = {
            "player_0": aggregate(traces_p0, turn_lo=lo, turn_hi=hi),
            "player_1": aggregate(traces_p1, turn_lo=lo, turn_hi=hi),
        }

    for label, _, _ in windows:
        print()
        print(f"################ WINDOW = {label} ################")
        print_summary(f"A = {args.agent_a} (player 0)", aggs[label]["player_0"])
        print_summary(f"B = {args.agent_b} (player 1)", aggs[label]["player_1"])

    if args.out:
        payload = {
            "agent_a": args.agent_a,
            "agent_b": args.agent_b,
            "num_games": args.num_games,
            "seed_base": args.seed_base,
            "aggregate_by_window": aggs,
            "per_game": [
                {"player_0": g[0], "player_1": g[1]}
                for g in games
            ],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1, default=str)
        print(f"[saved] {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
