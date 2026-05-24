"""Expert replay analyzer — extract behaviour signatures from top-N kaggle
JSON replays (no game execution; pure log parsing).

Companion to ``replay_analyze.py`` which RUNS games. This one parses
existing Kaggle episode JSONs (downloaded via meta-kaggle / kaggle API)
and computes the same family of metrics on per-turn time series for
each player in each episode.

Day 4 Track 1.5 deliverable. Designed for the user-supplied
``top10_episodes_*/episodes/episodes/*.json`` archives.

Per-turn metrics (per player):
  * ships_per_fleet_mean         — mean ships across this turn's launches
  * ships_per_fleet_max          — max across this turn's launches (decisive strike)
  * num_launches                 — count of fleet emissions
  * ships_total_launched         — sum of ships across this turn's launches
  * garrison_total               — total ships on player's planets, turn-start
  * planet_count                 — number of planets owned, turn-start
  * production_share             — fraction of global production capacity owned
  * fleet_count_alive            — total fleets in flight owned by player

Episode-level summaries:
  * winner_id, final_rewards, final_statuses
  * episode length (turns played)
  * per-player aggregate: launches_total, ships_total, mean_launch_size,
    p50/p95/p99 launch sizes, decisive_turn (turn of largest single launch),
    stockpile_curve (10-pct percentile of turns)

Usage:
    python -m orbit_wars_rl.scripts.analyze_expert_replay \
        --replay-glob 'top10_episodes_*/episodes/episodes/75873267.json' \
        --out logs/expert_75873267.json \
        --print-summary

Multiple files are processed independently; their results are written
as a list under the JSON ``episodes`` key.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Action format in kaggle replay JSON:
#   [src_planet_id, angle_rad, ships_int] per fleet, list per turn
# Planet row: [id, owner, x, y, radius, ships, production]


def _planet_ships_for_owner(obs: Dict[str, Any], owner: int) -> int:
    return sum(int(p[5]) for p in (obs.get("planets") or []) if int(p[1]) == owner)


def _planet_prod_for_owner(obs: Dict[str, Any], owner: int) -> int:
    return sum(int(p[6]) for p in (obs.get("planets") or []) if int(p[1]) == owner)


def _planet_count_for_owner(obs: Dict[str, Any], owner: int) -> int:
    return sum(1 for p in (obs.get("planets") or []) if int(p[1]) == owner)


def _total_prod(obs: Dict[str, Any]) -> int:
    return sum(int(p[6]) for p in (obs.get("planets") or []))


def _fleet_count_for_owner(obs: Dict[str, Any], owner: int) -> int:
    # Fleet row in obs: [id, owner, src_planet_id, dst_x, dst_y, ships, ...] —
    # exact columns vary by orbit_wars version. We only need owner = field[1].
    fleets = obs.get("fleets") or []
    return sum(1 for f in fleets if len(f) > 1 and int(f[1]) == owner)


# -------------------------- per-episode analysis --------------------------


def analyze_episode(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        replay = json.load(f)

    cfg = replay.get("configuration", {})
    episode_steps = int(cfg.get("episodeSteps", 500))
    num_players = len(replay.get("rewards", []))
    steps = replay.get("steps") or []

    # Per-turn per-player metric series, initialized to empty.
    series: List[List[Dict[str, Any]]] = [
        [] for _ in range(num_players)
    ]

    # Track launches (raw ship counts) over the whole episode.
    all_launch_sizes: List[List[int]] = [[] for _ in range(num_players)]
    decisive_strike: List[Tuple[int, int]] = [(0, 0) for _ in range(num_players)]

    for t, step in enumerate(steps):
        # Skip incomplete (truncated) turn rows.
        if not step or len(step) < num_players:
            break

        # obs is shared content; each player has their own perspective but
        # planets/fleets are global, so player 0's obs is sufficient.
        obs0 = step[0].get("observation", {}) or {}
        total_prod = max(1, _total_prod(obs0))

        for p in range(num_players):
            cell = step[p] or {}
            obs = cell.get("observation", {}) or obs0
            action = cell.get("action") or []

            ships_list = [int(a[2]) for a in action if len(a) >= 3]
            launches = len(ships_list)
            ships_launched = sum(ships_list)
            spf_mean = (ships_launched / launches) if launches else 0.0
            spf_max = max(ships_list) if ships_list else 0
            all_launch_sizes[p].extend(ships_list)

            if spf_max > decisive_strike[p][1]:
                decisive_strike[p] = (t, spf_max)

            row = {
                "turn": t,
                "num_launches": launches,
                "ships_launched": ships_launched,
                "ships_per_fleet_mean": round(spf_mean, 2),
                "ships_per_fleet_max": spf_max,
                "garrison_total": _planet_ships_for_owner(obs, p),
                "planet_count": _planet_count_for_owner(obs, p),
                "production_share": round(_planet_prod_for_owner(obs, p) / total_prod, 3),
                "fleet_count_alive": _fleet_count_for_owner(obs, p),
            }
            series[p].append(row)

    # ---------- per-player aggregate ----------
    per_player: List[Dict[str, Any]] = []
    for p in range(num_players):
        sizes = all_launch_sizes[p]
        sizes_sorted = sorted(sizes)
        def _pct(q: float) -> float:
            if not sizes_sorted:
                return 0.0
            k = max(0, min(len(sizes_sorted) - 1, int(round(q * (len(sizes_sorted) - 1)))))
            return float(sizes_sorted[k])

        # Stockpile curve: garrison at deciles of the episode.
        decile_garrison: List[int] = []
        n = len(series[p])
        if n > 0:
            for d in range(11):  # 0%, 10%, ..., 100%
                idx = min(n - 1, int(round(d * (n - 1) / 10)))
                decile_garrison.append(series[p][idx]["garrison_total"])

        # 0-emit rate.
        n_zero = sum(1 for row in series[p] if row["num_launches"] == 0)
        zero_emit_rate = (n_zero / n) if n else 0.0

        # First-80 vs full-game windowed metrics, matching replay_analyze.py.
        def _window(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not rows:
                return dict(turns=0, zero_emit_rate=0.0, spf_mean=0.0,
                            mean_garrison=0.0, mean_planets=0.0,
                            mean_production_share=0.0)
            launches = [r for r in rows if r["num_launches"] > 0]
            spf_all_sizes: List[int] = []
            for r in launches:
                # reconstruct from mean*launches (we did not keep per-launch list)
                spf_all_sizes.extend([int(r["ships_per_fleet_mean"])] * r["num_launches"])
            n_zero_w = sum(1 for r in rows if r["num_launches"] == 0)
            return dict(
                turns=len(rows),
                zero_emit_rate=round(n_zero_w / len(rows), 3),
                spf_mean=round(statistics.mean(spf_all_sizes), 1) if spf_all_sizes else 0.0,
                spf_max_global=max(spf_all_sizes) if spf_all_sizes else 0,
                mean_garrison=round(statistics.mean(r["garrison_total"] for r in rows), 1),
                mean_planets=round(statistics.mean(r["planet_count"] for r in rows), 2),
                mean_production_share=round(
                    statistics.mean(r["production_share"] for r in rows), 3
                ),
            )

        first80 = _window(series[p][:80])
        full = _window(series[p])

        per_player.append(dict(
            player=p,
            final_reward=int(replay.get("rewards", [0]*num_players)[p]),
            final_status=str(replay.get("statuses", [""]*num_players)[p]),
            total_launches=len(sizes),
            total_ships_launched=sum(sizes),
            mean_launch_size=round(statistics.mean(sizes), 1) if sizes else 0.0,
            p50_launch_size=int(_pct(0.50)),
            p95_launch_size=int(_pct(0.95)),
            p99_launch_size=int(_pct(0.99)),
            max_launch_size=max(sizes) if sizes else 0,
            decisive_turn=decisive_strike[p][0],
            decisive_ships=decisive_strike[p][1],
            zero_emit_rate=round(zero_emit_rate, 3),
            garrison_decile_curve=decile_garrison,
            first80=first80,
            full=full,
        ))

    return dict(
        path=str(path),
        episode_id=replay.get("id"),
        configuration=cfg,
        num_players=num_players,
        episode_length=len(steps),
        episode_steps_cfg=episode_steps,
        final_rewards=replay.get("rewards"),
        final_statuses=replay.get("statuses"),
        per_player=per_player,
        per_turn_series=series,
    )


# ---------------------------- pretty printer -----------------------------


def print_summary(ep: Dict[str, Any], turn_series: bool = False) -> None:
    print(f"\n=== Episode {ep['episode_id']} ({Path(ep['path']).name}) ===")
    print(f"  config: {ep['configuration']}")
    print(f"  players: {ep['num_players']}  turns played: {ep['episode_length']}")
    print(f"  final rewards : {ep['final_rewards']}")
    print(f"  final statuses: {ep['final_statuses']}")
    print()
    print(f"  {'p':>2} {'reward':>6} {'launches':>8} {'mean':>5} {'p50':>4} {'p95':>4} {'p99':>4} {'max':>5} {'dt':>4} {'z0':>5} {'g/p_full':>10}")
    for pp in ep["per_player"]:
        print(
            f"  {pp['player']:>2} {pp['final_reward']:>+6d} {pp['total_launches']:>8d} "
            f"{pp['mean_launch_size']:>5.0f} {pp['p50_launch_size']:>4d} "
            f"{pp['p95_launch_size']:>4d} {pp['p99_launch_size']:>4d} "
            f"{pp['max_launch_size']:>5d} {pp['decisive_turn']:>4d} "
            f"{pp['zero_emit_rate']:>5.2f} "
            f"{pp['full']['mean_garrison']:>5.0f}/{pp['full']['mean_planets']:.2f}"
        )
    print()
    print("  garrison decile curve (turn% : ships), per player:")
    for pp in ep["per_player"]:
        c = pp["garrison_decile_curve"]
        c_str = " ".join(f"{v:>4d}" for v in c)
        print(f"    p{pp['player']} (R={pp['final_reward']:+d}): {c_str}")

    print()
    print("  first-80 vs full window comparison:")
    print(f"    {'p':>2} {'window':>8} {'spf_mean':>9} {'spf_max':>8} {'z0_rate':>8} {'mean_garr':>10} {'planets':>8} {'prod_share':>11}")
    for pp in ep["per_player"]:
        for label, w in [("first80", pp["first80"]), ("full", pp["full"])]:
            print(
                f"    {pp['player']:>2} {label:>8} {w['spf_mean']:>9.1f} "
                f"{w.get('spf_max_global', 0):>8d} {w['zero_emit_rate']:>8.2f} "
                f"{w['mean_garrison']:>10.1f} {w['mean_planets']:>8.2f} "
                f"{w['mean_production_share']:>11.3f}"
            )

    if turn_series:
        # Print a sparse trace for the winner.
        winner = max(ep["per_player"], key=lambda x: x["final_reward"])
        rows = ep["per_turn_series"][winner["player"]]
        print()
        print(f"  turn-by-turn trace for winner p{winner['player']} (every 50 turns):")
        print(f"    {'turn':>4} {'launches':>8} {'shipsL':>7} {'spfMean':>7} {'spfMax':>7} {'garr':>5} {'plnt':>5} {'prod%':>6} {'fleet':>5}")
        for row in rows[::50]:
            print(
                f"    {row['turn']:>4d} {row['num_launches']:>8d} "
                f"{row['ships_launched']:>7d} {row['ships_per_fleet_mean']:>7.1f} "
                f"{row['ships_per_fleet_max']:>7d} {row['garrison_total']:>5d} "
                f"{row['planet_count']:>5d} {row['production_share']:>6.2f} "
                f"{row['fleet_count_alive']:>5d}"
            )


# ----------------------------- entry point -------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--replay-glob", action="append", required=True,
                   help="glob pattern for replay JSONs; can be repeated")
    p.add_argument("--out", default=None, help="optional path to dump JSON results")
    p.add_argument("--print-summary", action="store_true")
    p.add_argument("--print-trace", action="store_true",
                   help="also print per-turn trace for the winner")
    p.add_argument("--max-files", type=int, default=10,
                   help="hard cap on files analyzed (large dataset safety)")
    args = p.parse_args(argv)

    paths: List[Path] = []
    for pattern in args.replay_glob:
        for s in sorted(glob.glob(pattern)):
            paths.append(Path(s))
    paths = paths[: args.max_files]
    if not paths:
        print("ERR: no replay files matched", file=sys.stderr)
        return 2

    results: List[Dict[str, Any]] = []
    for path in paths:
        try:
            ep = analyze_episode(path)
        except Exception as e:  # noqa: BLE001
            print(f"WARN: failed to parse {path}: {e}", file=sys.stderr)
            continue
        results.append(ep)
        if args.print_summary:
            print_summary(ep, turn_series=args.print_trace)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dict(episodes=results), f, indent=2)
        print(f"\nwrote {len(results)} episode(s) to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
