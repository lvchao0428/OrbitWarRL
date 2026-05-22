"""Run a single agent through a gauntlet of opponents and print a summary table.

Example:
    python -m orbit_wars_rl.scripts.h2h_gauntlet \
        --agent submission_rl_v4p2.py \
        --opponents submission_rl_v1.py submission_rl_v3p2_peak.py submission_v20_0513.py \
        --num-games 10 \
        --seeds 0,1,2,3,4,5,6,7,8,9

Notes:
* Plays each (agent vs opp) match BOTH as player A and player B (color symmetric)
  to neutralise any first-player advantage in the env. Reports
  combined win-rate.
* Per-seed determinism: if --seeds is given the same seed is used for the
  A/B and B/A variants, giving paired comparisons.
* Re-uses ``orbit_wars_rl.scripts.h2h_local._run_one_game`` so behaviour is
  identical to the existing 1-vs-1 script.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

from orbit_wars_rl.scripts.h2h_local import _run_one_game


def _parse_seeds(tokens: List[str] | None) -> List[int]:
    if not tokens:
        return []
    out: List[int] = []
    for tok in tokens:
        for part in tok.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(int(part))
            except ValueError:
                print(f"--seeds: cannot parse '{part}' as int", file=sys.stderr)
                raise
    return out


def _play_match(
    agent: str,
    opp: str,
    num_games: int,
    seeds: List[int],
) -> dict:
    """Play ``num_games`` per side (so 2*num_games total) and aggregate."""
    a_wins = 0  # 'agent' as A wins
    b_wins = 0  # 'agent' as B wins
    a_losses = 0
    b_losses = 0
    draws = 0
    errors = 0
    total_steps = 0

    for color, (left, right) in enumerate([(agent, opp), (opp, agent)]):
        for i in range(num_games):
            seed = seeds[i % len(seeds)] if seeds else None
            try:
                r = _run_one_game(left, right, seed=seed)
            except Exception as exc:  # noqa: BLE001
                print(f"  {Path(opp).name} game {i} (color={color}): error {exc}")
                errors += 1
                continue
            ra, rb = r["rewards"]
            total_steps += r["num_steps"]
            if ra is None or rb is None:
                errors += 1
                continue
            agent_reward = ra if color == 0 else rb
            opp_reward = rb if color == 0 else ra
            if agent_reward > opp_reward:
                if color == 0:
                    a_wins += 1
                else:
                    b_wins += 1
            elif opp_reward > agent_reward:
                if color == 0:
                    a_losses += 1
                else:
                    b_losses += 1
            else:
                draws += 1

    return {
        "wins": a_wins + b_wins,
        "wins_as_A": a_wins,
        "wins_as_B": b_wins,
        "losses": a_losses + b_losses,
        "draws": draws,
        "errors": errors,
        "total": 2 * num_games,
        "avg_steps": total_steps / max(1, 2 * num_games - errors),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=str, required=True,
                    help="path to the candidate submission (.py)")
    ap.add_argument("--opponents", type=str, nargs="+", required=True,
                    help="paths to opponent submission files (.py)")
    ap.add_argument("--num-games", type=int, default=5,
                    help="games per color side (total games per opp = 2*num)")
    ap.add_argument("--seeds", type=str, nargs="*", default=None,
                    help="optional fixed seeds (space- or comma-separated)")
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)

    for p in [args.agent, *args.opponents]:
        if not Path(p).exists():
            print(f"file not found: {p}", file=sys.stderr)
            return 2

    print(f"agent: {args.agent}")
    print(f"games per opponent: 2 * {args.num_games} (both colors)")
    if seeds:
        print(f"seeds: {seeds}")
    print()

    rows = []
    for opp in args.opponents:
        t0 = time.time()
        res = _play_match(args.agent, opp, args.num_games, seeds)
        elapsed = time.time() - t0
        rows.append((opp, res, elapsed))
        # Live print.
        wr = res["wins"] / max(1, res["total"] - res["errors"])
        print(
            f"vs {Path(opp).name:<35s} "
            f"W={res['wins']:>2}/{res['total']:<2}  "
            f"L={res['losses']:>2}  D={res['draws']:>2}  "
            f"asA={res['wins_as_A']}  asB={res['wins_as_B']}  "
            f"errs={res['errors']}  "
            f"WR={wr:.2f}  avg_steps={res['avg_steps']:.0f}  "
            f"({elapsed:.1f}s)"
        )

    print()
    print("=" * 70)
    print("summary")
    print("=" * 70)
    total_wins = sum(r[1]["wins"] for r in rows)
    total_played = sum(r[1]["total"] - r[1]["errors"] for r in rows)
    print(f"overall: {total_wins}/{total_played} = "
          f"{total_wins/max(1,total_played):.3f} WR across {len(rows)} opponents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
