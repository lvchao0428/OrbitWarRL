"""Replay dumper: run agent A vs agent B in kaggle env and print per-step
summary of who-owns-what, total ships, fleets in flight, and the actions
each agent took. Designed to diagnose "v4p2 vs v20 = 0/20, avg_steps=137"
where we need to see HOW the policy is losing (rushed home, bad
lead-target, sun-rays into fleet, etc.) instead of just the final
reward.

Usage on 5090 (cheap, ~10s per game):
    python -m orbit_wars_rl.scripts.replay_dump \
        --agent-a submission_rl_v4p2.py \
        --agent-b submission_v20_0513.py \
        --seed 0 \
        --dump-every 10

Tip:
* ``--dump-every 10`` prints a turn summary every 10 steps + always around
  the last 20 steps before termination.
* ``--max-actions-shown 6`` clips the per-turn action list (longer lists
  rarely add info beyond "agent emits a lot of fleets").
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kaggle_environments import make


def _agent_summary(observation: Dict[str, Any], player: int) -> Dict[str, float | int]:
    """Counts of planets / total ships / fleets for ``player`` from one step obs."""
    planets = observation.get("planets") or []
    fleets = observation.get("fleets") or []
    n_planets = 0
    planet_ships = 0
    for p in planets:
        if int(p[1]) == player:
            n_planets += 1
            planet_ships += int(p[5])
    n_fleets = 0
    fleet_ships = 0
    inbound_to_mine = 0
    inbound_to_opp = 0
    for f in fleets:
        own = int(f[1])
        ships = int(f[6])
        if own == player:
            n_fleets += 1
            fleet_ships += ships
    return {
        "n_planets": n_planets,
        "planet_ships": planet_ships,
        "n_fleets": n_fleets,
        "fleet_ships": fleet_ships,
        "total_ships": planet_ships + fleet_ships,
    }


def _planet_owners(observation: Dict[str, Any]) -> Dict[int, int]:
    return {int(p[0]): int(p[1]) for p in (observation.get("planets") or [])}


def _format_summary(s: Dict[str, float | int]) -> str:
    return (
        f"P={s['n_planets']:>2} "
        f"sP={s['planet_ships']:>4} "
        f"F={s['n_fleets']:>2} "
        f"sF={s['fleet_ships']:>4} "
        f"S={s['total_ships']:>4}"
    )


def _format_action(action: Any, max_shown: int) -> str:
    if action is None:
        return "(None)"
    if not isinstance(action, list):
        return f"<not-a-list: {type(action).__name__}>"
    if not action:
        return "[]"
    if len(action) <= max_shown:
        body = ",".join(_fmt_move(m) for m in action)
    else:
        body = (
            ",".join(_fmt_move(m) for m in action[:max_shown])
            + f",...+{len(action)-max_shown}"
        )
    return f"[{body}]"


def _fmt_move(m: Any) -> str:
    if isinstance(m, list) and len(m) == 3:
        return f"({int(m[0])},{round(float(m[1]),2)},{int(m[2])})"
    return f"<bad:{m}>"


def _planet_diff_str(prev: Dict[int, int], curr: Dict[int, int]) -> str:
    """Compact 'planet 4: -1 -> 0' style changes."""
    changes = []
    for pid in sorted(set(prev.keys()) | set(curr.keys())):
        p, c = prev.get(pid, -2), curr.get(pid, -2)
        if p != c and p != -2 and c != -2:
            changes.append(f"P{pid}:{p}->{c}")
    return " ".join(changes) if changes else ""


def dump_replay(
    agent_a: str,
    agent_b: str,
    seed: Optional[int],
    dump_every: int,
    max_actions_shown: int,
    tail_steps: int,
) -> None:
    cfg = {}
    if seed is not None:
        cfg["seed"] = seed
    env = make("orbit_wars", configuration=cfg, debug=False)
    t0 = time.time()
    env.run([agent_a, agent_b])
    elapsed = time.time() - t0

    n_steps = len(env.steps)
    final_step = env.steps[-1]
    rewards: List[Any] = [s.get("reward") for s in final_step]
    status: List[Any] = [s.get("status") for s in final_step]

    print(f"agent A: {agent_a}")
    print(f"agent B: {agent_b}")
    print(f"seed   : {seed}")
    print(f"steps  : {n_steps}   (elapsed {elapsed:.1f}s)")
    print(f"result : a={rewards[0]} b={rewards[1]}  status={status}")

    if rewards[0] is not None and rewards[1] is not None:
        try:
            ra, rb = float(rewards[0]), float(rewards[1])
            outcome = "A wins" if ra > rb else ("B wins" if rb > ra else "draw")
        except (TypeError, ValueError):
            outcome = "?"
        print(f"outcome: {outcome}")

    print()
    print(f"{'step':>5} | {'A (player 0)':<26} | {'B (player 1)':<26} | "
          f"{'action A':<30} | {'action B':<30} | events")
    print("-" * 150)

    prev_owners: Dict[int, int] = {}
    dump_threshold = max(0, n_steps - tail_steps)
    for t, step_entries in enumerate(env.steps):
        if (t < dump_threshold) and (t % dump_every != 0) and (t != 0):
            continue
        obs_a = step_entries[0].get("observation", {})
        # In kaggle's 2p envs each agent sees its own obs slice, but for
        # planets/fleets the schema is shared (player field differs).
        obs_b = step_entries[1].get("observation", {})
        sa = _agent_summary(obs_a, player=0)
        sb = _agent_summary(obs_b, player=1)
        # actions are visible only on the entries that received them, i.e.
        # the env stores each agent's *previous* action under .action.
        act_a = step_entries[0].get("action")
        act_b = step_entries[1].get("action")
        curr_owners = _planet_owners(obs_a)
        owner_changes = _planet_diff_str(prev_owners, curr_owners) if prev_owners else ""

        marker = " "
        if t >= dump_threshold:
            marker = "*"  # mark "tail dump" rows so they stand out

        print(
            f"{marker}{t:>4} | {_format_summary(sa):<26} | {_format_summary(sb):<26} | "
            f"{_format_action(act_a, max_actions_shown):<30} | "
            f"{_format_action(act_b, max_actions_shown):<30} | "
            f"{owner_changes}"
        )
        prev_owners = curr_owners

    print("-" * 150)
    print(f"steps={n_steps} (* = tail dump)")
    print(f"final result: a={rewards[0]} b={rewards[1]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-a", type=str, required=True)
    ap.add_argument("--agent-b", type=str, required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--dump-every",
        type=int,
        default=10,
        help="dump a summary row every N steps in the bulk of the game",
    )
    ap.add_argument(
        "--tail-steps",
        type=int,
        default=25,
        help="always dump the last K steps regardless of dump-every",
    )
    ap.add_argument(
        "--max-actions-shown",
        type=int,
        default=6,
        help="clip per-turn action list to first K moves",
    )
    args = ap.parse_args()

    for f in (args.agent_a, args.agent_b):
        if not Path(f).exists():
            print(f"file not found: {f}", file=sys.stderr)
            return 2

    dump_replay(
        args.agent_a, args.agent_b,
        seed=args.seed,
        dump_every=args.dump_every,
        max_actions_shown=args.max_actions_shown,
        tail_steps=args.tail_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
