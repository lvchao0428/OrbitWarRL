"""Frame-by-frame env parity test against kaggle_environments orbit_wars.

Usage:
    python -m orbit_wars_rl.parity.run_env_parity --seed 42 --steps 60

What it does:
  1. Spin up a kaggle orbit_wars env with the given seed.
  2. Take the step-0 obs and build one of our EnvStates from it.
  3. For each turn, generate identical (small, scripted) actions for both
     envs and step them.
  4. After each step, diff our state vs kaggle's obs and print a report row.
  5. At the end, print a summary of which categories of mismatch occurred.

The scripted actions are deliberately *simple* so we can isolate which env
mechanic produces a divergence. Sequence:
    step 0: each player launches a small fleet at the opposite home
    step 1-10: nothing (production accumulates, fleets travel)
    step 11+: more launches
    Throughout: do nothing destructive that would confuse the diff
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# Quiet kaggle_environments noise about OpenSpiel.
import logging
logging.getLogger("kaggle_environments").setLevel(logging.WARNING)

import kaggle_environments  # noqa: E402

from orbit_wars_rl.env import constants  # noqa: E402
from orbit_wars_rl.env.actions import MultiPlayerAction  # noqa: E402
from orbit_wars_rl.env.dynamics import (  # noqa: E402
    launch_fleets, move_and_collide, produce, resolve_combat,
)
from orbit_wars_rl.env.state import EnvState  # noqa: E402
from orbit_wars_rl.parity.kaggle_bridge import (  # noqa: E402
    diff_state, format_diff, kaggle_obs_to_envstate, parse_planet,
)


# --------------------------------------------------------------------------
# Stepping our env at low level (bypassing MultiPlayerAction encoding) so we
# can apply *raw* per-player [(src, angle, ships), ...] move lists identical
# to what kaggle accepts. This avoids any decoder mismatch and isolates the
# physics divergence we actually want to test.
# --------------------------------------------------------------------------


def _empty_multi_action() -> MultiPlayerAction:
    k = constants.MAX_FLEETS_PER_TURN
    return MultiPlayerAction(
        src_idx=jnp.zeros((k,), dtype=jnp.int32),
        dst_idx=jnp.zeros((k,), dtype=jnp.int32),
        pct_bin=jnp.zeros((k,), dtype=jnp.int32),
        emit_mask=jnp.zeros((k,), dtype=jnp.bool_),
    )


def _apply_raw_moves(state: EnvState, moves_per_player: list[list[list]]) -> EnvState:
    """Directly insert each move's fleet, bypassing the multi-action decoder.

    moves_per_player[p] is a list of [src_id, angle, ships] from player p.
    This isolates physics from the policy head, so a diff here means our
    physics is wrong, not our decoder.
    """
    cur = state
    for player, moves in enumerate(moves_per_player):
        for src_id, angle, ships in moves:
            src_id = int(src_id)
            ships = int(ships)
            angle = float(angle)
            if ships <= 0:
                continue
            # Bypass decoder, call _insert_one_fleet directly.
            from orbit_wars_rl.env.dynamics import _insert_one_fleet
            src_x = cur.planet_x[src_id]
            src_y = cur.planet_y[src_id]
            src_radius = cur.planet_radius[src_id]
            cur = _insert_one_fleet(
                cur,
                valid=jnp.bool_(True),
                owner=player,
                ships=jnp.int32(ships),
                src_x=src_x,
                src_y=src_y,
                src_radius=src_radius,
                angle=jnp.float32(angle),
                src_idx=jnp.int32(src_id),
            )
    return cur


def step_our_env(state: EnvState, moves_per_player: list[list[list]]) -> EnvState:
    """Mirrors orbit_wars_rl.env.env.OrbitWarsEnv.step (Kaggle turn order):

      1. Comet expiration         -- skip (no comets)
      2. Comet spawning           -- skip
      3. Fleet launch             -- raw moves
      4. Production               -- produce()
      5. Fleet movement           -- move_and_collide()  (swept-pair, sees planet rotation)
      6. Combat resolution        -- resolve_combat()
      7. Planet rotation (apply)  -- rotate_planets()    (bookkeeping; physics already done)
    """
    from orbit_wars_rl.env.dynamics import rotate_planets
    s = _apply_raw_moves(state, moves_per_player)
    s = produce(s)
    s, hit_planet_idx, hit_mask = move_and_collide(s)
    s = resolve_combat(s, hit_planet_idx, hit_mask)
    s = rotate_planets(s)
    s = s.replace(step=s.step + 1)
    return s


# --------------------------------------------------------------------------
# Action script: deterministic for both kaggle and our env.
# --------------------------------------------------------------------------


def script_action(step: int, kaggle_obs: dict[str, Any]) -> list[list[list]]:
    """Decide what each of P0 and P1 do this turn, given kaggle's view.

    The script:
      - step 0: each player sends a small fleet toward the opposite home
      - step 5: another small fleet
      - step 12: a larger fleet
      - other: nothing
    """
    planets = [parse_planet(p) for p in kaggle_obs.get("planets", [])]
    by_id = {p.id: p for p in planets}

    def home_of(player: int):
        for p in planets:
            if p.owner == player:
                return p
        return None

    p0_home = home_of(0)
    p1_home = home_of(1)

    moves_p0: list[list] = []
    moves_p1: list[list] = []

    if p0_home is None or p1_home is None:
        return [moves_p0, moves_p1]

    def angle_a_to_b(a, b) -> float:
        return math.atan2(b.y - a.y, b.x - a.x)

    if step == 0:
        moves_p0.append([p0_home.id, angle_a_to_b(p0_home, p1_home), 3])
        moves_p1.append([p1_home.id, angle_a_to_b(p1_home, p0_home), 3])
    elif step == 5:
        moves_p0.append([p0_home.id, angle_a_to_b(p0_home, p1_home), 2])
        moves_p1.append([p1_home.id, angle_a_to_b(p1_home, p0_home), 2])
    elif step == 12:
        if p0_home.ships >= 8:
            moves_p0.append([p0_home.id, angle_a_to_b(p0_home, p1_home), 8])
        if p1_home.ships >= 8:
            moves_p1.append([p1_home.id, angle_a_to_b(p1_home, p0_home), 8])

    return [moves_p0, moves_p1]


# --------------------------------------------------------------------------
# Main driver.
# --------------------------------------------------------------------------


@dataclass
class RunSummary:
    total_steps: int
    clean_steps: int
    first_bad_step: int  # -1 if all clean

    # how many steps showed each category of divergence
    n_owner_mismatch: int
    n_ships_mismatch: int
    n_xy_drift_planets: int
    n_xy_drift_fleets: int
    n_fleet_count_mismatch: int
    n_comets_present: int
    max_planet_xy_drift: float

    # at the end, snapshot evidence for the report
    note: str


def run(seed: int, steps: int, verbose: bool) -> RunSummary:
    env = kaggle_environments.make(
        "orbit_wars",
        configuration={"seed": seed, "episodeSteps": max(steps + 2, 10)},
        debug=True,
    )
    init_obs = env.steps[0][0].observation
    print(f"=== kaggle orbit_wars env (seed={seed}) ===")
    print(f"  angular_velocity = {init_obs.get('angular_velocity'):.5f}")
    print(f"  n_planets = {len(init_obs.get('planets', []))}")
    print(f"  comet_planet_ids = {init_obs.get('comet_planet_ids')}")
    print()

    # Build our EnvState from kaggle's step-0 obs.
    our_state = kaggle_obs_to_envstate(init_obs)
    print(f"=== seeded our EnvState from kaggle obs ===")
    print(f"  our planet_mask sum   = {int(np.asarray(our_state.planet_mask).sum())}")
    print(f"  our planet_owner head = {np.asarray(our_state.planet_owner)[:6].tolist()}")
    print()

    summary = RunSummary(
        total_steps=0, clean_steps=0, first_bad_step=-1,
        n_owner_mismatch=0, n_ships_mismatch=0,
        n_xy_drift_planets=0, n_xy_drift_fleets=0,
        n_fleet_count_mismatch=0, n_comets_present=0,
        max_planet_xy_drift=0.0, note="",
    )

    prev_kaggle_obs = init_obs

    for t in range(steps):
        kaggle_obs_pre = env.steps[-1][0].observation
        moves = script_action(t, kaggle_obs_pre)

        # Step kaggle env with the exact same moves.
        env.step([moves[0], moves[1]])
        kaggle_obs_post = env.steps[-1][0].observation

        # Step our env with the exact same moves.
        our_state = step_our_env(our_state, moves)

        d = diff_state(kaggle_obs_post, our_state, prev_kaggle_obs=kaggle_obs_pre)
        if verbose or not d.is_clean():
            print(format_diff(d))

        summary.total_steps += 1
        if d.is_clean():
            summary.clean_steps += 1
        else:
            if summary.first_bad_step < 0:
                summary.first_bad_step = t

        if d.planet_owner_mismatches: summary.n_owner_mismatch += 1
        if d.planet_ships_max_abs_diff > 0: summary.n_ships_mismatch += 1
        if d.planet_xy_max_abs_diff > 0.01: summary.n_xy_drift_planets += 1
        if d.fleet_xy_max_abs_diff > 0.05 and d.n_fleets_kaggle > 0:
            summary.n_xy_drift_fleets += 1
        if d.n_fleets_kaggle != d.n_fleets_ours: summary.n_fleet_count_mismatch += 1
        if d.comet_count > 0: summary.n_comets_present += 1
        summary.max_planet_xy_drift = max(summary.max_planet_xy_drift,
                                          d.planet_xy_max_abs_diff)

        prev_kaggle_obs = kaggle_obs_post

    return summary


def print_summary(s: RunSummary) -> None:
    print()
    print("=" * 60)
    print("PARITY SUMMARY")
    print("=" * 60)
    print(f"  total_steps             = {s.total_steps}")
    print(f"  clean_steps             = {s.clean_steps}")
    print(f"  first_bad_step          = {s.first_bad_step}")
    print()
    print("  per-category divergence counts:")
    print(f"    planet owner          : {s.n_owner_mismatch}")
    print(f"    planet ships          : {s.n_ships_mismatch}")
    print(f"    planet xy (orbit?)    : {s.n_xy_drift_planets}  "
          f"(max abs={s.max_planet_xy_drift:.3f})")
    print(f"    fleet xy              : {s.n_xy_drift_fleets}")
    print(f"    fleet count mismatch  : {s.n_fleet_count_mismatch}")
    print(f"    comets present        : {s.n_comets_present}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=60,
                        help="number of steps to compare (cap at episodeSteps-2)")
    parser.add_argument("--verbose", action="store_true",
                        help="print diff line for every step, not just BAD ones")
    args = parser.parse_args()

    summary = run(args.seed, args.steps, args.verbose)
    print_summary(summary)
    return 0 if summary.clean_steps == summary.total_steps else 1


if __name__ == "__main__":
    sys.exit(main())
