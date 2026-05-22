"""Quick parity sanity check against ``kaggle_environments``.

The MVP env intentionally drops planet rotation, comets, and continuous-segment
collision. So we don't expect byte equivalence. Instead this script:

1. Spawns a kaggle env with seed S.
2. Pulls its planet positions, productions, ship counts.
3. Builds an equivalent MVP state with those exact positions (static).
4. Runs a fixed action script for ~20 turns in both envs.
5. Reports per-step diffs on shared invariants:
     - planet ownership distribution
     - per-planet ship count (after combat)
     - total player ship counts
   on planets that are *not* orbiting and *not* comets in the kaggle env.

This catches gross bugs (off-by-one in production order, wrong combat math,
missed sun collision) without forcing pixel parity.

Usage:
    python -m orbit_wars_rl.scripts.parity_check --seeds 0 1 2 3 4 --turns 20
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from orbit_wars_rl.env import OrbitWarsEnv, constants
from orbit_wars_rl.env.actions import PlayerAction
from orbit_wars_rl.env.state import EnvState


def _build_mvp_state_from_kaggle(kobs, num_players: int = 2) -> EnvState:
    """Map a kaggle observation onto our MVP state for static-only planets."""
    planets = list(kobs.get("planets") or [])
    n = len(planets)
    if n > constants.MAX_PLANETS:
        raise RuntimeError(f"kaggle env produced {n} planets > MAX_PLANETS={constants.MAX_PLANETS}")

    owners = np.full((constants.MAX_PLANETS,), -2, dtype=np.int8)
    xs = np.zeros((constants.MAX_PLANETS,), dtype=np.float32)
    ys = np.zeros((constants.MAX_PLANETS,), dtype=np.float32)
    rads = np.zeros((constants.MAX_PLANETS,), dtype=np.float32)
    ships = np.zeros((constants.MAX_PLANETS,), dtype=np.int32)
    prods = np.zeros((constants.MAX_PLANETS,), dtype=np.int32)
    mask = np.zeros((constants.MAX_PLANETS,), dtype=np.bool_)

    for i, p in enumerate(planets):
        owners[i] = np.int8(p[1])
        xs[i] = np.float32(p[2])
        ys[i] = np.float32(p[3])
        rads[i] = np.float32(p[4])
        ships[i] = np.int32(p[5])
        prods[i] = np.int32(p[6])
        mask[i] = True

    return EnvState(
        planet_owner=jnp.asarray(owners),
        planet_x=jnp.asarray(xs),
        planet_y=jnp.asarray(ys),
        planet_radius=jnp.asarray(rads),
        planet_ships=jnp.asarray(ships),
        planet_prod=jnp.asarray(prods),
        planet_mask=jnp.asarray(mask),
        fleet_owner=jnp.full((constants.MAX_FLEETS,), -2, dtype=jnp.int8),
        fleet_x=jnp.zeros((constants.MAX_FLEETS,), dtype=jnp.float32),
        fleet_y=jnp.zeros((constants.MAX_FLEETS,), dtype=jnp.float32),
        fleet_angle=jnp.zeros((constants.MAX_FLEETS,), dtype=jnp.float32),
        fleet_ships=jnp.zeros((constants.MAX_FLEETS,), dtype=jnp.int32),
        fleet_mask=jnp.zeros((constants.MAX_FLEETS,), dtype=jnp.bool_),
        step=jnp.int32(0),
        done=jnp.bool_(False),
        rng=jnp.asarray([0, 0], dtype=jnp.uint32),
        angular_velocity=jnp.float32(0.0),
        planet_orbit_radius=jnp.zeros((constants.MAX_PLANETS,), dtype=jnp.float32),
        planet_orbit_phase=jnp.zeros((constants.MAX_PLANETS,), dtype=jnp.float32),
        planet_is_orbiting=jnp.zeros((constants.MAX_PLANETS,), dtype=jnp.bool_),
    )


def _angle_to_nearest_enemy_or_neutral(planets, my_planet_id: int, player: int) -> Tuple[int, float]:
    me = planets[my_planet_id]
    targets = [p for p in planets if p[0] != my_planet_id and p[1] != player]
    if not targets:
        return -1, 0.0
    targets.sort(key=lambda p: math.hypot(p[2] - me[2], p[3] - me[3]))
    t = targets[0]
    return int(t[0]), math.atan2(t[3] - me[3], t[2] - me[2])


def _scripted_kaggle_agent(obs, _config, player_id: int):
    planets = list(obs.get("planets") or [])
    me_first = next((p for p in planets if p[1] == player_id and p[5] > 4), None)
    if me_first is None:
        return []
    tid, angle = _angle_to_nearest_enemy_or_neutral(planets, planets.index(me_first), player_id)
    if tid == -1:
        return []
    ships = max(1, me_first[5] // 2)
    return [[me_first[0], angle, ships]]


def _scripted_mvp_action(state: EnvState, player: int) -> PlayerAction:
    mask = np.asarray(state.planet_mask)
    owners = np.asarray(state.planet_owner)
    ships = np.asarray(state.planet_ships)
    xs = np.asarray(state.planet_x)
    ys = np.asarray(state.planet_y)

    mine = [i for i in range(len(mask)) if mask[i] and owners[i] == player and ships[i] > 4]
    if not mine:
        return PlayerAction(src_idx=jnp.int32(0), dst_idx=jnp.int32(0), pct_bin=jnp.int32(0))
    src = mine[0]
    targets = [
        i for i in range(len(mask))
        if mask[i] and owners[i] != player and i != src
    ]
    if not targets:
        return PlayerAction(src_idx=jnp.int32(src), dst_idx=jnp.int32(0), pct_bin=jnp.int32(0))
    targets.sort(key=lambda i: math.hypot(xs[i] - xs[src], ys[i] - ys[src]))
    dst = targets[0]
    return PlayerAction(
        src_idx=jnp.int32(src),
        dst_idx=jnp.int32(dst),
        pct_bin=jnp.int32(1),
    )


def _kaggle_orbiting_or_comet_ids(kobs, ang_vel: float) -> set[int]:
    out: set[int] = set()
    for cgroup in kobs.get("comets") or []:
        for pid in cgroup.get("planet_ids") or []:
            out.add(int(pid))
    for cid in kobs.get("comet_planet_ids") or []:
        out.add(int(cid))
    init_planets = kobs.get("initial_planets") or []
    if abs(ang_vel) < 1e-12:
        return out
    for ip in init_planets:
        pid, _own, x, y, r, *_ = ip
        d = math.hypot(x - constants.SUN_X, y - constants.SUN_Y)
        if d + r < constants.SUN_RADIUS + 40.0:
            out.add(int(pid))
    return out


def run_parity(seed: int, turns: int = 20) -> dict:
    try:
        from kaggle_environments import make
    except ImportError as e:
        raise SystemExit(f"kaggle_environments not installed: {e}")

    kenv = make("orbit_wars", configuration={"seed": seed}, debug=False)
    init = kenv.reset(num_agents=2)
    kobs0 = init[0].observation
    ang_vel = float(kobs0.get("angular_velocity", 0.0) or 0.0)
    skip_ids = _kaggle_orbiting_or_comet_ids(kobs0, ang_vel)

    mvp_state = _build_mvp_state_from_kaggle(kobs0)
    mvp_env = OrbitWarsEnv(num_groups=4, episode_steps=500)

    diffs_per_turn: list[dict] = []
    cur_obs = init

    for t in range(turns):
        kobs_p0 = cur_obs[0].observation
        kobs_p1 = cur_obs[1].observation

        action_0 = _scripted_kaggle_agent(kobs_p0, None, 0)
        action_1 = _scripted_kaggle_agent(kobs_p1, None, 1)
        cur_obs = kenv.step([action_0, action_1])

        a_mvp_0 = _scripted_mvp_action(mvp_state, 0)
        a_mvp_1 = _scripted_mvp_action(mvp_state, 1)
        mvp_state, _ = mvp_env.step(mvp_state, (a_mvp_0, a_mvp_1))

        kobs = cur_obs[0].observation
        kplanets = list(kobs.get("planets") or [])

        diff_count = 0
        max_ship_diff = 0
        compared = 0
        for kp in kplanets:
            pid = int(kp[0])
            if pid in skip_ids:
                continue
            if pid >= constants.MAX_PLANETS:
                continue
            if not bool(mvp_state.planet_mask[pid]):
                continue
            k_owner, k_ships = int(kp[1]), int(kp[5])
            m_owner = int(mvp_state.planet_owner[pid])
            m_ships = int(mvp_state.planet_ships[pid])
            compared += 1
            if k_owner != m_owner:
                diff_count += 1
            if abs(k_ships - m_ships) > 1:
                diff_count += 1
                max_ship_diff = max(max_ship_diff, abs(k_ships - m_ships))

        diffs_per_turn.append(dict(turn=t + 1, compared=compared, diffs=diff_count, max_ship_diff=max_ship_diff))

        if cur_obs[0].status == "DONE":
            break

    return dict(seed=seed, num_turns=len(diffs_per_turn), per_turn=diffs_per_turn, skip_ids=sorted(skip_ids))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--turns", type=int, default=20)
    args = ap.parse_args()

    overall_ok = True
    for s in args.seeds:
        try:
            r = run_parity(s, turns=args.turns)
        except Exception as exc:
            print(f"[seed {s}] FAILED to run: {exc}")
            overall_ok = False
            continue
        total_diffs = sum(d["diffs"] for d in r["per_turn"])
        max_total = max((d["max_ship_diff"] for d in r["per_turn"]), default=0)
        print(
            f"[seed {s}] turns={r['num_turns']} skipped_planets={len(r['skip_ids'])} "
            f"total_per_planet_diffs={total_diffs} max_ship_diff={max_total}"
        )
        if total_diffs > 0 and max_total > 5:
            overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
