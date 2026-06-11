"""Bridge between kaggle_environments orbit_wars obs and our ``EnvState``.

This module's *only* job is to convert. It does not run RL code, does not
import flax, and does not touch model weights. It exists so parity tests can:

  1. Create a kaggle env, grab its initial obs.
  2. Convert that obs to one of our ``EnvState``.
  3. Step both envs with identical actions, dump both states.
  4. Compare field by field.

Anything we cannot represent yet (orbital rotation angle, comet trajectories,
fleet ids -> our slot ids) is documented inline so the parity report can call
out exactly what's missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.state import EnvState


# --------------------------------------------------------------------------
# kaggle obs schema reminders (from a real env, kaggle_environments==1.29.1)
#
# obs["planets"]: list[list[id, owner, x, y, radius, ships, production]]
# obs["fleets"]:  list[list[id, owner, x, y, angle, from_planet_id, ships]]
# obs["initial_planets"]: same shape as planets, snapshot at step 0 with
#     home planets shown as owner=-1 (their pre-assignment state).
# obs["angular_velocity"]: float, rad/turn for the orbiting group
# obs["next_fleet_id"]: int (we ignore)
# obs["comets"]: list[dict(planet_ids, paths, path_index)]; empty before step 50
# obs["comet_planet_ids"]: list[int]; ids that are comets
# obs["player"]: 0 or 1 (perspective)
# obs["step"]: int
# --------------------------------------------------------------------------


@dataclass
class KagglePlanet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


@dataclass
class KaggleFleet:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


def parse_planet(row: list) -> KagglePlanet:
    pid, owner, x, y, radius, ships, prod = row
    return KagglePlanet(int(pid), int(owner), float(x), float(y),
                        float(radius), int(ships), int(prod))


def parse_fleet(row: list) -> KaggleFleet:
    fid, owner, x, y, angle, src, ships = row
    return KaggleFleet(int(fid), int(owner), float(x), float(y),
                       float(angle), int(src), int(ships))


def kaggle_obs_to_envstate(obs: dict[str, Any]) -> EnvState:
    """Build one of our ``EnvState`` from a kaggle env observation.

    Slot-id mapping: kaggle planet id N goes into our planet_owner[N], etc.
    The kaggle env keeps ids stable across the whole episode, so this works.

    Fleet slot-id mapping: kaggle fleet id N goes into our fleet_owner[N].
    Same stability assumption.
    """
    planets = [parse_planet(p) for p in obs.get("planets", [])]
    fleets = [parse_fleet(f) for f in obs.get("fleets", [])]

    p_n = len(planets)
    if p_n > constants.MAX_PLANETS:
        raise ValueError(
            f"kaggle env returned {p_n} planets, but our MAX_PLANETS = "
            f"{constants.MAX_PLANETS}. Bump constants.MAX_PLANETS."
        )

    # planets must be at slot == id; kaggle assigns ids 0..n-1 in order
    planet_owner = np.full((constants.MAX_PLANETS,), constants.PADDING_OWNER, dtype=np.int8)
    planet_x = np.zeros((constants.MAX_PLANETS,), dtype=np.float32)
    planet_y = np.zeros((constants.MAX_PLANETS,), dtype=np.float32)
    planet_radius = np.zeros((constants.MAX_PLANETS,), dtype=np.float32)
    planet_ships = np.zeros((constants.MAX_PLANETS,), dtype=np.int32)
    planet_prod = np.zeros((constants.MAX_PLANETS,), dtype=np.int32)
    planet_mask = np.zeros((constants.MAX_PLANETS,), dtype=bool)

    # Per-planet orbital fields. We derive these from the live (x, y) and the
    # episode-wide angular_velocity exposed in the kaggle obs.
    angular_velocity = float(obs.get("angular_velocity", 0.0))
    planet_orbit_radius = np.zeros((constants.MAX_PLANETS,), dtype=np.float32)
    planet_orbit_phase = np.zeros((constants.MAX_PLANETS,), dtype=np.float32)
    planet_is_orbiting = np.zeros((constants.MAX_PLANETS,), dtype=bool)

    for p in planets:
        slot = p.id
        if slot != planets.index(p):
            # kaggle env is supposed to assign ids in order; warn if not
            pass
        planet_owner[slot] = p.owner
        planet_x[slot] = p.x
        planet_y[slot] = p.y
        planet_radius[slot] = p.radius
        planet_ships[slot] = p.ships
        planet_prod[slot] = p.production
        planet_mask[slot] = True

        # Distance to sun centre + initial phase (from +x axis).
        dx = p.x - constants.SUN_X
        dy = p.y - constants.SUN_Y
        orb_r = math.hypot(dx, dy)
        planet_orbit_radius[slot] = orb_r
        planet_orbit_phase[slot] = math.atan2(dy, dx)
        # Kaggle rule: orbits iff orbital_radius + planet_radius < 50.
        planet_is_orbiting[slot] = (orb_r + p.radius) < constants.ORBIT_RADIUS_LIMIT

    f_n = len(fleets)
    if f_n > constants.MAX_FLEETS:
        raise ValueError(
            f"kaggle env returned {f_n} fleets, but our MAX_FLEETS = "
            f"{constants.MAX_FLEETS}. Bump constants.MAX_FLEETS."
        )

    fleet_owner = np.full((constants.MAX_FLEETS,), constants.PADDING_OWNER, dtype=np.int8)
    fleet_x = np.zeros((constants.MAX_FLEETS,), dtype=np.float32)
    fleet_y = np.zeros((constants.MAX_FLEETS,), dtype=np.float32)
    fleet_angle = np.zeros((constants.MAX_FLEETS,), dtype=np.float32)
    fleet_ships = np.zeros((constants.MAX_FLEETS,), dtype=np.int32)
    fleet_mask = np.zeros((constants.MAX_FLEETS,), dtype=bool)

    # We CANNOT just slot fleets by kaggle id -- kaggle fleet ids grow monotonically
    # across the entire episode (per next_fleet_id) and quickly exceed MAX_FLEETS.
    # Instead, pack live fleets into the first len(fleets) slots in order.
    # This means we lose the mapping from "kaggle id" to "our slot" -- callers
    # who need that mapping must build it themselves.
    for i, f in enumerate(fleets):
        fleet_owner[i] = f.owner
        fleet_x[i] = f.x
        fleet_y[i] = f.y
        fleet_angle[i] = f.angle
        fleet_ships[i] = f.ships
        fleet_mask[i] = True

    step = int(obs.get("step", 0))

    return EnvState(
        planet_owner=jnp.asarray(planet_owner),
        planet_x=jnp.asarray(planet_x),
        planet_y=jnp.asarray(planet_y),
        planet_radius=jnp.asarray(planet_radius),
        planet_ships=jnp.asarray(planet_ships),
        planet_prod=jnp.asarray(planet_prod),
        planet_mask=jnp.asarray(planet_mask),
        fleet_owner=jnp.asarray(fleet_owner),
        fleet_x=jnp.asarray(fleet_x),
        fleet_y=jnp.asarray(fleet_y),
        fleet_angle=jnp.asarray(fleet_angle),
        fleet_ships=jnp.asarray(fleet_ships),
        fleet_mask=jnp.asarray(fleet_mask),
        step=jnp.int32(step),
        done=jnp.bool_(False),
        rng=jnp.zeros((2,), dtype=jnp.uint32),
        angular_velocity=jnp.float32(angular_velocity),
        planet_orbit_radius=jnp.asarray(planet_orbit_radius),
        planet_orbit_phase=jnp.asarray(planet_orbit_phase),
        planet_is_orbiting=jnp.asarray(planet_is_orbiting),
        # parity bridge can't recover the true home idx from a mid-game kaggle
        # observation; zero is a safe sentinel (shaping not used in parity).
        home_planet_idx=jnp.zeros((2,), dtype=jnp.int32),
        match_score=jnp.zeros((constants.NUM_PLAYERS,), dtype=jnp.int32),
        match_idx=jnp.int32(0),
        init_planet_ships=jnp.asarray(planet_ships),
        init_planet_owner=jnp.asarray(planet_owner),
        global_hist=__import__(
            "orbit_wars_rl.features.history", fromlist=["empty_global_hist"]
        ).empty_global_hist(),
    )


@dataclass
class StateDiff:
    """Per-field diff between kaggle ground-truth and our env at a single step."""
    step: int
    n_planets: int
    n_fleets_kaggle: int
    n_fleets_ours: int

    # planet-level
    planet_owner_mismatches: list[int]   # planet ids where owner differs
    planet_ships_max_abs_diff: int
    planet_ships_total_abs_diff: int
    planet_xy_max_abs_diff: float        # checks orbital motion
    planet_xy_moved_count: int           # # planets that moved between steps (kaggle)

    # fleet-level (best effort -- we identify by (owner, ships, angle) since ids differ)
    fleet_owner_set_matches: bool        # multiset of owners equal?
    fleet_ships_set_matches: bool        # multiset of ship counts equal?
    fleet_xy_max_abs_diff: float         # best-pair match; -1 if no fleets

    # other
    comet_count: int                     # observed comets in kaggle obs
    note: str

    def is_clean(self) -> bool:
        if self.planet_owner_mismatches: return False
        if self.planet_ships_max_abs_diff > 0: return False
        if self.planet_xy_max_abs_diff > 0.01: return False  # orbital motion not modeled
        if not self.fleet_owner_set_matches: return False
        if not self.fleet_ships_set_matches: return False
        if self.fleet_xy_max_abs_diff > 0.05 and self.n_fleets_kaggle > 0: return False
        return True


def diff_state(kaggle_obs: dict[str, Any], our_state: EnvState,
               prev_kaggle_obs: dict[str, Any] | None = None) -> StateDiff:
    """Compute a per-field diff between a kaggle obs and one of our EnvStates."""
    k_planets = [parse_planet(p) for p in kaggle_obs.get("planets", [])]
    k_fleets = [parse_fleet(f) for f in kaggle_obs.get("fleets", [])]
    step = int(kaggle_obs.get("step", -1))

    n_planets = len(k_planets)
    own_planet_owner = np.asarray(our_state.planet_owner)
    own_planet_ships = np.asarray(our_state.planet_ships)
    own_planet_x = np.asarray(our_state.planet_x)
    own_planet_y = np.asarray(our_state.planet_y)

    owner_mismatches = []
    ships_diffs = []
    xy_diffs = []
    for p in k_planets:
        if own_planet_owner[p.id] != p.owner:
            owner_mismatches.append(p.id)
        ships_diffs.append(int(abs(own_planet_ships[p.id] - p.ships)))
        xy_diffs.append(max(abs(own_planet_x[p.id] - p.x),
                            abs(own_planet_y[p.id] - p.y)))

    # how many planets moved between this step and last step in kaggle
    moved = 0
    if prev_kaggle_obs is not None:
        prev_planets = {p[0]: (p[2], p[3]) for p in prev_kaggle_obs.get("planets", [])}
        for p in k_planets:
            if p.id in prev_planets:
                px, py = prev_planets[p.id]
                if abs(p.x - px) > 0.01 or abs(p.y - py) > 0.01:
                    moved += 1

    # fleet multiset match (kaggle and ours may use different ids; compare bags)
    own_fleet_mask = np.asarray(our_state.fleet_mask)
    own_fleet_owner = np.asarray(our_state.fleet_owner)
    own_fleet_ships = np.asarray(our_state.fleet_ships)
    own_fleet_x = np.asarray(our_state.fleet_x)
    own_fleet_y = np.asarray(our_state.fleet_y)
    own_n = int(own_fleet_mask.sum())

    k_owner_bag = sorted([f.owner for f in k_fleets])
    own_owner_bag = sorted([int(own_fleet_owner[i]) for i in range(own_fleet_mask.shape[0])
                            if own_fleet_mask[i]])
    owner_set_match = (k_owner_bag == own_owner_bag)

    k_ships_bag = sorted([f.ships for f in k_fleets])
    own_ships_bag = sorted([int(own_fleet_ships[i]) for i in range(own_fleet_mask.shape[0])
                            if own_fleet_mask[i]])
    ships_set_match = (k_ships_bag == own_ships_bag)

    # best-pair xy diff (greedy match by (owner, ships))
    fleet_xy_max = -1.0
    if k_fleets and own_n > 0:
        own_unused = [i for i in range(own_fleet_mask.shape[0]) if own_fleet_mask[i]]
        for kf in k_fleets:
            best_d, best_i = float("inf"), None
            for i in own_unused:
                if int(own_fleet_owner[i]) != kf.owner:
                    continue
                if int(own_fleet_ships[i]) != kf.ships:
                    continue
                d = max(abs(float(own_fleet_x[i]) - kf.x),
                        abs(float(own_fleet_y[i]) - kf.y))
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None:
                own_unused.remove(best_i)
                fleet_xy_max = max(fleet_xy_max, best_d)

    comets = kaggle_obs.get("comets", [])
    comet_count = len(comets)

    return StateDiff(
        step=step,
        n_planets=n_planets,
        n_fleets_kaggle=len(k_fleets),
        n_fleets_ours=own_n,
        planet_owner_mismatches=owner_mismatches,
        planet_ships_max_abs_diff=max(ships_diffs) if ships_diffs else 0,
        planet_ships_total_abs_diff=sum(ships_diffs),
        planet_xy_max_abs_diff=max(xy_diffs) if xy_diffs else 0.0,
        planet_xy_moved_count=moved,
        fleet_owner_set_matches=owner_set_match,
        fleet_ships_set_matches=ships_set_match,
        fleet_xy_max_abs_diff=fleet_xy_max,
        comet_count=comet_count,
        note="",
    )


def format_diff(d: StateDiff) -> str:
    flag = "OK " if d.is_clean() else "BAD"
    lines = [
        f"[{flag}] step={d.step}  "
        f"planets={d.n_planets}  fleets(k/us)={d.n_fleets_kaggle}/{d.n_fleets_ours}  "
        f"comets={d.comet_count}",
    ]
    if d.planet_owner_mismatches:
        lines.append(f"     owner mismatch on planets: {d.planet_owner_mismatches}")
    if d.planet_ships_max_abs_diff > 0:
        lines.append(f"     planet ships diff: max={d.planet_ships_max_abs_diff}  "
                     f"total={d.planet_ships_total_abs_diff}")
    if d.planet_xy_max_abs_diff > 0.01:
        lines.append(f"     planet xy max_diff={d.planet_xy_max_abs_diff:.3f}  "
                     f"(kaggle moved {d.planet_xy_moved_count} planets this step)")
    if not d.fleet_owner_set_matches:
        lines.append(f"     fleet owner multiset mismatch")
    if not d.fleet_ships_set_matches:
        lines.append(f"     fleet ships multiset mismatch")
    if d.fleet_xy_max_abs_diff > 0.05:
        lines.append(f"     fleet xy max_diff={d.fleet_xy_max_abs_diff:.3f}")
    if d.comet_count > 0:
        lines.append(f"     ! comets present in kaggle ({d.comet_count}); we don't model them")
    return "\n".join(lines)
