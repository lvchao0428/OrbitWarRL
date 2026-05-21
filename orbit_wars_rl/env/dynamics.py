"""Jit-pure step logic. NO Python control flow on traced values.

Turn order mirrors the official Orbit Wars spec (see README.md, section
"Turn Order"), minus the rotation/comet steps which the MVP drops:

  1. _launch_fleets  -- spawn fleets from player actions
  2. _produce        -- owned planets generate ships
  3. _move_fleets    -- advance each fleet by its speed (no rotation/comet)
  4. _resolve_combat -- arrived fleets fight planet garrisons

Everything is shape-stable: fleets that die have ``fleet_mask`` flipped off,
but their slot stays around until a new launch overwrites it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.actions import PlayerAction, decode_action
from orbit_wars_rl.env.state import EnvState


# ---------- speed model ----------------------------------------------------

_LOG_1000 = jnp.log(jnp.float32(1000.0))


def fleet_speed(ships: jnp.ndarray, max_speed: float = constants.DEFAULT_MAX_SHIP_SPEED) -> jnp.ndarray:
    """Mirrors orbit_wars: speed = 1 + (max-1) * (log(ships)/log(1000))^1.5."""
    s = jnp.maximum(ships.astype(jnp.float32), 1.0)
    rel = (jnp.log(s) / _LOG_1000) ** 1.5
    spd = 1.0 + (max_speed - 1.0) * rel
    return jnp.minimum(spd, max_speed)


# ---------- launch ---------------------------------------------------------


def _insert_one_fleet(
    state: EnvState,
    valid: jnp.ndarray,
    owner: int,
    ships: jnp.ndarray,
    src_x: jnp.ndarray,
    src_y: jnp.ndarray,
    src_radius: jnp.ndarray,
    angle: jnp.ndarray,
    src_idx: jnp.ndarray,
) -> EnvState:
    """Place a new fleet in the first inactive slot. No-op if ``valid`` is False.

    Spawn just outside the source planet's rim, matching the spec.
    """
    free_slots = jnp.logical_not(state.fleet_mask)
    free_idx = jnp.argmax(free_slots)
    has_free = free_slots.any()
    do_insert = valid & has_free & (ships > 0)

    pad = 0.5
    spawn_r = src_radius + pad
    spawn_x = src_x + spawn_r * jnp.cos(angle)
    spawn_y = src_y + spawn_r * jnp.sin(angle)

    fleet_owner = state.fleet_owner.at[free_idx].set(
        jnp.where(do_insert, jnp.int8(owner), state.fleet_owner[free_idx])
    )
    fleet_x = state.fleet_x.at[free_idx].set(
        jnp.where(do_insert, spawn_x, state.fleet_x[free_idx])
    )
    fleet_y = state.fleet_y.at[free_idx].set(
        jnp.where(do_insert, spawn_y, state.fleet_y[free_idx])
    )
    fleet_angle = state.fleet_angle.at[free_idx].set(
        jnp.where(do_insert, angle, state.fleet_angle[free_idx])
    )
    fleet_ships = state.fleet_ships.at[free_idx].set(
        jnp.where(do_insert, ships, state.fleet_ships[free_idx])
    )
    fleet_mask = state.fleet_mask.at[free_idx].set(
        jnp.where(do_insert, jnp.bool_(True), state.fleet_mask[free_idx])
    )

    new_planet_ships = state.planet_ships.at[src_idx].add(
        jnp.where(do_insert, -ships, jnp.int32(0))
    )

    return state.replace(
        fleet_owner=fleet_owner,
        fleet_x=fleet_x,
        fleet_y=fleet_y,
        fleet_angle=fleet_angle,
        fleet_ships=fleet_ships,
        fleet_mask=fleet_mask,
        planet_ships=new_planet_ships,
    )


def launch_fleets(state: EnvState, actions: tuple[PlayerAction, ...]) -> EnvState:
    """Apply each player's action sequentially. MVP: one fleet per player."""

    def step_player(s: EnvState, p_action_tup: tuple[int, PlayerAction]) -> EnvState:
        p, a = p_action_tup
        valid, src_idx, ships, angle, _dst_idx = decode_action(s, a, p)
        src_x = s.planet_x[src_idx]
        src_y = s.planet_y[src_idx]
        src_radius = s.planet_radius[src_idx]
        return _insert_one_fleet(
            s, valid, p, ships, src_x, src_y, src_radius, angle, src_idx
        )

    cur = state
    for p, a in enumerate(actions):
        cur = step_player(cur, (p, a))
    return cur


# ---------- production -----------------------------------------------------


def produce(state: EnvState) -> EnvState:
    """Owned (non-neutral, non-padding) planets generate ``planet_prod`` ships."""
    owned = (state.planet_owner >= 0) & state.planet_mask
    add = jnp.where(owned, state.planet_prod, jnp.int32(0))
    return state.replace(planet_ships=state.planet_ships + add)


# ---------- move + collide -------------------------------------------------


def _segment_circle_hit(
    ax: jnp.ndarray, ay: jnp.ndarray,
    bx: jnp.ndarray, by: jnp.ndarray,
    cx: jnp.ndarray, cy: jnp.ndarray,
    r: jnp.ndarray,
) -> jnp.ndarray:
    """True iff segment a -> b comes within radius ``r`` of point ``(cx, cy)``.

    Closed-form, fully vectorizable; no branches.
    """
    abx = bx - ax
    aby = by - ay
    apx = cx - ax
    apy = cy - ay
    ab2 = abx * abx + aby * aby
    safe_ab2 = jnp.where(ab2 < 1e-9, 1.0, ab2)
    t = (apx * abx + apy * aby) / safe_ab2
    t = jnp.clip(t, 0.0, 1.0)
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    dx = closest_x - cx
    dy = closest_y - cy
    return (dx * dx + dy * dy) <= (r * r)


def move_and_collide(state: EnvState) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    """Advance fleets one tick and detect planet collisions.

    Returns:
      new_state: with fleet_x/y advanced, fleet_mask cleared for fleets that died
                 (out-of-bounds, sun, or hit a planet).
      hit_planet_idx: int32[MAX_FLEETS], planet slot each fleet collided into
                      (irrelevant where ``hit_mask`` is False).
      hit_mask:       bool[MAX_FLEETS], True iff fleet just hit a planet this tick.
    """
    speed = fleet_speed(state.fleet_ships)
    dx = jnp.cos(state.fleet_angle) * speed
    dy = jnp.sin(state.fleet_angle) * speed
    new_x = state.fleet_x + dx
    new_y = state.fleet_y + dy

    in_bounds = (new_x >= 0.0) & (new_x <= constants.BOARD) & (new_y >= 0.0) & (new_y <= constants.BOARD)

    hits_sun = _segment_circle_hit(
        state.fleet_x, state.fleet_y, new_x, new_y,
        jnp.float32(constants.SUN_X), jnp.float32(constants.SUN_Y),
        jnp.float32(constants.SUN_RADIUS),
    )

    pf_dx = state.planet_x[None, :] - new_x[:, None]
    pf_dy = state.planet_y[None, :] - new_y[:, None]
    pf_dist2 = pf_dx * pf_dx + pf_dy * pf_dy
    planet_r = state.planet_radius[None, :]
    inside_planet = pf_dist2 <= (planet_r * planet_r)
    inside_planet = inside_planet & state.planet_mask[None, :] & state.fleet_mask[:, None]

    any_hit = inside_planet.any(axis=1)
    masked_dist2 = jnp.where(inside_planet, pf_dist2, jnp.float32(1e9))
    hit_planet_idx = jnp.argmin(masked_dist2, axis=1).astype(jnp.int32)

    fleet_dies_no_combat = state.fleet_mask & (jnp.logical_not(in_bounds) | hits_sun) & jnp.logical_not(any_hit)
    fleet_dies_combat = state.fleet_mask & any_hit & in_bounds & jnp.logical_not(hits_sun)
    fleet_survives = state.fleet_mask & in_bounds & jnp.logical_not(hits_sun) & jnp.logical_not(any_hit)

    new_mask = fleet_survives
    new_x_final = jnp.where(state.fleet_mask, new_x, state.fleet_x)
    new_y_final = jnp.where(state.fleet_mask, new_y, state.fleet_y)

    hit_mask = fleet_dies_combat

    new_state = state.replace(
        fleet_x=new_x_final,
        fleet_y=new_y_final,
        fleet_mask=new_mask,
    )
    _ = fleet_dies_no_combat
    return new_state, hit_planet_idx, hit_mask


# ---------- combat ---------------------------------------------------------

_OWNER_BUCKETS: int = constants.NUM_PLAYERS + 1


def _arrivals_per_planet(
    fleet_owner: jnp.ndarray,
    fleet_ships: jnp.ndarray,
    hit_planet_idx: jnp.ndarray,
    hit_mask: jnp.ndarray,
    num_planets: int,
) -> jnp.ndarray:
    """Build a [P, NUM_PLAYERS+1] table of incoming ships per (planet, owner).

    Owner index ``0..NUM_PLAYERS-1`` is players; ``NUM_PLAYERS`` is the (unused)
    neutral bucket -- neutrals never launch fleets, but we keep a row to keep
    indexing uniform if reused later.
    """
    owner_bucket = jnp.where(fleet_owner >= 0, fleet_owner.astype(jnp.int32), constants.NUM_PLAYERS)
    contributes = hit_mask & (fleet_ships > 0)
    ships_active = jnp.where(contributes, fleet_ships, jnp.int32(0))
    plane_idx = jnp.where(contributes, hit_planet_idx, jnp.int32(0))

    flat_idx = plane_idx * _OWNER_BUCKETS + owner_bucket
    table = jnp.zeros((num_planets * _OWNER_BUCKETS,), dtype=jnp.int32)
    table = table.at[flat_idx].add(ships_active)
    return table.reshape((num_planets, _OWNER_BUCKETS))


def _resolve_one_planet(
    planet_owner: jnp.ndarray,
    planet_garrison: jnp.ndarray,
    arrivals_row: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Resolve combat at a single planet given arrivals (one row of the table).

    Follows the spec: top attacker fights second attacker, survivor fights the
    garrison; ties produce no survivor.
    """
    forces = arrivals_row[: constants.NUM_PLAYERS]
    largest = jnp.max(forces)
    is_largest = forces == largest

    multiple_top = jnp.sum(is_largest.astype(jnp.int32)) > 1
    sum_top = jnp.sum(jnp.where(is_largest, forces, jnp.int32(0)))
    second = jnp.where(multiple_top, largest, jnp.max(jnp.where(is_largest, jnp.int32(-1), forces)))
    second = jnp.maximum(second, jnp.int32(0))

    attacker_id = jnp.where(multiple_top, jnp.int32(-1), jnp.argmax(forces).astype(jnp.int32))
    attacker_force = jnp.where(multiple_top, jnp.int32(0), largest)

    survivor = jnp.where(multiple_top, jnp.int32(0), attacker_force - second)
    has_combat = (sum_top > 0) & jnp.logical_not(multiple_top) & (survivor > 0)

    same_owner = attacker_id == planet_owner.astype(jnp.int32)

    reinforced_garrison = planet_garrison + survivor
    flips_owner = (jnp.logical_not(same_owner)) & (survivor > planet_garrison)
    flipped_garrison = survivor - planet_garrison
    holds_garrison = planet_garrison - survivor

    new_garrison = jnp.where(
        has_combat,
        jnp.where(same_owner, reinforced_garrison,
                  jnp.where(flips_owner, flipped_garrison, holds_garrison)),
        planet_garrison,
    )
    new_owner = jnp.where(
        has_combat & flips_owner,
        attacker_id.astype(jnp.int8),
        planet_owner,
    )
    return new_owner, jnp.maximum(new_garrison, jnp.int32(0))


def resolve_combat(
    state: EnvState,
    hit_planet_idx: jnp.ndarray,
    hit_mask: jnp.ndarray,
) -> EnvState:
    """Aggregate arrivals per planet, then run per-planet combat in parallel."""
    arrivals = _arrivals_per_planet(
        state.fleet_owner, state.fleet_ships,
        hit_planet_idx, hit_mask, constants.MAX_PLANETS,
    )
    new_owner, new_garrison = jax.vmap(_resolve_one_planet)(
        state.planet_owner, state.planet_ships, arrivals
    )
    new_owner = jnp.where(state.planet_mask, new_owner, state.planet_owner)
    new_garrison = jnp.where(state.planet_mask, new_garrison, state.planet_ships)
    return state.replace(planet_owner=new_owner, planet_ships=new_garrison)
