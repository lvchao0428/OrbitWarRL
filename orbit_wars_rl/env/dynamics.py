"""Jit-pure step logic. NO Python control flow on traced values.

Turn order mirrors the official Orbit Wars spec (see overview.txt section
"Turn Order"). Comets are still TODO; everything else matches Kaggle:

  1. _launch_fleets  -- spawn fleets from player actions
  2. _produce        -- owned planets generate ships
  3. move_and_collide -- advance fleets, check sun/oob/planet collisions
  4. rotate_planets   -- spin orbiting planets by angular_velocity (PR1)
  5. (TODO) sweep planet motion against surviving fleets       -- PR2
  6. _resolve_combat -- arrived fleets fight planet garrisons

Everything is shape-stable: fleets that die have ``fleet_mask`` flipped off,
but their slot stays around until a new launch overwrites it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.actions import MultiPlayerAction, PlayerAction, decode_action
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

    # Spawn offset matches kaggle_environments orbit_wars: planet_radius + 0.1.
    # Confirmed by reverse-engineering kaggle env state at multiple seeds:
    # fleet position one tick after launch == (spawn + speed*direction), where
    # spawn = planet_center + (radius+0.1) * (cos(angle), sin(angle)).
    # See orbit_wars_rl/parity/run_env_parity.py for the regression test.
    pad = 0.1
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


def _launch_one_player_multi(
    state: EnvState,
    action: MultiPlayerAction,
    player: int,
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    """Launch up to K fleets sequentially for a single player.

    Uses lax.scan over K so the unroll is static and JIT-friendly. A running
    ``reserved_ships[MAX_PLANETS]`` counter tracks how much garrison has
    already been spent earlier in the same turn so a later step from the same
    src never oversubscribes the planet (the decoder downgrades ships to 0).

    Returns ``(new_state, valid_mask[K], ships_per_launch[K])``. The two
    arrays are useful for downstream shaping rewards that need to know what
    actually got dispatched. Callers that don't care can ignore them.
    """
    k = action.src_idx.shape[0]

    def step(carry, t):
        s, reserved = carry
        single = PlayerAction(
            src_idx=action.src_idx[t],
            dst_idx=action.dst_idx[t],
            pct_bin=action.pct_bin[t],
        )
        emit = action.emit_mask[t]
        valid_dec, src_idx, ships, angle, _dst = decode_action(s, single, player, reserved_ships=reserved)
        valid = valid_dec & emit
        ships_eff = jnp.where(valid, ships, jnp.int32(0))

        src_x = s.planet_x[src_idx]
        src_y = s.planet_y[src_idx]
        src_radius = s.planet_radius[src_idx]
        s_new = _insert_one_fleet(
            s, valid, player, ships_eff, src_x, src_y, src_radius, angle, src_idx
        )
        new_reserved = reserved.at[src_idx].add(ships_eff)
        return (s_new, new_reserved), (valid, ships_eff)

    reserved0 = jnp.zeros((constants.MAX_PLANETS,), dtype=jnp.int32)
    (final_state, _), (valid_per_step, ships_per_step) = jax.lax.scan(
        step, (state, reserved0), jnp.arange(k)
    )
    return final_state, valid_per_step, ships_per_step


def launch_fleets(
    state: EnvState,
    actions: tuple[MultiPlayerAction, MultiPlayerAction],
) -> EnvState:
    """Apply each player's multi-fleet action sequentially.

    ``actions`` is a tuple of ``MultiPlayerAction`` (one per player). Each
    player's actions are processed in player-id order, then within each
    player by autoregressive step. ``planet_ships`` is decremented eagerly
    inside ``_insert_one_fleet``, plus a per-turn ``reserved_ships`` buffer
    prevents two steps emitted by the same player from oversubscribing the
    same source planet (decoder downgrades ships to 0).

    Convenience wrapper that discards the per-launch metadata. Use
    ``launch_fleets_with_info`` when you need ``(valid_mask, ships)`` arrays
    for shaping rewards.
    """
    cur, _, _ = launch_fleets_with_info(state, actions)
    return cur


def launch_fleets_with_info(
    state: EnvState,
    actions: tuple[MultiPlayerAction, MultiPlayerAction],
) -> tuple[EnvState, tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """Like ``launch_fleets`` but also returns ``(valid, ships)`` arrays per player.

    Returns:
      * ``new_state``
      * ``valid``  -- tuple ``(valid_p0[K], valid_p1[K])`` bool arrays
      * ``ships``  -- tuple ``(ships_p0[K], ships_p1[K])`` int32 arrays
    """
    cur, valid_p0, ships_p0 = _launch_one_player_multi(state, actions[0], 0)
    cur, valid_p1, ships_p1 = _launch_one_player_multi(cur, actions[1], 1)
    return cur, (valid_p0, valid_p1), (ships_p0, ships_p1)


# ---------- production -----------------------------------------------------


def produce(state: EnvState) -> EnvState:
    """Owned (non-neutral, non-padding) planets generate ``planet_prod`` ships."""
    owned = (state.planet_owner >= 0) & state.planet_mask
    add = jnp.where(owned, state.planet_prod, jnp.int32(0))
    return state.replace(planet_ships=state.planet_ships + add)


# ---------- planet rotation ------------------------------------------------


def _planet_paths(state: EnvState) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """For each planet, compute (old_x, old_y, new_x, new_y) for this tick.

    Mirrors Kaggle's interpreter() lines 530-547: position is recomputed from
    the *initial* (radius, phase) plus omega*step, so phase drift is bounded
    by floating-point precision of one cos/sin call, not by accumulated
    addition. We achieve the same effect by carrying ``planet_orbit_phase``
    explicitly (we already update it per step), so here we just compute the
    new phase = current_phase + omega.

    First-turn gate matches Kaggle: ``step == 0`` going INTO the turn means
    obs is at step 0 already, planets stayed put through step 0->1.
    """
    omega = state.angular_velocity
    rotate_mask = state.planet_is_orbiting & state.planet_mask
    first_turn_skip = state.step == jnp.int32(0)
    effective_omega = jnp.where(first_turn_skip, jnp.float32(0.0), omega)
    delta_phase = jnp.where(rotate_mask, effective_omega, jnp.float32(0.0))
    new_phase = state.planet_orbit_phase + delta_phase

    sun_x = jnp.float32(constants.SUN_X)
    sun_y = jnp.float32(constants.SUN_Y)
    new_x_raw = sun_x + state.planet_orbit_radius * jnp.cos(new_phase)
    new_y_raw = sun_y + state.planet_orbit_radius * jnp.sin(new_phase)
    # Static / padding planets don't move.
    new_x = jnp.where(rotate_mask, new_x_raw, state.planet_x)
    new_y = jnp.where(rotate_mask, new_y_raw, state.planet_y)
    return state.planet_x, state.planet_y, new_x, new_y


def rotate_planets(state: EnvState) -> EnvState:
    """Spin orbiting planets by ``angular_velocity`` for one tick.

    Idempotent given the same state. Use after combat resolution to apply
    this tick's planet motion. Note: collision detection in
    ``move_and_collide`` already uses the swept planet segment, so the
    'apply' step is just bookkeeping.
    """
    _ox, _oy, new_x, new_y = _planet_paths(state)
    omega = state.angular_velocity
    rotate_mask = state.planet_is_orbiting & state.planet_mask
    first_turn_skip = state.step == jnp.int32(0)
    effective_omega = jnp.where(first_turn_skip, jnp.float32(0.0), omega)
    delta_phase = jnp.where(rotate_mask, effective_omega, jnp.float32(0.0))
    new_phase = state.planet_orbit_phase + delta_phase
    return state.replace(
        planet_orbit_phase=new_phase,
        planet_x=new_x,
        planet_y=new_y,
    )


# ---------- move + collide (swept-pair, matches Kaggle source) -------------


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


def _swept_pair_hit(
    ax: jnp.ndarray, ay: jnp.ndarray,
    bx: jnp.ndarray, by: jnp.ndarray,
    px0: jnp.ndarray, py0: jnp.ndarray,
    px1: jnp.ndarray, py1: jnp.ndarray,
    r: jnp.ndarray,
) -> jnp.ndarray:
    """Continuous collision test: does fleet segment A->B come within ``r`` of
    planet segment P0->P1 for some t in [0, 1]?

    Port of ``swept_pair_hit`` from kaggle_environments orbit_wars.py (line 46).
    Both objects move linearly over the tick (planet rotation is linearised
    to its chord). Returns True iff the time-of-closest-approach distance
    is < r at some t in [0, 1].
    """
    d0x = ax - px0
    d0y = ay - py0
    dvx = (bx - ax) - (px1 - px0)
    dvy = (by - ay) - (py1 - py0)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    # Branch-free: when a == 0, fall back to constant-distance test (c <= 0).
    a_zero = a < 1e-12
    safe_a = jnp.where(a_zero, 1.0, a)
    disc = b * b - 4.0 * safe_a * c
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)
    parabolic_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    return jnp.where(a_zero, c <= 0.0, parabolic_hit)


def move_and_collide(state: EnvState) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    """Advance fleets one tick and detect planet collisions (swept-pair).

    Mirrors kaggle_environments orbit_wars interpreter() lines 516-633:
      1. Compute each planet's start/end position (planet segment) using the
         current state and orbital deltas for orbiting planets.
      2. For each fleet, walk its move segment A -> B and against each
         planet's swept segment P0 -> P1 via ``_swept_pair_hit``. The first
         (closest by endpoint) hit kills the fleet and sends it into combat.
      3. Surviving fleets are tested for out-of-bounds and sun crossing
         after the planet test (kaggle priority: planet hit beats OOB/sun).

    Returns:
      new_state: fleets advanced; fleet_mask cleared for any fleet that
                 died (OOB, sun, planet hit). NOTE: planet positions are
                 NOT updated here -- call ``rotate_planets`` after combat.
      hit_planet_idx: int32[MAX_FLEETS], planet slot each fleet hit
                      (irrelevant where ``hit_mask`` is False).
      hit_mask:       bool[MAX_FLEETS], True iff fleet hit a planet this tick.
    """
    speed = fleet_speed(state.fleet_ships)
    fa = state.fleet_angle
    fox = state.fleet_x
    foy = state.fleet_y
    fnx = fox + jnp.cos(fa) * speed
    fny = foy + jnp.sin(fa) * speed

    in_bounds = (fnx >= 0.0) & (fnx <= constants.BOARD) & (fny >= 0.0) & (fny <= constants.BOARD)

    hits_sun = _segment_circle_hit(
        fox, foy, fnx, fny,
        jnp.float32(constants.SUN_X), jnp.float32(constants.SUN_Y),
        jnp.float32(constants.SUN_RADIUS),
    )

    # Planet swept segments for this tick. Static planets get (cur, cur).
    pox, poy, pnx, pny = _planet_paths(state)

    # Vectorized fleet-vs-planet swept-pair hit test.
    F = fox.shape[0]
    P = pox.shape[0]
    fox_b = fox[:, None]
    foy_b = foy[:, None]
    fnx_b = fnx[:, None]
    fny_b = fny[:, None]
    pox_b = pox[None, :]
    poy_b = poy[None, :]
    pnx_b = pnx[None, :]
    pny_b = pny[None, :]
    pr_b = state.planet_radius[None, :]

    swept = _swept_pair_hit(fox_b, foy_b, fnx_b, fny_b, pox_b, poy_b, pnx_b, pny_b, pr_b)
    swept = swept & state.planet_mask[None, :] & state.fleet_mask[:, None]

    any_hit = swept.any(axis=1)

    # Tiebreak: when several planets are hit, pick the closest by *fleet
    # endpoint* (matches Kaggle's first-iteration tiebreak, since the loop
    # short-circuits on the first hit in planet-id order). We use endpoint
    # distance because it's deterministic and rarely matters; the more
    # common case is a single planet hit.
    pf_dx = pnx_b - fnx_b
    pf_dy = pny_b - fny_b
    pf_dist2 = pf_dx * pf_dx + pf_dy * pf_dy
    masked_dist2 = jnp.where(swept, pf_dist2, jnp.float32(1e9))
    hit_planet_idx = jnp.argmin(masked_dist2, axis=1).astype(jnp.int32)

    # Priority: planet hit > sun > OOB. Matches Kaggle (line 587).
    fleet_dies_no_combat = state.fleet_mask & jnp.logical_not(any_hit) & (
        jnp.logical_not(in_bounds) | hits_sun
    )
    fleet_dies_combat = state.fleet_mask & any_hit
    fleet_survives = state.fleet_mask & jnp.logical_not(any_hit) & in_bounds & jnp.logical_not(hits_sun)

    new_mask = fleet_survives
    new_x_final = jnp.where(state.fleet_mask, fnx, fox)
    new_y_final = jnp.where(state.fleet_mask, fny, foy)

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
