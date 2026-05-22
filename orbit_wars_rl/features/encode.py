"""EnvState -> network-ready features. All jit-pure.

Each per-entity row is built so that **padding rows are exactly zero**, with
``mask`` carrying the validity flag. This keeps attention key-padding clean and
makes the global aggregates safe to compute via sum / mean.

Feature layouts (kept in lock-step with the heads):

  planet (19 dims):
    [0..2]  owner one-hot from the *current player*'s POV: [is_mine, is_enemy, is_neutral]
    [3]     x_norm  in [-1, 1]
    [4]     y_norm  in [-1, 1]
    [5]     radius_norm
    [6]     log1p(ships) / 8
    [7]     prod / 5
    [8]     dist_to_sun / BOARD
    [9]     fraction inbound friendly ships / max
    [10]    fraction inbound enemy ships  / max
    [11]    is_padding (1 if pad, else 0) -- redundant w/ mask but harmless
    [12]    is_orbiting (1 if planet rotates around sun)
    [13]    orbit_phase / pi  -- current angle from +x axis, in [-1, 1]
    [14]    orbit_radius / BOARD_HALF  -- distance to sun, in [0, ~1]
    [15]    x_at_t_plus_15  (lead-target predicted x; static planets = current x)
    [16]    y_at_t_plus_15
    [17]    x_at_t_plus_30
    [18]    y_at_t_plus_30

  Lead-target features (v4.2): pre-compute each planet's predicted position
  at t+15 / t+30 steps. This is what `top_players_rl.txt` calls "lead-target":
  policy must aim at where the orbiting planet WILL BE when fleet arrives.
  Fleet speed scales with ships (1.0 .. 6.0), so 15 & 30 cover near/far
  intercepts (board diagonal ~141 units; small fleets travel ~30 in 30 steps).
  Without these the 2-layer transformer would have to learn cos/sin in-
  context, which empirically (v4.0 + v4.1 both failed) it cannot do.

  fleet (8 dims): unchanged.

  global (11 dims):
    [0]    step / episode_steps
    [1..6] ship/planet/prod fractions (mine vs opp)
    [7..9] phase one-hot [early, mid, late]
    [10]   angular_velocity / OMEGA_MAX  -- episode-wide rotation speed, in [0.5, 1]
"""

from __future__ import annotations

from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.dynamics import fleet_speed
from orbit_wars_rl.env.state import EnvState


PLANET_FEAT_DIM = 19
FLEET_FEAT_DIM = 8
GLOBAL_FEAT_DIM = 11

# Lead-target prediction times (in env steps). Chosen so small fleets
# (speed=1.0, ~30 steps to cross half the board) and big fleets (speed=6,
# ~5 steps) both have a reasonable intercept prediction.
LEAD_TIMES = (15.0, 30.0)

_BOARD = jnp.float32(constants.BOARD)
_BOARD_HALF = jnp.float32(constants.BOARD * 0.5)
_SUN = jnp.array([constants.SUN_X, constants.SUN_Y], dtype=jnp.float32)


@chex.dataclass(frozen=True)
class EncodedObs:
    """Player-conditioned observation tensors."""

    planet_feats: chex.Array
    planet_mask: chex.Array
    fleet_feats: chex.Array
    fleet_mask: chex.Array
    global_feats: chex.Array

    my_planet_mask: chex.Array
    enemy_planet_mask: chex.Array
    neutral_planet_mask: chex.Array


def _inbound_ships(
    state: EnvState,
    owner_keep: int,
    planet_x: jnp.ndarray,
    planet_y: jnp.ndarray,
) -> jnp.ndarray:
    """Approximate the total in-flight ships heading toward each planet.

    We use a coarse heuristic (closest planet to the fleet's *velocity ray*),
    sufficient for features. Each fleet is attributed to exactly one planet.
    """
    speed = fleet_speed(state.fleet_ships)
    eps = jnp.float32(1e-6)
    spd_safe = jnp.maximum(speed, eps)
    dxn = jnp.cos(state.fleet_angle)
    dyn = jnp.sin(state.fleet_angle)

    rel_x = planet_x[None, :] - state.fleet_x[:, None]
    rel_y = planet_y[None, :] - state.fleet_y[:, None]
    proj = rel_x * dxn[:, None] + rel_y * dyn[:, None]
    perp_x = rel_x - proj * dxn[:, None]
    perp_y = rel_y - proj * dyn[:, None]
    perp_d2 = perp_x * perp_x + perp_y * perp_y

    in_front = proj > 0
    cost = jnp.where(in_front, perp_d2, jnp.float32(1e9))
    cost = jnp.where(state.planet_mask[None, :], cost, jnp.float32(1e9))
    target_idx = jnp.argmin(cost, axis=1)

    is_owner_keep = state.fleet_owner == owner_keep
    contributes = state.fleet_mask & is_owner_keep
    ships_keep = jnp.where(contributes, state.fleet_ships, jnp.int32(0))

    inbound = jnp.zeros((constants.MAX_PLANETS,), dtype=jnp.float32)
    inbound = inbound.at[target_idx].add(ships_keep.astype(jnp.float32))
    return inbound


def _encode_planets(state: EnvState, player: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    is_mine = (state.planet_owner == player) & state.planet_mask
    is_enemy = (state.planet_owner >= 0) & (state.planet_owner != player) & state.planet_mask
    is_neutral = (state.planet_owner == constants.NEUTRAL_OWNER) & state.planet_mask

    x_norm = (state.planet_x - _BOARD_HALF) / _BOARD_HALF
    y_norm = (state.planet_y - _BOARD_HALF) / _BOARD_HALF
    radius_norm = state.planet_radius / 5.0
    log_ships = jnp.log1p(jnp.maximum(state.planet_ships, 0).astype(jnp.float32)) / 8.0
    prod_norm = state.planet_prod.astype(jnp.float32) / 5.0
    dist_sun = jnp.sqrt(
        (state.planet_x - _SUN[0]) ** 2 + (state.planet_y - _SUN[1]) ** 2
    ) / _BOARD

    opp = 1 - player
    in_friend = _inbound_ships(state, player, state.planet_x, state.planet_y)
    in_foe = _inbound_ships(state, opp, state.planet_x, state.planet_y)
    in_friend_norm = jnp.log1p(in_friend) / 8.0
    in_foe_norm = jnp.log1p(in_foe) / 8.0

    is_padding = jnp.logical_not(state.planet_mask).astype(jnp.float32)

    is_orbiting = state.planet_is_orbiting.astype(jnp.float32)
    orbit_phase_norm = state.planet_orbit_phase / jnp.float32(jnp.pi)
    orbit_radius_norm = state.planet_orbit_radius / _BOARD_HALF

    # Lead-target features: predicted (x, y) at t + LEAD_TIMES[k] steps.
    # For static planets the prediction is just current pos. For orbiting we
    # advance phase by omega * T (kaggle linearises rotation to chord per
    # tick, but multi-step prediction is just exact rotation).
    omega = state.angular_velocity
    rotate_mask_f = state.planet_is_orbiting.astype(jnp.float32)
    lead_x_15, lead_y_15 = _predict_planet_pos(
        state.planet_orbit_phase, state.planet_orbit_radius, omega,
        jnp.float32(LEAD_TIMES[0]), state.planet_x, state.planet_y, rotate_mask_f,
    )
    lead_x_30, lead_y_30 = _predict_planet_pos(
        state.planet_orbit_phase, state.planet_orbit_radius, omega,
        jnp.float32(LEAD_TIMES[1]), state.planet_x, state.planet_y, rotate_mask_f,
    )
    lead_x_15_norm = (lead_x_15 - _BOARD_HALF) / _BOARD_HALF
    lead_y_15_norm = (lead_y_15 - _BOARD_HALF) / _BOARD_HALF
    lead_x_30_norm = (lead_x_30 - _BOARD_HALF) / _BOARD_HALF
    lead_y_30_norm = (lead_y_30 - _BOARD_HALF) / _BOARD_HALF

    feats = jnp.stack(
        [
            is_mine.astype(jnp.float32),
            is_enemy.astype(jnp.float32),
            is_neutral.astype(jnp.float32),
            x_norm,
            y_norm,
            radius_norm,
            log_ships,
            prod_norm,
            dist_sun,
            in_friend_norm,
            in_foe_norm,
            is_padding,
            is_orbiting,
            orbit_phase_norm,
            orbit_radius_norm,
            lead_x_15_norm,
            lead_y_15_norm,
            lead_x_30_norm,
            lead_y_30_norm,
        ],
        axis=-1,
    )
    feats = feats * state.planet_mask[:, None].astype(jnp.float32)
    return feats, is_mine, is_enemy, is_neutral


def _predict_planet_pos(
    orbit_phase: jnp.ndarray,
    orbit_radius: jnp.ndarray,
    omega: jnp.ndarray,
    lead_time: jnp.ndarray,
    cur_x: jnp.ndarray,
    cur_y: jnp.ndarray,
    rotate_mask_f: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute (x, y) at t + lead_time for each planet. Static planets keep
    current pos. Caller passes ``rotate_mask_f`` as float32 mask (1.0 for
    orbiting, 0.0 for static) so we can fuse the where-cond branchlessly.
    """
    new_phase = orbit_phase + omega * lead_time
    sun_x = _SUN[0]
    sun_y = _SUN[1]
    rot_x = sun_x + orbit_radius * jnp.cos(new_phase)
    rot_y = sun_y + orbit_radius * jnp.sin(new_phase)
    out_x = rotate_mask_f * rot_x + (1.0 - rotate_mask_f) * cur_x
    out_y = rotate_mask_f * rot_y + (1.0 - rotate_mask_f) * cur_y
    return out_x, out_y


def _encode_fleets(state: EnvState, player: int) -> jnp.ndarray:
    is_mine = (state.fleet_owner == player) & state.fleet_mask
    is_enemy = (state.fleet_owner >= 0) & (state.fleet_owner != player) & state.fleet_mask

    x_norm = (state.fleet_x - _BOARD_HALF) / _BOARD_HALF
    y_norm = (state.fleet_y - _BOARD_HALF) / _BOARD_HALF
    sin_a = jnp.sin(state.fleet_angle)
    cos_a = jnp.cos(state.fleet_angle)
    log_ships = jnp.log1p(jnp.maximum(state.fleet_ships, 0).astype(jnp.float32)) / 8.0
    zero = jnp.zeros_like(x_norm)

    feats = jnp.stack(
        [
            is_mine.astype(jnp.float32),
            is_enemy.astype(jnp.float32),
            zero,
            x_norm,
            y_norm,
            sin_a,
            cos_a,
            log_ships,
        ],
        axis=-1,
    )
    feats = feats * state.fleet_mask[:, None].astype(jnp.float32)
    return feats


def _encode_global(state: EnvState, player: int, episode_steps: int) -> jnp.ndarray:
    opp = 1 - player

    def player_ships(p: int) -> jnp.ndarray:
        ps_planet = jnp.where(
            (state.planet_owner == p) & state.planet_mask, state.planet_ships, 0
        ).sum()
        ps_fleet = jnp.where(
            (state.fleet_owner == p) & state.fleet_mask, state.fleet_ships, 0
        ).sum()
        return (ps_planet + ps_fleet).astype(jnp.float32)

    def player_planets(p: int) -> jnp.ndarray:
        return ((state.planet_owner == p) & state.planet_mask).sum().astype(jnp.float32)

    def player_prod(p: int) -> jnp.ndarray:
        return jnp.where(
            (state.planet_owner == p) & state.planet_mask, state.planet_prod, 0
        ).sum().astype(jnp.float32)

    total_ships = jnp.maximum(player_ships(player) + player_ships(opp), jnp.float32(1.0))
    total_planets = jnp.maximum(player_planets(player) + player_planets(opp), jnp.float32(1.0))
    total_prod = jnp.maximum(player_prod(player) + player_prod(opp), jnp.float32(1.0))

    step_norm = state.step.astype(jnp.float32) / jnp.float32(episode_steps)
    is_early = (step_norm < 0.18).astype(jnp.float32)
    is_mid = ((step_norm >= 0.18) & (step_norm < 0.64)).astype(jnp.float32)
    is_late = (step_norm >= 0.64).astype(jnp.float32)

    av_norm = state.angular_velocity / jnp.float32(constants.ORBIT_OMEGA_MAX)

    return jnp.stack(
        [
            step_norm,
            player_ships(player) / total_ships,
            player_ships(opp) / total_ships,
            player_planets(player) / total_planets,
            player_planets(opp) / total_planets,
            player_prod(player) / total_prod,
            player_prod(opp) / total_prod,
            is_early,
            is_mid,
            is_late,
            av_norm,
        ],
    )


def encode(state: EnvState, player: int, episode_steps: int = constants.DEFAULT_EPISODE_STEPS) -> EncodedObs:
    planet_feats, is_mine, is_enemy, is_neutral = _encode_planets(state, player)
    fleet_feats = _encode_fleets(state, player)
    global_feats = _encode_global(state, player, episode_steps)

    return EncodedObs(
        planet_feats=planet_feats,
        planet_mask=state.planet_mask,
        fleet_feats=fleet_feats,
        fleet_mask=state.fleet_mask,
        global_feats=global_feats,
        my_planet_mask=is_mine,
        enemy_planet_mask=is_enemy,
        neutral_planet_mask=is_neutral,
    )
