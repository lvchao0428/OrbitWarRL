"""EnvState -> network-ready features. All jit-pure.

Each per-entity row is built so that **padding rows are exactly zero**, with
``mask`` carrying the validity flag. This keeps attention key-padding clean and
makes the global aggregates safe to compute via sum / mean.

Feature layouts (kept in lock-step with the heads):

  planet (22 dims):
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
    [19]    threat_ratio  -- soft foe inbound / garrison (A1')
    [20]    net_inbound   -- (foe - friend) soft inbound / garrison
    [21]    eta_foe_min   -- min ETA of inbound foe fleets / episode_steps

  fleet (8 dims): unchanged.

  global (14 dims):
    [0]    step / episode_steps
    [1..6] ship/planet/prod fractions (mine vs opp)
    [7..9] phase one-hot [early, mid, late]
    [10]   angular_velocity / OMEGA_MAX
    [11]   log1p(total_garrison_mine) / 10
    [12]   n_fleets_mine / MAX_FLEETS
    [13]   n_fleets_enemy / MAX_FLEETS
"""

from __future__ import annotations

from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.dynamics import fleet_speed
from orbit_wars_rl.env.state import EnvState


PLANET_FEAT_DIM = 22
FLEET_FEAT_DIM = 8
GLOBAL_FEAT_DIM = 14

LEAD_TIMES = (15.0, 30.0)

_BOARD = jnp.float32(constants.BOARD)
_BOARD_HALF = jnp.float32(constants.BOARD * 0.5)
_SUN = jnp.array([constants.SUN_X, constants.SUN_Y], dtype=jnp.float32)
_BIG_ETA = jnp.float32(1e6)


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


def _inbound_all_foes(
    state: EnvState,
    player: int,
    planet_x: jnp.ndarray,
    planet_y: jnp.ndarray,
) -> jnp.ndarray:
    """Hard inbound from every opponent (4P aggregates players != ``player``)."""
    if constants.NUM_PLAYERS == 2:
        return _inbound_ships(state, 1 - player, planet_x, planet_y)
    inbound = jnp.zeros((constants.MAX_PLANETS,), dtype=jnp.float32)
    for p in (0, 1, 2, 3):
        w = jnp.equal(p, player).astype(jnp.float32)
        inbound = inbound + (1.0 - w) * _inbound_ships(state, p, planet_x, planet_y)
    return inbound


def _soft_inbound_all_foes_and_eta(
    state: EnvState,
    player: int,
    planet_x: jnp.ndarray,
    planet_y: jnp.ndarray,
    episode_steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Soft foe inbound + min ETA aggregated over all opponents."""
    if constants.NUM_PLAYERS == 2:
        return _soft_inbound_and_eta(state, 1 - player, planet_x, planet_y, episode_steps)
    inbound = jnp.zeros((constants.MAX_PLANETS,), dtype=jnp.float32)
    eta_min = jnp.full((constants.MAX_PLANETS,), _BIG_ETA, dtype=jnp.float32)
    for p in (0, 1, 2, 3):
        w = jnp.equal(p, player).astype(jnp.float32)
        soft, eta = _soft_inbound_and_eta(state, p, planet_x, planet_y, episode_steps)
        inbound = inbound + (1.0 - w) * soft
        eta_min = jnp.minimum(eta_min, jnp.where(w > 0, _BIG_ETA, eta))
    eta_min = jnp.where(eta_min >= _BIG_ETA, jnp.float32(0.0), eta_min)
    return inbound, eta_min


def _foe_player_ships(state: EnvState, player: int) -> jnp.ndarray:
    """Aggregate opponent ship count for global fractions (sum in 4P)."""
    if constants.NUM_PLAYERS == 2:
        ps_planet = jnp.where(
            (state.planet_owner == (1 - player)) & state.planet_mask, state.planet_ships, 0
        ).sum()
        ps_fleet = jnp.where(
            (state.fleet_owner == (1 - player)) & state.fleet_mask, state.fleet_ships, 0
        ).sum()
        return (ps_planet + ps_fleet).astype(jnp.float32)
    total = jnp.float32(0.0)
    for p in (0, 1, 2, 3):
        w = jnp.equal(p, player).astype(jnp.float32)
        ps_planet = jnp.where(
            (state.planet_owner == p) & state.planet_mask, state.planet_ships, 0
        ).sum()
        ps_fleet = jnp.where(
            (state.fleet_owner == p) & state.fleet_mask, state.fleet_ships, 0
        ).sum()
        total = total + (1.0 - w) * (ps_planet + ps_fleet).astype(jnp.float32)
    return total


def _foe_player_planets(state: EnvState, player: int) -> jnp.ndarray:
    if constants.NUM_PLAYERS == 2:
        return ((state.planet_owner == (1 - player)) & state.planet_mask).sum().astype(jnp.float32)
    total = jnp.float32(0.0)
    for p in (0, 1, 2, 3):
        w = jnp.equal(p, player).astype(jnp.float32)
        n = ((state.planet_owner == p) & state.planet_mask).sum().astype(jnp.float32)
        total = total + (1.0 - w) * n
    return total


def _foe_player_prod(state: EnvState, player: int) -> jnp.ndarray:
    if constants.NUM_PLAYERS == 2:
        return jnp.where(
            (state.planet_owner == (1 - player)) & state.planet_mask, state.planet_prod, 0
        ).sum().astype(jnp.float32)
    total = jnp.float32(0.0)
    for p in (0, 1, 2, 3):
        w = jnp.equal(p, player).astype(jnp.float32)
        prod = jnp.where(
            (state.planet_owner == p) & state.planet_mask, state.planet_prod, 0
        ).sum().astype(jnp.float32)
        total = total + (1.0 - w) * prod
    return total


def _foe_player_fleets(state: EnvState, player: int) -> jnp.ndarray:
    if constants.NUM_PLAYERS == 2:
        return ((state.fleet_owner == (1 - player)) & state.fleet_mask).sum().astype(jnp.float32)
    total = jnp.float32(0.0)
    for p in (0, 1, 2, 3):
        w = jnp.equal(p, player).astype(jnp.float32)
        n = ((state.fleet_owner == p) & state.fleet_mask).sum().astype(jnp.float32)
        total = total + (1.0 - w) * n
    return total


def _inbound_ships(
    state: EnvState,
    owner_keep: int,
    planet_x: jnp.ndarray,
    planet_y: jnp.ndarray,
) -> jnp.ndarray:
    """Approximate the total in-flight ships heading toward each planet."""
    speed = fleet_speed(state.fleet_ships)
    eps = jnp.float32(1e-6)
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


def _soft_inbound_and_eta(
    state: EnvState,
    owner_keep: int,
    planet_x: jnp.ndarray,
    planet_y: jnp.ndarray,
    episode_steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Inverse-square weighted inbound ships + min foe ETA per planet (A1')."""
    eps = jnp.float32(1e-4)
    rel_x = planet_x[None, :] - state.fleet_x[:, None]
    rel_y = planet_y[None, :] - state.fleet_y[:, None]
    dist2 = rel_x * rel_x + rel_y * rel_y + eps
    dist = jnp.sqrt(dist2)
    speed = fleet_speed(state.fleet_ships)
    eta = dist / jnp.maximum(speed[:, None], eps)

    cos_a = jnp.cos(state.fleet_angle)
    sin_a = jnp.sin(state.fleet_angle)
    proj = rel_x * cos_a[:, None] + rel_y * sin_a[:, None]
    toward = (proj > 0).astype(jnp.float32)

    is_owner = ((state.fleet_owner == owner_keep) & state.fleet_mask).astype(jnp.float32)
    weight = toward * is_owner[:, None] / dist2
    ships = state.fleet_ships.astype(jnp.float32)
    inbound_soft = (weight * ships[:, None]).sum(axis=0)

    eta_masked = jnp.where(weight > 0, eta, _BIG_ETA)
    eta_min = eta_masked.min(axis=0)
    eta_min = jnp.where(eta_min >= _BIG_ETA, jnp.float32(0.0), eta_min)
    eta_norm = eta_min / jnp.float32(max(episode_steps, 1))
    return inbound_soft, eta_norm


def _encode_planets(
    state: EnvState, player: int, episode_steps: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    in_friend = _inbound_ships(state, player, state.planet_x, state.planet_y)
    in_foe = _inbound_all_foes(state, player, state.planet_x, state.planet_y)
    in_friend_norm = jnp.log1p(in_friend) / 8.0
    in_foe_norm = jnp.log1p(in_foe) / 8.0

    foe_soft, eta_foe_min = _soft_inbound_all_foes_and_eta(
        state, player, state.planet_x, state.planet_y, episode_steps
    )
    friend_soft, _ = _soft_inbound_and_eta(
        state, player, state.planet_x, state.planet_y, episode_steps
    )
    garr = jnp.maximum(state.planet_ships.astype(jnp.float32), jnp.float32(1.0))
    threat_ratio = foe_soft / garr
    net_inbound = (foe_soft - friend_soft) / garr
    threat_ratio = jnp.clip(threat_ratio / 8.0, 0.0, 1.0)
    net_inbound = jnp.clip(net_inbound / 8.0, -1.0, 1.0)

    is_padding = jnp.logical_not(state.planet_mask).astype(jnp.float32)

    is_orbiting = state.planet_is_orbiting.astype(jnp.float32)
    orbit_phase_norm = state.planet_orbit_phase / jnp.float32(jnp.pi)
    orbit_radius_norm = state.planet_orbit_radius / _BOARD_HALF

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
            threat_ratio,
            net_inbound,
            eta_foe_min,
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

    def player_garrison(p: int) -> jnp.ndarray:
        return jnp.where(
            (state.planet_owner == p) & state.planet_mask, state.planet_ships, 0
        ).sum().astype(jnp.float32)

    def player_fleets(p: int) -> jnp.ndarray:
        return ((state.fleet_owner == p) & state.fleet_mask).sum().astype(jnp.float32)

    foe_ships = _foe_player_ships(state, player)
    foe_planets = _foe_player_planets(state, player)
    foe_prod = _foe_player_prod(state, player)
    foe_fleets = _foe_player_fleets(state, player)

    total_ships = jnp.maximum(player_ships(player) + foe_ships, jnp.float32(1.0))
    total_planets = jnp.maximum(player_planets(player) + foe_planets, jnp.float32(1.0))
    total_prod = jnp.maximum(player_prod(player) + foe_prod, jnp.float32(1.0))

    step_norm = state.step.astype(jnp.float32) / jnp.float32(episode_steps)
    is_early = (step_norm < 0.18).astype(jnp.float32)
    is_mid = ((step_norm >= 0.18) & (step_norm < 0.64)).astype(jnp.float32)
    is_late = (step_norm >= 0.64).astype(jnp.float32)

    av_norm = state.angular_velocity / jnp.float32(constants.ORBIT_OMEGA_MAX)
    total_garr_norm = jnp.log1p(player_garrison(player)) / 10.0
    n_fleets_mine = player_fleets(player) / jnp.float32(constants.MAX_FLEETS)
    n_fleets_enemy = foe_fleets / jnp.float32(constants.MAX_FLEETS)

    return jnp.stack(
        [
            step_norm,
            player_ships(player) / total_ships,
            foe_ships / total_ships,
            player_planets(player) / total_planets,
            foe_planets / total_planets,
            player_prod(player) / total_prod,
            foe_prod / total_prod,
            is_early,
            is_mid,
            is_late,
            av_norm,
            total_garr_norm,
            n_fleets_mine,
            n_fleets_enemy,
        ],
    )


def encode(state: EnvState, player: int, episode_steps: int = constants.DEFAULT_EPISODE_STEPS) -> EncodedObs:
    planet_feats, is_mine, is_enemy, is_neutral = _encode_planets(state, player, episode_steps)
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
