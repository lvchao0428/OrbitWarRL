"""Global temporal history stack (Lux-style frame stack for macro signals).

Each step we append an 8-dim player-POV slice to a ring buffer of length
``HIST_LEN``. ``encode`` flattens the buffer onto the base 27 global dims.
"""

from __future__ import annotations

import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.state import EnvState

HIST_LEN = constants.HIST_LEN
TEMPORAL_GLOBAL_DIM = constants.TEMPORAL_GLOBAL_DIM
PLANET_HIST_LEN = constants.PLANET_HIST_LEN
PLANET_HIST_DIM = constants.PLANET_HIST_DIM

# Indices within each 8-dim temporal slice (stable contract for rewards).
TGF_SHIP_MINE = 0
TGF_SHIP_FOE = 1
TGF_PROD_MINE = 2
TGF_PROD_FOE = 3
TGF_PROD_ADV = 4
TGF_GARR_ADV = 5
TGF_FLEET_MASS = 6
TGF_THREAT = 7


def empty_global_hist() -> jnp.ndarray:
    """Shape [NUM_PLAYERS, HIST_LEN, TEMPORAL_GLOBAL_DIM]."""
    return jnp.zeros(
        (constants.NUM_PLAYERS, HIST_LEN, TEMPORAL_GLOBAL_DIM),
        dtype=jnp.float32,
    )


def temporal_global_slice(
    state: EnvState,
    player: int,
    episode_steps: int,
    wins_needed: int = 1,
) -> jnp.ndarray:
    """8-dim macro snapshot from *current* state (player POV)."""
    del episode_steps, wins_needed  # reserved for future match-aware slices

    def _ships(p: int) -> jnp.ndarray:
        ps = jnp.where(
            (state.planet_owner == p) & state.planet_mask, state.planet_ships, 0
        ).sum()
        fs = jnp.where(
            (state.fleet_owner == p) & state.fleet_mask, state.fleet_ships, 0
        ).sum()
        return (ps + fs).astype(jnp.float32)

    def _prod(p: int) -> jnp.ndarray:
        return jnp.where(
            (state.planet_owner == p) & state.planet_mask, state.planet_prod, 0
        ).sum().astype(jnp.float32)

    def _garr(p: int) -> jnp.ndarray:
        return jnp.where(
            (state.planet_owner == p) & state.planet_mask, state.planet_ships, 0
        ).sum().astype(jnp.float32)

    opp = 1 - player
    my_ships = _ships(player)
    foe_ships = _ships(opp)
    my_prod = _prod(player)
    foe_prod = _prod(opp)
    my_garr = _garr(player)
    foe_garr = jnp.where(
        (state.planet_owner >= 0) & (state.planet_owner != player) & state.planet_mask,
        state.planet_ships,
        0,
    ).sum().astype(jnp.float32)
    my_fleet = jnp.where(
        (state.fleet_owner == player) & state.fleet_mask, state.fleet_ships, 0
    ).sum().astype(jnp.float32)
    foe_fleet = jnp.where(
        (state.fleet_owner >= 0) & (state.fleet_owner != player) & state.fleet_mask,
        state.fleet_ships,
        0,
    ).sum().astype(jnp.float32)

    total_ships = jnp.maximum(my_ships + foe_ships, jnp.float32(1.0))
    total_prod = jnp.maximum(my_prod + foe_prod, jnp.float32(1.0))
    my_total = jnp.maximum(my_garr + my_fleet, jnp.float32(1.0))

    prod_adv = jnp.clip(
        (my_prod - foe_prod) / total_prod, -1.0, 1.0
    )
    garr_adv = jnp.clip(
        (my_garr - foe_garr) / jnp.maximum(my_garr + foe_garr, jnp.float32(1.0)),
        -1.0,
        1.0,
    )
    fleet_mass = my_fleet / my_total
    threat = jnp.clip(
        foe_fleet / jnp.maximum(my_garr, jnp.float32(1.0)), 0.0, 2.0
    ) / jnp.float32(2.0)

    return jnp.stack(
        [
            my_ships / total_ships,
            foe_ships / total_ships,
            my_prod / total_prod,
            foe_prod / total_prod,
            prod_adv,
            garr_adv,
            fleet_mass,
            threat,
        ],
    ).astype(jnp.float32)


def update_global_hist(
    state: EnvState,
    episode_steps: int,
    wins_needed: int = 1,
) -> EnvState:
    """Shift history left and append current temporal slices for both players."""
    slices = jnp.stack(
        [
            temporal_global_slice(state, 0, episode_steps, wins_needed),
            temporal_global_slice(state, 1, episode_steps, wins_needed),
        ],
        axis=0,
    )
    rolled = jnp.roll(state.global_hist, shift=-1, axis=1)
    new_hist = rolled.at[:, -1, :].set(slices)
    return state.replace(global_hist=new_hist)


def flatten_global_hist(state: EnvState, player: int) -> jnp.ndarray:
    """Shape [HIST_LEN * TEMPORAL_GLOBAL_DIM]."""
    return state.global_hist[player].reshape(-1)


def recent_garr_advantage_mean(state: EnvState, player: int, n: int = 10) -> jnp.ndarray:
    """Mean garr_advantage over the last ``n`` hist frames (scalar)."""
    trail = state.global_hist[player, -n:, TGF_GARR_ADV]
    return trail.mean()


def empty_planet_hist() -> jnp.ndarray:
    """Shape [NUM_PLAYERS, PLANET_HIST_LEN, MAX_PLANETS, PLANET_HIST_DIM]."""
    return jnp.zeros(
        (
            constants.NUM_PLAYERS,
            PLANET_HIST_LEN,
            constants.MAX_PLANETS,
            PLANET_HIST_DIM,
        ),
        dtype=jnp.float32,
    )


def _inbound_foe_ships(state: EnvState, player: int) -> jnp.ndarray:
    """Hard inbound from all opponents toward each planet (simplified)."""
    speed = jnp.float32(1.0)  # unused; heading heuristic only
    del speed
    eps = jnp.float32(1e-6)
    dxn = jnp.cos(state.fleet_angle)
    dyn = jnp.sin(state.fleet_angle)

    rel_x = state.planet_x[None, :] - state.fleet_x[:, None]
    rel_y = state.planet_y[None, :] - state.fleet_y[:, None]
    proj = rel_x * dxn[:, None] + rel_y * dyn[:, None]
    perp_x = rel_x - proj * dxn[:, None]
    perp_y = rel_y - proj * dyn[:, None]
    perp_d2 = perp_x * perp_x + perp_y * perp_y

    in_front = proj > 0
    cost = jnp.where(in_front, perp_d2, jnp.float32(1e9))
    cost = jnp.where(state.planet_mask[None, :], cost, jnp.float32(1e9))
    target_idx = jnp.argmin(cost, axis=1)

    is_opp = (
        (state.fleet_owner >= 0)
        & (state.fleet_owner != player)
        & state.fleet_mask
    )
    ships_opp = jnp.where(is_opp, state.fleet_ships, jnp.int32(0))

    inbound = jnp.zeros((constants.MAX_PLANETS,), dtype=jnp.float32)
    inbound = inbound.at[target_idx].add(ships_opp.astype(jnp.float32))
    return inbound


def _planet_hist_slice(state: EnvState, player: int) -> jnp.ndarray:
    """4-dim per-planet slice for one player POV. Shape [MAX_PLANETS, 4]."""
    is_mine = (
        (state.planet_owner == player) & state.planet_mask
    ).astype(jnp.float32)
    garr_norm = (
        jnp.log1p(jnp.maximum(state.planet_ships, 0).astype(jnp.float32)) / jnp.float32(8.0)
    )
    inbound_foe = _inbound_foe_ships(state, player)
    inbound_foe_norm = jnp.log1p(inbound_foe) / jnp.float32(8.0)

    prev_is_mine = state.planet_hist[player, -1, :, 0]
    was_flipped = (is_mine != prev_is_mine).astype(jnp.float32)

    return jnp.stack([is_mine, garr_norm, inbound_foe_norm, was_flipped], axis=-1)


def update_planet_hist(state: EnvState, episode_steps: int = 500) -> EnvState:
    """Shift planet history left and append current per-planet slice."""
    del episode_steps  # reserved for future match-aware slices
    slices = jnp.stack(
        [_planet_hist_slice(state, p) for p in range(constants.NUM_PLAYERS)],
        axis=0,
    )
    rolled = jnp.roll(state.planet_hist, shift=-1, axis=1)
    new_hist = rolled.at[:, -1, :, :].set(slices)
    return state.replace(planet_hist=new_hist)


def flatten_planet_hist(state: EnvState, player: int) -> jnp.ndarray:
    """Shape [MAX_PLANETS, PLANET_HIST_LEN * PLANET_HIST_DIM] = [40, 20]."""
    return state.planet_hist[player].transpose(1, 0, 2).reshape(constants.MAX_PLANETS, -1)
