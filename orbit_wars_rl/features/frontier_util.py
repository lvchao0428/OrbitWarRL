"""Frontier / interior / shuffle helpers shared by encode and rewards (v27)."""

from __future__ import annotations

import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.state import EnvState

# Planets closer than this are treated as graph neighbors for frontier scoring.
_NEIGHBOR_DIST = jnp.float32(120.0)


def _pairwise_dist(state: EnvState) -> jnp.ndarray:
    dx = state.planet_x[:, None] - state.planet_x[None, :]
    dy = state.planet_y[:, None] - state.planet_y[None, :]
    return jnp.sqrt(dx * dx + dy * dy + jnp.float32(1e-6))


def expandable_mask(state: EnvState, player: int) -> jnp.ndarray:
    """Per-planet: True if neutral or enemy (capturable from expansion POV)."""
    owners = state.planet_owner
    is_neutral = (owners == constants.NEUTRAL_OWNER) & state.planet_mask
    is_enemy = (owners >= 0) & (owners != player) & state.planet_mask
    return is_neutral | is_enemy


def frontier_score_per_planet(state: EnvState, player: int) -> jnp.ndarray:
    """[P] in [0,1]: adjacency to neutral/enemy within _NEIGHBOR_DIST."""
    dist = _pairwise_dist(state)
    live = state.planet_mask[:, None] & state.planet_mask[None, :]
    neigh = (dist < _NEIGHBOR_DIST) & live & (dist > jnp.float32(0.0))
    expand = expandable_mask(state, player)
    n_expand_neigh = (neigh & expand[None, :]).astype(jnp.float32).sum(axis=-1)
    n_neigh = neigh.astype(jnp.float32).sum(axis=-1)
    score = n_expand_neigh / jnp.maximum(n_neigh, jnp.float32(1.0))
    return jnp.where(state.planet_mask, score, jnp.float32(0.0))


def interior_planet_bin(
    state: EnvState, player: int, frontier: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Owned planets with no expandable neighbor (safe interior)."""
    if frontier is None:
        frontier = frontier_score_per_planet(state, player)
    is_mine = (state.planet_owner == player) & state.planet_mask
    return (is_mine & (frontier <= jnp.float32(0.05))).astype(jnp.float32)


def capture_need_exact_norm(
    state: EnvState,
    player: int,
    need: jnp.ndarray,
    friendly_inbound: jnp.ndarray,
) -> jnp.ndarray:
    """[P] remaining garrison need after friendly inbound, / need (capturable)."""
    expand = expandable_mask(state, player)
    remaining = jnp.maximum(need - friendly_inbound, jnp.float32(0.0))
    exact = jnp.clip(remaining / jnp.maximum(need, jnp.float32(1.0)), 0.0, 1.0)
    return jnp.where(expand, exact, jnp.float32(0.0))


def shuffle_dst_risk_norm(
    state: EnvState,
    player: int,
    threat_ratio: jnp.ndarray,
    friendly_inbound: jnp.ndarray,
) -> jnp.ndarray:
    """[P] high on owned, low-threat planets already receiving friendlies."""
    is_mine = (state.planet_owner == player) & state.planet_mask
    low_threat = threat_ratio <= jnp.float32(0.12)
    has_inbound = friendly_inbound > jnp.float32(1.0)
    risk = (low_threat & has_inbound).astype(jnp.float32)
    return jnp.where(is_mine, risk, jnp.float32(0.0))


def home_under_threat_flag(state: EnvState, player: int, enemy_w1: jnp.ndarray) -> jnp.float32:
    """Scalar: any owned high-prod planet sees enemy fleet ETA w1."""
    is_mine = (state.planet_owner == player) & state.planet_mask
    prod_f = state.planet_prod.astype(jnp.float32)
    prod_thresh = jnp.max(jnp.where(is_mine, prod_f, jnp.float32(0.0))) * jnp.float32(0.5)
    high_prod = is_mine & (prod_f >= jnp.maximum(prod_thresh, jnp.float32(3.0)))
    threatened = high_prod & (enemy_w1 > jnp.float32(0.0))
    return jnp.where(jnp.any(threatened), jnp.float32(1.0), jnp.float32(0.0))
