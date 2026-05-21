"""Reward shaping. MVP keeps it minimal: +1 win, -1 loss, 0 draw / ongoing."""

from __future__ import annotations

import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.state import EnvState


def player_total_ships(state: EnvState, player: int) -> jnp.ndarray:
    """Total ships (planets + fleets) for ``player``, masked by validity."""
    planet_mine = (state.planet_owner == player) & state.planet_mask
    fleet_mine = (state.fleet_owner == player) & state.fleet_mask
    planet_sum = jnp.where(planet_mine, state.planet_ships, 0).sum()
    fleet_sum = jnp.where(fleet_mine, state.fleet_ships, 0).sum()
    return (planet_sum + fleet_sum).astype(jnp.int32)


def player_alive(state: EnvState, player: int) -> jnp.ndarray:
    """A player is alive iff they own >=1 planet OR >=1 in-flight fleet."""
    has_planet = ((state.planet_owner == player) & state.planet_mask).any()
    has_fleet = ((state.fleet_owner == player) & state.fleet_mask).any()
    return has_planet | has_fleet


def terminal_reward(state: EnvState, player: int) -> jnp.ndarray:
    """+1 / -1 / 0 based on final ship count comparison (2P MVP).

    Only meaningful when ``done`` is True; callers should mask otherwise.
    """
    me = player_total_ships(state, player)
    opp_id = 1 - player
    opp = player_total_ships(state, opp_id)
    win = me > opp
    loss = me < opp
    return (win.astype(jnp.float32) - loss.astype(jnp.float32)).astype(jnp.float32)


def is_terminal(state: EnvState, episode_steps: int = constants.DEFAULT_EPISODE_STEPS) -> jnp.ndarray:
    """Step cap reached OR <=1 player still has any ships."""
    step_done = state.step >= episode_steps
    alive_count = sum(player_alive(state, p) for p in range(constants.NUM_PLAYERS))
    elim_done = alive_count <= 1
    return step_done | elim_done
