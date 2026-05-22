"""Reward shaping.

* Terminal: +1 win / -1 loss / 0 draw based on final ship totals.
* Per-step shaping: a small bonus proportional to the *change* in
  (my ships - opp ships), normalized by a reference total. This gives
  the value head/policy a denser signal so it doesn't have to wait
  for the terminal +/-1 200 steps in the future.

The shaping is potential-based-ish (delta of a static potential) so it
shouldn't change the optimal policy in the limit, only speed up learning.
"""

from __future__ import annotations

import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.state import EnvState


# Scale for dense shaping. Small enough that the terminal +/-1 still dominates
# the long-horizon return but large enough to give a useful gradient mid-game.
SHAPING_SCALE: float = 0.1

# Reference ship total used to normalize the (delta my - delta opp) signal.
# Picked so a typical mid-game ship-balance swing of ~10 ships yields a
# noticeable delta in the bounded tanh potential.
SHAPING_REF: float = 30.0


def player_total_ships(state: EnvState, player: int) -> jnp.ndarray:
    """Total ships (planets + fleets) for ``player``, masked by validity."""
    planet_mine = (state.planet_owner == player) & state.planet_mask
    fleet_mine = (state.fleet_owner == player) & state.fleet_mask
    planet_sum = jnp.where(planet_mine, state.planet_ships, 0).sum()
    fleet_sum = jnp.where(fleet_mine, state.fleet_ships, 0).sum()
    return (planet_sum + fleet_sum).astype(jnp.int32)


def shaping_potential(state: EnvState, player: int) -> jnp.ndarray:
    """Bounded potential function: tanh of normalized ship difference.

    Used to compute a potential-based shaping reward ``F = phi(s') - phi(s)``.
    The terminal value of ``phi`` is at most ~1, which is on the same order
    as the terminal +/-1 reward; ``SHAPING_SCALE`` is applied at the call site.
    """
    me = player_total_ships(state, player).astype(jnp.float32)
    opp = player_total_ships(state, 1 - player).astype(jnp.float32)
    diff = (me - opp) / jnp.float32(SHAPING_REF)
    return jnp.tanh(diff)


def shaping_delta(prev_state: EnvState, next_state: EnvState, player: int) -> jnp.ndarray:
    """Potential-based shaping reward F = phi(next) - phi(prev), scaled."""
    return jnp.float32(SHAPING_SCALE) * (
        shaping_potential(next_state, player) - shaping_potential(prev_state, player)
    )


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
