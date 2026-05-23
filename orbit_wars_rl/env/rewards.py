"""Reward / termination — kept in lock-step with kaggle_environments orbit_wars.

Reference (kaggle envs/orbit_wars/orbit_wars.py lines 684-715):

    terminated = False
    if step >= configuration.episodeSteps - 2:
        terminated = True
    if len(alive_players) <= 1:
        terminated = True
    if terminated:
        scores = [planet_ships + fleet_ships for each player]
        max_score = max(scores)
        for i in range(num_agents):
            if scores[i] == max_score and max_score > 0:
                reward = +1
            else:
                reward = -1

Three subtle behaviours of the kaggle rule we MUST match:

1.  ``scores[i] == max_score`` includes ties. If both players end with
    250 ships, BOTH get +1 (it's a draw on kaggle's leaderboard but the
    reward signal is +1 / +1, not 0 / 0). The 1st-place player's
    "+1 -1 is enough" post on the kaggle forum (top_players_rl.txt §73)
    assumes this rule.

2.  ``and max_score > 0`` means double-elimination (both players reached
    0 ships) yields -1 / -1, not +1 / +1. This case is rare but real
    when both home planets are wiped out simultaneously.

3.  Termination at ``step >= episodeSteps - 2`` (NOT ``>= episodeSteps``).
    We match this with ``state.step >= episode_steps - 2``.

Day 3 audit found we had +1/0/-1 (ties => 0). That meant:
   * two no-op players for 80 turns -> our env returns 0, kaggle returns +1
   * which means our trained policies cannot distinguish "boring tie" from
     "really lose" in their reward landscape, even though kaggle treats
     them very differently. Fixed below.

Per-step shaping (DAY2 §10+):
   * Set to 0 by default since top_players_rl.txt §73 says +1/-1 is enough.
   * Override via env var ORBITWARS_SHAPING_SCALE=0.1 if you want to bring
     it back. The shaping function itself is unchanged, just default-off
     so we don't accidentally train with shaping that nobody asked for.
"""

from __future__ import annotations

import os as _os

import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.state import EnvState


# Default = 0.0 (sparse +1/-1 reward, top1 §73). Pre-Day-3 default was 0.1;
# every config doc since v5p2 launched with ORBITWARS_SHAPING_SCALE=0.0.
# Making 0.0 the default closes the "forgot to set env var" trap.
SHAPING_SCALE: float = float(_os.environ.get("ORBITWARS_SHAPING_SCALE", "0.0"))

# Reference ship total used to normalize the (delta my - delta opp) signal.
SHAPING_REF: float = 30.0


def player_total_ships(state: EnvState, player: int) -> jnp.ndarray:
    """Total ships (planets + fleets) for ``player``, masked by validity.

    Matches kaggle's score computation: planets[i][5] (ships) + fleets[i][6]
    summed over only-mine entities.
    """
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
    """+1 / -1 reward, matching kaggle_environments orbit_wars.py:710-715.

    Rule: ``+1`` iff this player's score equals the max score AND max > 0.
    Everyone else (including the loser of a 5-vs-250 game, and BOTH players
    in a 0-vs-0 mutual-elimination) gets ``-1``.

    Notable consequence: a ship-count tie (both players end with the same
    non-zero total) yields ``+1`` for BOTH players, NOT a 0/0 draw.

    Only meaningful when ``done`` is True; callers should mask otherwise.
    """
    me = player_total_ships(state, player)
    opp = player_total_ships(state, 1 - player)
    max_score = jnp.maximum(me, opp)
    # I "win" iff I have the max and the max is > 0.
    i_win = (me == max_score) & (max_score > 0)
    return jnp.where(i_win, jnp.float32(1.0), jnp.float32(-1.0))


def is_terminal(state: EnvState, episode_steps: int = constants.DEFAULT_EPISODE_STEPS) -> jnp.ndarray:
    """Step cap reached OR <=1 player still has any ships.

    Kaggle: ``step >= episodeSteps - 2`` (kaggle env orbit_wars.py:686).
    We match that exactly: with ``episode_steps`` interpreted as the kaggle
    ``episodeSteps`` config value, the env ends when ``state.step >= episode_steps - 2``.
    """
    step_done = state.step >= (episode_steps - 2)
    alive_count = sum(player_alive(state, p) for p in range(constants.NUM_PLAYERS))
    elim_done = alive_count <= 1
    return step_done | elim_done
