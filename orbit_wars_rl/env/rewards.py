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

# ----- Day 4 shaping family (additive, all default 0 = backward compat) -----
#
# Empirically calibrated against 5 top-10 expert kaggle episodes (DAY4 §12).
# Winners share three signatures the v7 reward landscape is silent on:
#
#   (a) PROD_SHARE: winners hold 0.26-0.55 of global production capacity
#       vs losers <= 0.18 (4P baseline = 0.25). PERFECT separation across
#       5 games. This is the strongest single predictor of victory.
#
#   (b) PLANET_SHARE: winners average 11 planets vs losers <= 7 (4P
#       baseline = 8). Strongly correlated with (a) but happens EARLIER
#       in the episode -- expansion precedes garrison growth.
#
#   (c) FLEET_SIZE (log-scaled): max single launches 575-2076 ships;
#       p95 of launches 227-1379. Linear NORM=20 was way too small --
#       the reward saturates at 20 and the model never learns large
#       decisive strikes. Log scale keeps marginal reward all the way
#       to 1000+ ships.
#
# Two old terms (KEEP_HOME, FLEET_SIZE-v1) are retained for backward compat
# but default to 0 -- KEEP_HOME rewards the wrong behaviour (constant
# stockpile, whereas winners do stockpile-then-release cycles).
#
# Magnitudes calibrated so per-episode accumulated shaping <= terminal +/-1.
SHAPING_KEEP_HOME: float = float(_os.environ.get("ORBITWARS_SHAPING_KEEP_HOME", "0.0"))
SHAPING_FLEET_SIZE: float = float(_os.environ.get("ORBITWARS_SHAPING_FLEET_SIZE", "0.0"))
SHAPING_FLEET_NORM: float = float(_os.environ.get("ORBITWARS_SHAPING_FLEET_NORM", "20.0"))
SHAPING_FLEET_FLOOR: float = float(_os.environ.get("ORBITWARS_SHAPING_FLEET_FLOOR", "0.2"))

# --- Day 4 §12 (post-expert-replay) ---
#
# PROD_SHARE_REWARD: per-step alpha * (my_prod_share - 1/N).
#   Naturally zero-sum: winner gets +, loser gets -, both bounded in
#   [-(1-1/N), +(1-1/N)]. Single-step reward in [-0.005, +0.005] when
#   alpha=0.01 and N=2 -- 500 turns -> <= ±2.5 (matches terminal scale).
#
# PLANET_SHARE_REWARD: per-step beta * (my_planet_share - 1/N).
#   Early-episode signal (expansion precedes production growth). Smaller
#   coefficient because it duplicates part of PROD_SHARE info.
#
# FLEET_SIZE_LOG: per-launch gamma * (log1p(ships)/log1p(LOG_REF) - FLOOR).
#   Log scale -> 1 ship: -0.31*gamma; 20 ships: -0.013*gamma;
#   100 ships: +0.44*gamma; 500 ships: ~+0.7*gamma (capped); 2000 ships:
#   capped at +0.7*gamma. Replaces SHAPING_FLEET_SIZE v1 (linear+saturated).
SHAPING_PROD_SHARE: float = float(_os.environ.get("ORBITWARS_SHAPING_PROD_SHARE", "0.0"))
SHAPING_PLANET_SHARE: float = float(_os.environ.get("ORBITWARS_SHAPING_PLANET_SHARE", "0.0"))
SHAPING_FLEET_LOG: float = float(_os.environ.get("ORBITWARS_SHAPING_FLEET_LOG", "0.0"))
SHAPING_FLEET_LOG_REF: float = float(_os.environ.get("ORBITWARS_SHAPING_FLEET_LOG_REF", "500.0"))
SHAPING_FLEET_LOG_FLOOR: float = float(_os.environ.get("ORBITWARS_SHAPING_FLEET_LOG_FLOOR", "0.3"))


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


def keep_home_reward(state: EnvState, player: int) -> jnp.ndarray:
    """SHAPING_KEEP_HOME * tanh(log1p(home_ships) / 8).

    Reads the per-episode-fixed ``home_planet_idx`` from state. If the home
    has been captured (owner != player) we return 0 -- losing home stops the
    reward stream, which is the correct gradient direction.
    """
    home_idx = state.home_planet_idx[player]
    home_ships = state.planet_ships[home_idx].astype(jnp.float32)
    home_owner = state.planet_owner[home_idx]
    still_mine = home_owner == player
    raw = jnp.tanh(jnp.log1p(home_ships) / jnp.float32(8.0))
    return jnp.float32(SHAPING_KEEP_HOME) * jnp.where(still_mine, raw, jnp.float32(0.0))


def fleet_size_reward(valid_mask: jnp.ndarray, ships_per_launch: jnp.ndarray) -> jnp.ndarray:
    """SHAPING_FLEET_SIZE * sum_over_valid_launches(clip(ships/NORM,0,1) - FLOOR).

    [Day 4 v1; default coef = 0 since v9, replaced by fleet_size_log_reward.]

    Caller passes the VALID launch mask (after env applied its own checks)
    and the actual ships dispatched per launch. Both shapes are [K].

    The (ships/NORM - FLOOR) form is negative for tiny launches (<= FLOOR*NORM
    ships) and positive otherwise. NORM=20, FLOOR=0.2 → break-even at 4 ships,
    max +0.8 per launch at >= 20 ships.

    Returned shape: scalar. K=8 launches at max contribute SHAPING_FLEET_SIZE*8*0.8
    per turn = 6.4 * SHAPING_FLEET_SIZE -- keep coefficient small.
    """
    valid_f = valid_mask.astype(jnp.float32)
    ships_f = ships_per_launch.astype(jnp.float32)
    norm = jnp.float32(SHAPING_FLEET_NORM)
    floor = jnp.float32(SHAPING_FLEET_FLOOR)
    per_launch = jnp.clip(ships_f / norm, 0.0, 1.0) - floor
    return jnp.float32(SHAPING_FLEET_SIZE) * (per_launch * valid_f).sum()


# ===== Day 4 §12 (post-expert-replay) shaping family =====================


def _player_alive_planet_mask(state: EnvState, player: int) -> jnp.ndarray:
    return state.planet_mask & (state.planet_owner == player)


def prod_share_reward(state: EnvState, player: int) -> jnp.ndarray:
    """SHAPING_PROD_SHARE * (my_prod_share - 1/N).

    ``my_prod_share = sum(prod for planets I own) / sum(prod for ALL live planets)``.
    For 2-player env (N=2), baseline = 0.5; reward sign equals sign of
    "am I ahead in production capacity right now?". Naturally zero-sum.

    Live planets only (mask=True). If no live planets exist the divisor
    falls back to 1 so we return 0 instead of NaN.
    """
    mine_mask = _player_alive_planet_mask(state, player)
    live_mask = state.planet_mask
    my_prod = jnp.where(mine_mask, state.planet_prod, jnp.int32(0)).sum().astype(jnp.float32)
    total_prod = jnp.maximum(
        jnp.where(live_mask, state.planet_prod, jnp.int32(0)).sum().astype(jnp.float32),
        jnp.float32(1.0),
    )
    n_players = jnp.float32(2.0)  # current env is 2P; see DAY4 §12 for 4P notes
    return jnp.float32(SHAPING_PROD_SHARE) * (my_prod / total_prod - 1.0 / n_players)


def planet_share_reward(state: EnvState, player: int) -> jnp.ndarray:
    """SHAPING_PLANET_SHARE * (my_planet_share - 1/N).

    Counts live owned planets / total live planets. Early-episode signal
    (territory grows before garrison/production caps).
    """
    mine_mask = _player_alive_planet_mask(state, player)
    live_mask = state.planet_mask
    my_count = mine_mask.sum().astype(jnp.float32)
    total_count = jnp.maximum(live_mask.sum().astype(jnp.float32), jnp.float32(1.0))
    n_players = jnp.float32(2.0)
    return jnp.float32(SHAPING_PLANET_SHARE) * (my_count / total_count - 1.0 / n_players)


def fleet_size_log_reward(valid_mask: jnp.ndarray, ships_per_launch: jnp.ndarray) -> jnp.ndarray:
    """SHAPING_FLEET_LOG * sum_over_valid_launches(clip01(log1p(ships)/log1p(REF)) - FLOOR).

    Log-scaled fleet-size shaping motivated by expert replays (top winners
    launch 500-2000 ships; linear NORM=20 saturates immediately).

    LOG_REF=500 (saturation point), FLOOR=0.3:
      ships=1   -> log1p(1)/log1p(500) ~ 0.111 -> -0.189 * coef  (penalty)
      ships=4   -> ~0.259                       -> -0.041 * coef  (mild penalty)
      ships=10  -> ~0.386                       -> +0.086 * coef
      ships=100 -> ~0.741                       -> +0.441 * coef
      ships=500 -> 1.000                        -> +0.700 * coef
      ships=2000-> clipped to 1.0               -> +0.700 * coef (no over-reward)
    """
    valid_f = valid_mask.astype(jnp.float32)
    ships_f = ships_per_launch.astype(jnp.float32)
    log_ref = jnp.log1p(jnp.float32(SHAPING_FLEET_LOG_REF))
    raw = jnp.log1p(jnp.maximum(ships_f, 0.0)) / log_ref
    norm_clip = jnp.clip(raw, 0.0, 1.0)
    per_launch = norm_clip - jnp.float32(SHAPING_FLEET_LOG_FLOOR)
    return jnp.float32(SHAPING_FLEET_LOG) * (per_launch * valid_f).sum()


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
