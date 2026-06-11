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

# --- Day 5 (post-top10 deep-analysis) shaping family ---
#
# Motivation (TOP10_REPLAY_METRICS.zh.md, 2630 episodes):
#
# R1 PROD_SHARE_DELTA: winners gain prod_share +0.286 over an episode; losers
#   -0.020. The level-form PROD_SHARE rewards "being ahead" (history-dependent,
#   weak credit assignment). The delta form rewards "the action that just
#   moved share", giving PPO a tight gradient on the actual capture / loss.
#   Uses (next - prev) on the consolidated per-step prod_share, so the sum
#   over an episode telescopes to the episode-level prod_share change.
#
# R4 EMIT_LOG: rewards valid-launch effort per turn on a log scale -- log1p
#   so 0 launches = 0 reward, 1 launch = small, 2 = larger, etc. Unlike a
#   threshold-based multi-emit bonus this has no behavioural cutoff: there
#   is no incentive to "stop at exactly K" launches. Caller passes the
#   per-turn valid_mask (shape [K]).
#
# R2 RELEASE_BONUS: log-scaled fleet size * tanh(src_garr / src_prod /
#   RELEASE_K - 1). The source planet's stockpile is normalized by its own
#   per-turn production -- a high-prod planet is "expected" to hold more
#   before releasing. No EMA or stored history needed; the comparison uses
#   each planet's static prod, which is invariant across self-play strength.
SHAPING_PROD_SHARE_DELTA: float = float(
    _os.environ.get("ORBITWARS_SHAPING_PROD_SHARE_DELTA", "0.0")
)
SHAPING_EMIT_LOG: float = float(_os.environ.get("ORBITWARS_SHAPING_EMIT_LOG", "0.0"))
SHAPING_RELEASE: float = float(_os.environ.get("ORBITWARS_SHAPING_RELEASE", "0.0"))
SHAPING_RELEASE_K: float = float(_os.environ.get("ORBITWARS_SHAPING_RELEASE_K", "20.0"))
SHAPING_CAPTURE: float = float(_os.environ.get("ORBITWARS_SHAPING_CAPTURE", "0.0"))
# When 1, emit_log only applies if at least one valid launch has release_factor>0.
SHAPING_EMIT_GATED: float = float(_os.environ.get("ORBITWARS_SHAPING_EMIT_GATED", "0.0"))

# --- Day 8 / f35: anti-1-ship + high-prod capture ---
#
# Root cause (DAY8 §11/§12): 55% of launches send 1-3 ships from near-empty
# source planets. The smooth fleet_size_log floor barely penalizes this. A
# sharp per-launch penalty for tiny fleets gives PPO a direct gradient against
# the "trickle from empty shells" failure mode.
#
# ONE_SHIP_PENALTY: -coef per valid launch with ships <= ONE_SHIP_THRESH.
#   Flat penalty (not log) so it bites regardless of source size.
# HIGH_PROD_CAPTURE: extra reward when a newly-captured planet has prod>=THRESH,
#   scaled by its production -- pushes the policy to target factories, not mites.
SHAPING_ONE_SHIP_PENALTY: float = float(
    _os.environ.get("ORBITWARS_SHAPING_ONE_SHIP_PENALTY", "0.0")
)
SHAPING_ONE_SHIP_THRESH: float = float(
    _os.environ.get("ORBITWARS_SHAPING_ONE_SHIP_THRESH", "3.0")
)
SHAPING_HIGH_PROD_CAPTURE: float = float(
    _os.environ.get("ORBITWARS_SHAPING_HIGH_PROD_CAPTURE", "0.0")
)
SHAPING_HIGH_PROD_THRESH: float = float(
    _os.environ.get("ORBITWARS_SHAPING_HIGH_PROD_THRESH", "3.0")
)

# --- Day 11 / f42: fleet-scaled capture bonus ---
#
# Core insight: the model needs to learn "stockpile then strike with a large
# fleet" (Vadasz z0=34%, spf=25). Current CAPTURE rewards all flips equally
# regardless of fleet size, so trickle flips and decisive flips get the same
# signal. CAPTURE_FLEET_SCALE gives a multiplicative bonus proportional to
# the garrison left on a newly-captured planet (proxy for fleet size that
# arrived), creating a gradient: large-fleet capture >> trickle capture.
#
# Formula: coef * sum_over_gained( prod_i * log1p(garrison_i) ) / total_prod / 8
# The log1p/8 scale matches the fleet_size_log normalization.
# Only fires when a flip actually happens (0/1 direction), and scales with
# the investment that made it happen.
SHAPING_CAPTURE_FLEET_SCALE: float = float(
    _os.environ.get("ORBITWARS_SHAPING_CAPTURE_FLEET_SCALE", "0.0")
)

# --- Day 11 / f43: gated multi-emit bonus ---
#
# f41 ONE_SHIP_PENALTY killed e2+ (2nd/3rd routes are often smaller). f43
# replaces that tradeoff with a positive 0/1 gate: only reward 2+ launches
# when at least one fleet is "substantial" (>= MIN_SHIPS, default 8 = v20
# ABS_MIN_BATCH). Avoids paying for double-trickle while nudging multi-route
# bursts after a main strike.
SHAPING_MULTI_EMIT: float = float(
    _os.environ.get("ORBITWARS_SHAPING_MULTI_EMIT", "0.0")
)
SHAPING_MULTI_EMIT_MIN_SHIPS: float = float(
    _os.environ.get("ORBITWARS_SHAPING_MULTI_EMIT_MIN_SHIPS", "8.0")
)


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


def _strongest_opp_ships(state: EnvState, player: int) -> jnp.ndarray:
    """Strongest opponent by total ships (2P: the other player; 4P: max over others)."""
    if constants.NUM_PLAYERS == 2:
        return player_total_ships(state, 1 - player)
    s1 = player_total_ships(state, 1)
    s2 = player_total_ships(state, 2)
    s3 = player_total_ships(state, 3)
    if player == 0:
        return jnp.maximum(jnp.maximum(s1, s2), s3)
    if player == 1:
        return jnp.maximum(jnp.maximum(player_total_ships(state, 0), s2), s3)
    if player == 2:
        return jnp.maximum(jnp.maximum(player_total_ships(state, 0), s1), s3)
    return jnp.maximum(jnp.maximum(player_total_ships(state, 0), s1), s2)


def shaping_potential(state: EnvState, player: int) -> jnp.ndarray:
    """Bounded potential function: tanh of normalized ship difference.

    Used to compute a potential-based shaping reward ``F = phi(s') - phi(s)``.
    The terminal value of ``phi`` is at most ~1, which is on the same order
    as the terminal +/-1 reward; ``SHAPING_SCALE`` is applied at the call site.
    """
    me = player_total_ships(state, player).astype(jnp.float32)
    opp = _strongest_opp_ships(state, player).astype(jnp.float32)
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
    Baseline = ``1 / constants.NUM_PLAYERS`` so the reward is naturally zero-sum
    and stays correct under any player-count change.

    Live planets only (mask=True). If no live planets exist the divisor
    falls back to 1 so we return 0 instead of NaN.
    """
    share = _my_prod_share(state, player)
    n_players = jnp.float32(constants.NUM_PLAYERS)
    return jnp.float32(SHAPING_PROD_SHARE) * (share - 1.0 / n_players)


def _my_prod_share(state: EnvState, player: int) -> jnp.ndarray:
    """Pure helper: my owned production / total live production."""
    mine_mask = _player_alive_planet_mask(state, player)
    live_mask = state.planet_mask
    my_prod = jnp.where(mine_mask, state.planet_prod, jnp.int32(0)).sum().astype(jnp.float32)
    total_prod = jnp.maximum(
        jnp.where(live_mask, state.planet_prod, jnp.int32(0)).sum().astype(jnp.float32),
        jnp.float32(1.0),
    )
    return my_prod / total_prod


def prod_share_delta_reward(
    prev_state: EnvState, next_state: EnvState, player: int
) -> jnp.ndarray:
    """SHAPING_PROD_SHARE_DELTA * (share(next) - share(prev)).

    Per-step credit-assigning version of prod_share. Telescopes over an
    episode to ``coef * (share_end - share_start)``. Unlike the level form
    there is no constant baseline subtraction, so the per-step reward is
    zero when nothing changes (zero-sum on capture / loss events).
    """
    return jnp.float32(SHAPING_PROD_SHARE_DELTA) * (
        _my_prod_share(next_state, player) - _my_prod_share(prev_state, player)
    )


def planet_share_reward(state: EnvState, player: int) -> jnp.ndarray:
    """SHAPING_PLANET_SHARE * (my_planet_share - 1/N).

    Counts live owned planets / total live planets. Early-episode signal
    (territory grows before garrison/production caps). Uses
    ``constants.NUM_PLAYERS`` for the fair baseline.
    """
    mine_mask = _player_alive_planet_mask(state, player)
    live_mask = state.planet_mask
    my_count = mine_mask.sum().astype(jnp.float32)
    total_count = jnp.maximum(live_mask.sum().astype(jnp.float32), jnp.float32(1.0))
    n_players = jnp.float32(constants.NUM_PLAYERS)
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


def _release_factors(
    state: EnvState,
    src_idx: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Per-launch release_factor in (-1, +1); shape [K]."""
    src_idx_clip = jnp.clip(src_idx, 0, constants.MAX_PLANETS - 1)
    src_garr = state.planet_ships[src_idx_clip].astype(jnp.float32)
    src_prod = jnp.maximum(
        state.planet_prod[src_idx_clip].astype(jnp.float32),
        jnp.float32(1.0),
    )
    generations = src_garr / src_prod / jnp.float32(SHAPING_RELEASE_K)
    return jnp.tanh(generations - 1.0)


def emit_log_reward(valid_mask: jnp.ndarray) -> jnp.ndarray:
    """SHAPING_EMIT_LOG * log1p(num_valid_launches_this_turn).

    Rewards "actually emitted something" per turn on a saturating log scale.
    With K=8 launch slots the upper bound per turn is ``log1p(8) ~ 2.197``,
    so the per-turn upper reward is ``SHAPING_EMIT_LOG * 2.197``.

    Unlike a threshold "multi_emit >= 2" bonus this has no behavioural
    cutoff: marginal reward of going from 0->1, 1->2, 2->3 all monotonically
    decreases (log curvature) without any cliff. Returns 0 when no launches
    are valid (idle turn). Combined with ``fleet_size_log_reward`` this
    decouples "did I launch?" from "how big was the launch?".
    """
    valid_f = valid_mask.astype(jnp.float32)
    n_valid = valid_f.sum()
    return jnp.float32(SHAPING_EMIT_LOG) * jnp.log1p(n_valid)


def release_bonus_reward(
    state: EnvState,
    src_idx: jnp.ndarray,
    valid_mask: jnp.ndarray,
    ships_per_launch: jnp.ndarray,
) -> jnp.ndarray:
    """SHAPING_RELEASE * sum_per_launch(log_size * release_factor) for valid launches.

    The "release factor" is ``tanh(src_garr / src_prod / RELEASE_K - 1)``: the
    source planet's stockpile is normalized by its own per-turn production,
    yielding "how many turns of self-production are sitting here". A launch
    from a planet stockpiled to RELEASE_K-turns-of-own-production gives 0;
    above that the bonus rises smoothly toward +1 * log_size; below it
    decays toward -1 * log_size.

    Two key properties:
      1.  No fixed ship-count threshold; the baseline scales with each
          planet's intrinsic production. A high-prod planet is "expected"
          to hold more ships before releasing.
      2.  No state-stored EMA; the comparison uses the planet's static
          per-step production, which is invariant to self-play strength.

    Inputs come from ``dynamics.launch_fleets_with_info``:
      * ``src_idx`` shape [K] int32: source planet index per launch slot
      * ``valid_mask`` shape [K] bool
      * ``ships_per_launch`` shape [K] int32
    """
    valid_f = valid_mask.astype(jnp.float32)
    ships_f = ships_per_launch.astype(jnp.float32)

    log_ref = jnp.log1p(jnp.float32(SHAPING_FLEET_LOG_REF))
    log_size = jnp.clip(
        jnp.log1p(jnp.maximum(ships_f, 0.0)) / log_ref, 0.0, 1.0
    )  # [K]

    release_factor = _release_factors(state, src_idx, valid_mask)
    per_launch = log_size * release_factor
    return jnp.float32(SHAPING_RELEASE) * (per_launch * valid_f).sum()


def emit_log_reward_gated(
    valid_mask: jnp.ndarray,
    state: EnvState,
    src_idx: jnp.ndarray,
) -> jnp.ndarray:
    """emit_log scaled by 0/1 gate: only reward emit when releasing stockpile."""
    base = emit_log_reward(valid_mask)
    if SHAPING_EMIT_GATED <= 0.0:
        return base
    valid_f = valid_mask.astype(jnp.float32)
    release_factor = _release_factors(state, src_idx, valid_mask)
    has_release = ((release_factor * valid_f) > 0.0).any()
    return jnp.where(has_release, base, jnp.float32(0.0))


def one_ship_penalty_reward(
    valid_mask: jnp.ndarray, ships_per_launch: jnp.ndarray
) -> jnp.ndarray:
    """-SHAPING_ONE_SHIP_PENALTY per valid launch with ships <= ONE_SHIP_THRESH.

    Flat per-launch penalty (NOT log-scaled) for tiny "trickle" fleets. With
    K=8 slots the worst case is ``-coef * 8`` per turn, so keep coef small
    (e.g. 0.01 -> -0.08/turn floor). Returns 0 when the coef is 0 so existing
    configs are reward-bit-exact.
    """
    if SHAPING_ONE_SHIP_PENALTY <= 0.0:
        return jnp.float32(0.0)
    valid_f = valid_mask.astype(jnp.float32)
    ships_f = ships_per_launch.astype(jnp.float32)
    tiny = (ships_f <= jnp.float32(SHAPING_ONE_SHIP_THRESH)).astype(jnp.float32)
    n_tiny = (tiny * valid_f).sum()
    return -jnp.float32(SHAPING_ONE_SHIP_PENALTY) * n_tiny


def high_prod_capture_reward(
    prev_state: EnvState, next_state: EnvState, player: int
) -> jnp.ndarray:
    """SHAPING_HIGH_PROD_CAPTURE * sum(prod) over newly-captured prod>=THRESH planets.

    Rewards capturing factories specifically (v20 target_score prioritizes
    high-prod neutrals). Normalized by total production so the scale matches
    the other share-based terms. Returns 0 when coef is 0.
    """
    if SHAPING_HIGH_PROD_CAPTURE <= 0.0:
        return jnp.float32(0.0)
    prev_mine = (prev_state.planet_owner == player) & prev_state.planet_mask
    next_mine = (next_state.planet_owner == player) & next_state.planet_mask
    gained = next_mine & jnp.logical_not(prev_mine)
    prod_f = next_state.planet_prod.astype(jnp.float32)
    is_high = prod_f >= jnp.float32(SHAPING_HIGH_PROD_THRESH)
    prod_gained = jnp.where(gained & is_high, prod_f, jnp.float32(0.0)).sum()
    total_prod = jnp.maximum(prod_f.sum(), jnp.float32(1.0))
    return jnp.float32(SHAPING_HIGH_PROD_CAPTURE) * prod_gained / total_prod


def capture_flip_reward(
    prev_state: EnvState, next_state: EnvState, player: int
) -> jnp.ndarray:
    """SHAPING_CAPTURE * sum(prod on newly captured planets) / total_prod."""
    if SHAPING_CAPTURE <= 0.0:
        return jnp.float32(0.0)
    prev_mine = (prev_state.planet_owner == player) & prev_state.planet_mask
    next_mine = (next_state.planet_owner == player) & next_state.planet_mask
    gained = next_mine & jnp.logical_not(prev_mine)
    prod_gained = jnp.where(
        gained, next_state.planet_prod.astype(jnp.float32), jnp.float32(0.0)
    ).sum()
    total_prod = jnp.maximum(
        next_state.planet_prod.astype(jnp.float32).sum(), jnp.float32(1.0)
    )
    return jnp.float32(SHAPING_CAPTURE) * prod_gained / total_prod


def capture_fleet_scale_reward(
    prev_state: EnvState, next_state: EnvState, player: int
) -> jnp.ndarray:
    """Bonus for capturing with large fleets: incentivizes stockpile-then-strike.

    Only fires when a planet actually flips (0/1 gate). The bonus scales with
    both the planet's production value and the garrison left after capture
    (proxy for fleet size minus defender). This creates a gradient chain:
      large garrison on capture <- large fleet sent <- stockpiled before sending <- z0
    """
    if SHAPING_CAPTURE_FLEET_SCALE <= 0.0:
        return jnp.float32(0.0)
    prev_mine = (prev_state.planet_owner == player) & prev_state.planet_mask
    next_mine = (next_state.planet_owner == player) & next_state.planet_mask
    gained = next_mine & jnp.logical_not(prev_mine)
    prod_f = next_state.planet_prod.astype(jnp.float32)
    garr_f = next_state.planet_ships.astype(jnp.float32)
    total_prod = jnp.maximum(prod_f.sum(), jnp.float32(1.0))
    scale = jnp.log1p(garr_f) / jnp.float32(8.0)
    weighted = jnp.where(gained, prod_f * scale, jnp.float32(0.0)).sum()
    return jnp.float32(SHAPING_CAPTURE_FLEET_SCALE) * weighted / total_prod


def multi_emit_gated_bonus_reward(
    valid_mask: jnp.ndarray, ships_per_launch: jnp.ndarray
) -> jnp.ndarray:
    """Bonus for 2+ valid launches when the largest fleet is substantial.

    0/1 gate style: no reward for idle or single-launch turns; no reward for
    multi-trickle (all launches <= MIN_SHIPS). Aligns with Vadasz "one big
    burst + secondary routes" without ONE_SHIP_PENALTY punishing small 2nd legs.
    """
    if SHAPING_MULTI_EMIT <= 0.0:
        return jnp.float32(0.0)
    valid_f = valid_mask.astype(jnp.float32)
    n_valid = valid_f.sum()
    ships_f = ships_per_launch.astype(jnp.float32)
    max_ships = jnp.max(jnp.where(valid_mask, ships_f, jnp.float32(0.0)))
    multi = n_valid >= jnp.float32(2.0)
    substantial = max_ships >= jnp.float32(SHAPING_MULTI_EMIT_MIN_SHIPS)
    return jnp.float32(SHAPING_MULTI_EMIT) * jnp.where(
        multi & substantial, jnp.float32(1.0), jnp.float32(0.0)
    )


SHAPING_HOLD_BONUS: float = float(
    _os.environ.get("ORBITWARS_SHAPING_HOLD_BONUS", "0.0")
)

# --- v14e: anti-hoard penalty ---
#
# Structural fix for hoarding collapse observed in v14d: agent discovers "never
# emit = never lose garrison = high garr reward" and locks into z0>90%.
# This penalty fires when the agent's garrison ratio (my_garr / total_garr) is
# high AND it chose not to emit, creating a gradient against the hoarding
# equilibrium. The penalty scales smoothly with how far above the threshold the
# agent is, capped at 1.0.
#
# Formula: -coef * max(0, garr_ratio - thresh) / (1 - thresh) when not emitting.
# With thresh=0.6, garr_ratio=0.8 → penalty = -coef * 0.5
# With thresh=0.6, garr_ratio=1.0 → penalty = -coef * 1.0 (cap)
SHAPING_ANTI_HOARD: float = float(
    _os.environ.get("ORBITWARS_SHAPING_ANTI_HOARD", "0.0")
)
SHAPING_ANTI_HOARD_THRESH: float = float(
    _os.environ.get("ORBITWARS_SHAPING_ANTI_HOARD_THRESH", "0.6")
)


def hold_bonus_reward(
    valid_mask: jnp.ndarray,
    state: EnvState,
    player: int,
) -> jnp.ndarray:
    """Small positive reward when the agent holds (zero valid launches).

    Bridges the sparse reward gap by giving immediate credit for accumulating
    garrison. Only fires when no fleets were launched this turn.
    """
    if SHAPING_HOLD_BONUS <= 0.0:
        return jnp.float32(0.0)
    valid_f = valid_mask.astype(jnp.float32)
    n_valid = valid_f.sum()
    is_hold = n_valid == jnp.float32(0.0)
    my_mask = state.planet_owner == player
    my_garr = jnp.where(
        my_mask & state.planet_mask,
        state.planet_ships.astype(jnp.float32),
        jnp.float32(0.0),
    ).sum()
    has_garrison = my_garr > jnp.float32(20.0)
    return jnp.float32(SHAPING_HOLD_BONUS) * jnp.where(
        is_hold & has_garrison, jnp.float32(1.0), jnp.float32(0.0)
    )


def anti_hoard_penalty_reward(
    valid_mask: jnp.ndarray,
    state: EnvState,
    player: int,
) -> jnp.ndarray:
    """Negative reward when hoarding: high garrison ratio + zero launches.

    Fires when the agent's share of total garrison exceeds ANTI_HOARD_THRESH
    and it chose not to emit this turn. The penalty scales linearly from 0 at
    the threshold to -coef at ratio=1.0, preventing the "never emit" collapse.
    """
    if SHAPING_ANTI_HOARD <= 0.0:
        return jnp.float32(0.0)
    valid_f = valid_mask.astype(jnp.float32)
    n_valid = valid_f.sum()
    is_hold = n_valid == jnp.float32(0.0)

    my_mask = (state.planet_owner == player) & state.planet_mask
    my_garr = jnp.where(
        my_mask, state.planet_ships.astype(jnp.float32), jnp.float32(0.0)
    ).sum()
    total_garr = jnp.where(
        state.planet_mask, state.planet_ships.astype(jnp.float32), jnp.float32(0.0)
    ).sum()
    garr_ratio = my_garr / jnp.maximum(total_garr, jnp.float32(1.0))

    thresh = jnp.float32(SHAPING_ANTI_HOARD_THRESH)
    excess = jnp.clip((garr_ratio - thresh) / (1.0 - thresh), 0.0, 1.0)
    return -jnp.float32(SHAPING_ANTI_HOARD) * jnp.where(
        is_hold, excess, jnp.float32(0.0)
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
    scores = jnp.stack(
        [player_total_ships(state, p) for p in range(constants.NUM_PLAYERS)],
        axis=0,
    )
    max_score = scores.max()
    i_win = (scores[player] == max_score) & (max_score > 0)
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
