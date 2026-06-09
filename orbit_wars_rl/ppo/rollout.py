"""Rollout collection for PPO. Single-jit over scan + vmap over envs.

Player 0 is the learning agent; player 1 is an opponent policy.

The rollout collects **multi-fleet** actions: at each turn the policy emits
up to ``K = constants.MAX_FLEETS_PER_TURN`` (src, dst, pct) triples along
with a per-step ``emit_mask``. Per-step logp / entropy / value are stored
so the PPO update can recompute ratios under the new params.

Three rollout variants are exposed:

* ``make_rollout_fn(..., opponent_fn=random_opponent_action)`` -- fixed
  opponent. ``opponent_fn`` produces a *single* legacy ``PlayerAction`` that
  the env wraps to K=1 internally.
* ``make_rollout_fn_with_frozen_opp(model, ...)`` -- self-play variant where
  the opponent uses the same multi-action ``ActorCritic`` with a frozen
  parameter set.
* ``make_rollout_fn_with_buffer_reset(model, ..., buffer_states, reset_prob)``
  -- self-play variant that, on episode reset, replaces the freshly-generated
  random start state with a state sampled from a pre-collected buffer with
  probability ``reset_prob``.  This is the "v20 state-buffer curriculum"
  (Plan B) that seeds the learner into high-value mid-game positions extracted
  from v20 self-play.

All return jit'd closures with signature
``rollout_fn(params, states, rngs[, frozen_params]) -> (states, rngs, Rollout)``.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import chex
import jax
import jax.numpy as jnp

from orbit_wars_rl.env import OrbitWarsEnv, constants
from orbit_wars_rl.env.actions import (
    MultiPlayerAction,
    PlayerAction,
    single_to_multi,
)
from orbit_wars_rl.env.state import EnvState
from orbit_wars_rl.features import EncodedObs, encode
from orbit_wars_rl.net.model import ActorCritic


@chex.dataclass(frozen=True)
class Rollout:
    """[T, B, ...] tensors collected per rollout. B = num_envs.

    Action fields carry an extra trailing ``K`` axis (max fleets per turn).
    ``emit_mask`` is True for steps that actually launched a fleet -- PPO uses
    it to gate per-step logp/entropy.
    """

    obs_planet_feats: chex.Array
    obs_planet_mask: chex.Array
    obs_fleet_feats: chex.Array
    obs_fleet_mask: chex.Array
    obs_global_feats: chex.Array
    obs_my_planet_mask: chex.Array

    planet_ships_raw: chex.Array  # [T, B, P] int32 -- garrison at turn start
    planet_prod_raw: chex.Array   # [T, B, P] int32 -- production at turn start
    planet_owner_raw: chex.Array  # [T, B, P] int8  -- owner at turn start

    # f26: positions + home idx needed for K-loop pair features.
    planet_x_raw: chex.Array      # [T, B, P] float32
    planet_y_raw: chex.Array      # [T, B, P] float32
    home_idx_raw: chex.Array      # [T, B] int32 -- this player's home slot

    src_idx: chex.Array          # [T, B, K]
    dst_idx: chex.Array          # [T, B, K]
    pct_bin: chex.Array          # [T, B, K]
    emit_mask: chex.Array        # [T, B, K] bool   -- True iff this step actually launches
    emit_free_mask: chex.Array   # [T, B, K] bool   -- True iff emit head was a free choice

    src_logp: chex.Array         # [T, B, K] float -- per-step logp; 0 where !emit
    dst_logp: chex.Array         # [T, B, K]
    pct_logp: chex.Array         # [T, B, K]
    emit_logp: chex.Array        # [T, B, K]

    value: chex.Array            # [T, B]

    reward: chex.Array           # [T, B]
    done: chex.Array             # [T, B]

    last_value: chex.Array       # [B]

    # Zero-sum value head: opponent obs (None / zeros when not using zero-sum).
    opp_planet_feats: chex.Array | None = None   # [T, B, P, F]
    opp_planet_mask: chex.Array | None = None    # [T, B, P]
    opp_fleet_feats: chex.Array | None = None    # [T, B, F_max, F_dim]
    opp_fleet_mask: chex.Array | None = None     # [T, B, F_max]
    opp_global_feats: chex.Array | None = None   # [T, B, G]
    opp_my_planet_mask: chex.Array | None = None # [T, B, P]


OpponentFn = Callable[[jnp.ndarray, EncodedObs], PlayerAction]


def random_opponent_action(rng: jnp.ndarray, obs: EncodedObs) -> PlayerAction:
    """Pick a random owned source and random other planet as target.

    Single-fleet legacy action -- the env wraps it to a K=1 MultiPlayerAction.
    """
    my_mask = obs.my_planet_mask
    all_mask = obs.planet_mask
    rng_src, rng_dst, rng_pct = jax.random.split(rng, 3)

    src_logits = jnp.where(my_mask, jnp.float32(0.0), jnp.float32(-1e9))
    src_idx = jax.random.categorical(rng_src, src_logits)

    dst_valid = all_mask & jnp.logical_not(jnp.equal(jnp.arange(all_mask.shape[0]), src_idx))
    dst_logits = jnp.where(dst_valid, jnp.float32(0.0), jnp.float32(-1e9))
    dst_idx = jax.random.categorical(rng_dst, dst_logits)

    pct_bin = jax.random.randint(rng_pct, (), 0, constants.NUM_PCT_BINS)

    return PlayerAction(
        src_idx=src_idx.astype(jnp.int32),
        dst_idx=dst_idx.astype(jnp.int32),
        pct_bin=pct_bin.astype(jnp.int32),
    )


def _sampled_to_multi(sampled) -> MultiPlayerAction:
    return MultiPlayerAction(
        src_idx=sampled.src_idx,
        dst_idx=sampled.dst_idx,
        pct_bin=sampled.pct_bin,
        emit_mask=sampled.emit_mask,
    )


def _per_step_dict_from_sample(obs0, state, sampled, out):
    """Build the per-step dict written into the rollout for one env step."""
    return dict(
        obs_planet_feats=obs0.planet_feats,
        obs_planet_mask=obs0.planet_mask,
        obs_fleet_feats=obs0.fleet_feats,
        obs_fleet_mask=obs0.fleet_mask,
        obs_global_feats=obs0.global_feats,
        obs_my_planet_mask=obs0.my_planet_mask,
        planet_ships_raw=state.planet_ships,
        planet_prod_raw=state.planet_prod,
        planet_owner_raw=state.planet_owner,
        # f26: pair feats need geometry + home idx at turn-start.
        planet_x_raw=state.planet_x,
        planet_y_raw=state.planet_y,
        home_idx_raw=state.home_planet_idx[0],
        src_idx=sampled.src_idx,
        dst_idx=sampled.dst_idx,
        pct_bin=sampled.pct_bin,
        emit_mask=sampled.emit_mask,
        emit_free_mask=sampled.emit_free_mask,
        src_logp=sampled.src_logp,
        dst_logp=sampled.dst_logp,
        pct_logp=sampled.pct_logp,
        emit_logp=sampled.emit_logp,
        value=sampled.value,
        reward=out.reward,
        done=out.done,
    )


def _rollout_from_traj(traj_swapped: dict, last_values: jnp.ndarray) -> "Rollout":
    return Rollout(
        obs_planet_feats=traj_swapped["obs_planet_feats"],
        obs_planet_mask=traj_swapped["obs_planet_mask"],
        obs_fleet_feats=traj_swapped["obs_fleet_feats"],
        obs_fleet_mask=traj_swapped["obs_fleet_mask"],
        obs_global_feats=traj_swapped["obs_global_feats"],
        obs_my_planet_mask=traj_swapped["obs_my_planet_mask"],
        planet_ships_raw=traj_swapped["planet_ships_raw"],
        planet_prod_raw=traj_swapped["planet_prod_raw"],
        planet_owner_raw=traj_swapped["planet_owner_raw"],
        planet_x_raw=traj_swapped["planet_x_raw"],
        planet_y_raw=traj_swapped["planet_y_raw"],
        home_idx_raw=traj_swapped["home_idx_raw"],
        src_idx=traj_swapped["src_idx"],
        dst_idx=traj_swapped["dst_idx"],
        pct_bin=traj_swapped["pct_bin"],
        emit_mask=traj_swapped["emit_mask"],
        emit_free_mask=traj_swapped["emit_free_mask"],
        src_logp=traj_swapped["src_logp"],
        dst_logp=traj_swapped["dst_logp"],
        pct_logp=traj_swapped["pct_logp"],
        emit_logp=traj_swapped["emit_logp"],
        value=traj_swapped["value"],
        reward=traj_swapped["reward"],
        done=traj_swapped["done"],
        last_value=last_values,
    )


def make_rollout_fn(
    env: OrbitWarsEnv,
    model: ActorCritic,
    rollout_length: int,
    num_envs: int,
    opponent_fn: OpponentFn = random_opponent_action,
    episode_steps: int = constants.DEFAULT_EPISODE_STEPS,
):
    """Build a jit'd ``(params, state, rng) -> (state, rng, rollout)`` closure."""

    def _one_env_step(carry, _):
        state, rng, params = carry
        rng, r_act, r_opp, r_reset = jax.random.split(rng, 4)

        obs0 = encode(state, 0, episode_steps)
        obs1 = encode(state, 1, episode_steps)

        sampled = model.apply(
            params, obs0, r_act, state.planet_ships,
            state.planet_x, state.planet_y, state.home_planet_idx[0],
        )
        action_0 = _sampled_to_multi(sampled)
        opp_single = opponent_fn(r_opp, obs1)
        action_1 = single_to_multi(opp_single)

        next_state, out = env.step_and_autoreset(state, (action_0, action_1), r_reset)
        per_step = _per_step_dict_from_sample(obs0, state, sampled, out)
        new_carry = (next_state, rng, params)
        return new_carry, per_step

    def _scan_one_env(state: EnvState, rng: jnp.ndarray, params) -> tuple[EnvState, dict, jnp.ndarray]:
        (final_state, rng_out, _), traj = jax.lax.scan(
            _one_env_step, (state, rng, params), xs=None, length=rollout_length
        )
        final_obs0 = encode(final_state, 0, episode_steps)
        sampled_final = model.apply(
            params, final_obs0,
            jax.random.fold_in(rng_out, 1),
            final_state.planet_ships,
            final_state.planet_x, final_state.planet_y,
            final_state.home_planet_idx[0],
        )
        return final_state, traj, sampled_final.value

    def rollout_fn(
        params,
        states: EnvState,
        rngs: jnp.ndarray,
    ) -> tuple[EnvState, jnp.ndarray, Rollout]:
        final_states, trajs, last_values = jax.vmap(_scan_one_env, in_axes=(0, 0, None))(
            states, rngs, params
        )
        traj_swapped = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), trajs)
        rollout = _rollout_from_traj(traj_swapped, last_values)
        out_rngs = jax.vmap(lambda r: jax.random.fold_in(r, rollout_length + 7))(rngs)
        return final_states, out_rngs, rollout

    return jax.jit(rollout_fn)


# ---------------------------------------------------------------------------
# Buffer-reset rollout (Plan B: v20 state-buffer curriculum)
# ---------------------------------------------------------------------------

def make_rollout_fn_with_buffer_reset(
    env: OrbitWarsEnv,
    model: ActorCritic,
    rollout_length: int,
    num_envs: int,
    buffer_states: EnvState,   # [N, ...] stacked states from collect_states.py
    reset_prob: float = 0.30,  # probability of using a buffer state on episode reset
    episode_steps: int = constants.DEFAULT_EPISODE_STEPS,
):
    """Self-play rollout that seeds episode resets from a pre-collected buffer.

    On each episode reset (``done=True``), with probability ``reset_prob`` a
    state is drawn uniformly from ``buffer_states`` instead of generating a
    fresh random map.  The drawn state has ``step`` and ``done`` fields zeroed
    so it looks like a fresh episode to the policy and value head.

    The opponent is a frozen snapshot of the learner (same call-site as
    ``make_rollout_fn_with_frozen_opp``).  Pass ``frozen_params`` equal to
    the current ``params`` if you don't want any frozen-opp mixing.

    Arguments
    ---------
    buffer_states : EnvState
        A *batched* ``EnvState`` with a leading axis of size N (the buffer
        size).  All arrays must have shape ``[N, ...]``.  Typically loaded via
        ``load_state_buffer()``.
    reset_prob : float
        Probability in [0, 1] that a buffer state is used instead of a random
        one.  ``0.0`` degrades to the standard random-reset rollout.
        ``1.0`` means every reset draws from the buffer.
    """
    n_buffer = jax.tree_util.tree_leaves(buffer_states)[0].shape[0]

    def _sample_buffer_state(rng: jnp.ndarray) -> EnvState:
        """Draw one state from the buffer at random and zero out step/done."""
        idx = jax.random.randint(rng, (), 0, n_buffer)
        raw = jax.tree_util.tree_map(lambda x: x[idx], buffer_states)
        return raw.replace(
            step=jnp.int32(0),
            done=jnp.bool_(False),
            # Keep rng field intact; it's just entropy storage.
        )

    def _one_env_step(carry, _):
        state, rng, params, fparams = carry
        rng, r_act, r_opp, r_reset, r_buf_pick, r_buf_idx = jax.random.split(rng, 6)

        obs0 = encode(state, 0, episode_steps)
        obs1 = encode(state, 1, episode_steps)

        sampled = model.apply(
            params, obs0, r_act, state.planet_ships,
            state.planet_x, state.planet_y, state.home_planet_idx[0],
        )
        action_0 = _sampled_to_multi(sampled)
        opp_sampled = model.apply(
            fparams, obs1, r_opp, state.planet_ships,
            state.planet_x, state.planet_y, state.home_planet_idx[1],
        )
        action_1 = _sampled_to_multi(opp_sampled)

        next_state, out = env.step(state, (action_0, action_1))

        # Auto-reset logic: when done, choose between random map or buffer state.
        fresh_random = env.reset(r_reset)
        fresh_buffer = _sample_buffer_state(r_buf_idx)
        use_buffer = jax.random.uniform(r_buf_pick) < jnp.float32(reset_prob)
        fresh = jax.tree_util.tree_map(
            lambda a, b: jnp.where(use_buffer, a, b),
            fresh_buffer, fresh_random,
        )
        chosen = jax.tree_util.tree_map(
            lambda a, b: jnp.where(out.done, a, b),
            fresh, next_state,
        )

        per_step = _per_step_dict_from_sample(obs0, state, sampled, out)
        return (chosen, rng, params, fparams), per_step

    def _scan_one_env(state, rng, params, fparams):
        (final_state, rng_out, _, _), traj = jax.lax.scan(
            _one_env_step, (state, rng, params, fparams), xs=None, length=rollout_length
        )
        final_obs0 = encode(final_state, 0, episode_steps)
        sampled_final = model.apply(
            params, final_obs0,
            jax.random.fold_in(rng_out, 1),
            final_state.planet_ships,
            final_state.planet_x, final_state.planet_y,
            final_state.home_planet_idx[0],
        )
        return final_state, traj, sampled_final.value

    def rollout_fn(
        params,
        frozen_params,
        states: EnvState,
        rngs: jnp.ndarray,
    ) -> tuple[EnvState, jnp.ndarray, Rollout]:
        final_states, trajs, last_values = jax.vmap(
            _scan_one_env, in_axes=(0, 0, None, None)
        )(states, rngs, params, frozen_params)
        traj_swapped = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), trajs)
        rollout = _rollout_from_traj(traj_swapped, last_values)
        out_rngs = jax.vmap(lambda r: jax.random.fold_in(r, rollout_length + 13))(rngs)
        return final_states, out_rngs, rollout

    return jax.jit(rollout_fn)


def make_rollout_fn_with_frozen_opp(
    env: OrbitWarsEnv,
    model: ActorCritic,
    rollout_length: int,
    num_envs: int,
    episode_steps: int = constants.DEFAULT_EPISODE_STEPS,
):
    """Self-play rollout where the opponent uses a frozen ActorCritic snapshot.

    Both learner and opponent emit the full multi-fleet action; the learner's
    per-step logp/entropy/value are stored in the rollout for PPO. The frozen
    opponent samples stochastically (``deterministic=False``).
    """

    def _one_env_step(carry, _):
        state, rng, params, fparams = carry
        rng, r_act, r_opp, r_reset = jax.random.split(rng, 4)

        obs0 = encode(state, 0, episode_steps)
        obs1 = encode(state, 1, episode_steps)

        sampled = model.apply(
            params, obs0, r_act, state.planet_ships,
            state.planet_x, state.planet_y, state.home_planet_idx[0],
        )
        action_0 = _sampled_to_multi(sampled)
        opp_sampled = model.apply(
            fparams, obs1, r_opp, state.planet_ships,
            state.planet_x, state.planet_y, state.home_planet_idx[1],
        )
        action_1 = _sampled_to_multi(opp_sampled)

        next_state, out = env.step_and_autoreset(state, (action_0, action_1), r_reset)
        per_step = _per_step_dict_from_sample(obs0, state, sampled, out)
        new_carry = (next_state, rng, params, fparams)
        return new_carry, per_step

    def _scan_one_env(state, rng, params, fparams):
        (final_state, rng_out, _, _), traj = jax.lax.scan(
            _one_env_step, (state, rng, params, fparams), xs=None, length=rollout_length
        )
        final_obs0 = encode(final_state, 0, episode_steps)
        sampled_final = model.apply(
            params, final_obs0,
            jax.random.fold_in(rng_out, 1),
            final_state.planet_ships,
            final_state.planet_x, final_state.planet_y,
            final_state.home_planet_idx[0],
        )
        return final_state, traj, sampled_final.value

    def rollout_fn(
        params,
        frozen_params,
        states: EnvState,
        rngs: jnp.ndarray,
    ) -> tuple[EnvState, jnp.ndarray, Rollout]:
        final_states, trajs, last_values = jax.vmap(
            _scan_one_env, in_axes=(0, 0, None, None)
        )(states, rngs, params, frozen_params)
        traj_swapped = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), trajs)
        rollout = _rollout_from_traj(traj_swapped, last_values)
        out_rngs = jax.vmap(lambda r: jax.random.fold_in(r, rollout_length + 7))(rngs)
        return final_states, out_rngs, rollout

    return jax.jit(rollout_fn)
