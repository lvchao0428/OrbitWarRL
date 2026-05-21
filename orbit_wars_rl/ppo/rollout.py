"""Rollout collection for PPO. Single-jit over scan + vmap over envs.

Player 0 is the learning agent; player 1 is an opponent policy supplied as a
pure function ``opponent_action_fn(rng, obs) -> PlayerAction``. For MVP we
ship a "random valid src/dst" opponent; later we will swap in a frozen
snapshot of the learner.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import chex
import jax
import jax.numpy as jnp

from orbit_wars_rl.env import OrbitWarsEnv, constants
from orbit_wars_rl.env.actions import PlayerAction
from orbit_wars_rl.env.state import EnvState
from orbit_wars_rl.features import EncodedObs, encode
from orbit_wars_rl.net.model import ActorCritic


@chex.dataclass(frozen=True)
class Rollout:
    """[T, B, ...] tensors collected per rollout. B = num_envs."""

    obs_planet_feats: chex.Array
    obs_planet_mask: chex.Array
    obs_fleet_feats: chex.Array
    obs_fleet_mask: chex.Array
    obs_global_feats: chex.Array
    obs_my_planet_mask: chex.Array

    src_idx: chex.Array
    dst_idx: chex.Array
    pct_bin: chex.Array
    src_logp: chex.Array
    dst_logp: chex.Array
    pct_logp: chex.Array
    value: chex.Array

    reward: chex.Array
    done: chex.Array

    last_value: chex.Array


OpponentFn = Callable[[jnp.ndarray, EncodedObs], PlayerAction]


def random_opponent_action(rng: jnp.ndarray, obs: EncodedObs) -> PlayerAction:
    """Pick a random owned source and random other planet as target.

    Falls back to slot 0 if no owned planets. ``pct_bin`` always picks bin 1
    (50%). Single-env (no leading batch dim).
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

        sampled = model.apply(params, obs0, r_act)
        action_0 = PlayerAction(
            src_idx=sampled.src_idx,
            dst_idx=sampled.dst_idx,
            pct_bin=sampled.pct_bin,
        )
        action_1 = opponent_fn(r_opp, obs1)

        next_state, out = env.step_and_autoreset(state, (action_0, action_1), r_reset)

        per_step = dict(
            obs_planet_feats=obs0.planet_feats,
            obs_planet_mask=obs0.planet_mask,
            obs_fleet_feats=obs0.fleet_feats,
            obs_fleet_mask=obs0.fleet_mask,
            obs_global_feats=obs0.global_feats,
            obs_my_planet_mask=obs0.my_planet_mask,
            src_idx=sampled.src_idx,
            dst_idx=sampled.dst_idx,
            pct_bin=sampled.pct_bin,
            src_logp=sampled.src_logp,
            dst_logp=sampled.dst_logp,
            pct_logp=sampled.pct_logp,
            value=sampled.value,
            reward=out.reward,
            done=out.done,
        )
        new_carry = (next_state, rng, params)
        return new_carry, per_step

    def _scan_one_env(state: EnvState, rng: jnp.ndarray, params) -> tuple[EnvState, dict, jnp.ndarray]:
        (final_state, rng_out, _), traj = jax.lax.scan(
            _one_env_step, (state, rng, params), xs=None, length=rollout_length
        )
        final_obs0 = encode(final_state, 0, episode_steps)
        sampled_final = model.apply(params, final_obs0, jax.random.fold_in(rng_out, 1))
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
        last_values_T = last_values

        rollout = Rollout(
            obs_planet_feats=traj_swapped["obs_planet_feats"],
            obs_planet_mask=traj_swapped["obs_planet_mask"],
            obs_fleet_feats=traj_swapped["obs_fleet_feats"],
            obs_fleet_mask=traj_swapped["obs_fleet_mask"],
            obs_global_feats=traj_swapped["obs_global_feats"],
            obs_my_planet_mask=traj_swapped["obs_my_planet_mask"],
            src_idx=traj_swapped["src_idx"],
            dst_idx=traj_swapped["dst_idx"],
            pct_bin=traj_swapped["pct_bin"],
            src_logp=traj_swapped["src_logp"],
            dst_logp=traj_swapped["dst_logp"],
            pct_logp=traj_swapped["pct_logp"],
            value=traj_swapped["value"],
            reward=traj_swapped["reward"],
            done=traj_swapped["done"],
            last_value=last_values_T,
        )

        out_rngs = jax.vmap(lambda r: jax.random.fold_in(r, rollout_length + 7))(rngs)
        return final_states, out_rngs, rollout

    return jax.jit(rollout_fn)
