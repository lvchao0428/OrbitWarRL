"""Symmetric self-play rollout: one model controls BOTH players.

This is the simplest and most effective self-play strategy, as proven by
Frog Parade (Lux AI S3 #2): "I used the most basic form where the same
model plays for both players at once for N steps across many games
simultaneously before pausing to update the weights."

Key differences from `rollout.py`:
* Both players use the same `params` -- no frozen opponent, no buffer reset.
* Rewards are collected from player 0's perspective only (the environment
  is symmetric, so player 1's learning signal is implicit via parameter
  sharing).
* Opponent observations (player 1) are stored alongside player 0's for
  the zero-sum value head (the value head "cheats" by seeing both sides
  during training, which stabilizes value estimation).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbit_wars_rl.env import OrbitWarsEnv, constants
from orbit_wars_rl.env.state import EnvState
from orbit_wars_rl.features import encode
from orbit_wars_rl.net.model import ActorCritic
from orbit_wars_rl.ppo.rollout import (
    Rollout,
    _rollout_from_traj,
    _sampled_to_multi,
)


def _per_step_dict_symmetric(obs0, obs1, state, sampled, out):
    """Build per-step dict including opponent obs for zero-sum value head."""
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
        # Opponent obs for zero-sum value head.
        opp_planet_feats=obs1.planet_feats,
        opp_planet_mask=obs1.planet_mask,
        opp_fleet_feats=obs1.fleet_feats,
        opp_fleet_mask=obs1.fleet_mask,
        opp_global_feats=obs1.global_feats,
        opp_my_planet_mask=obs1.my_planet_mask,
    )


def _rollout_from_traj_symmetric(
    traj_swapped: dict, last_values: jnp.ndarray
) -> Rollout:
    """Build Rollout including opponent obs fields."""
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
        opp_planet_feats=traj_swapped["opp_planet_feats"],
        opp_planet_mask=traj_swapped["opp_planet_mask"],
        opp_fleet_feats=traj_swapped["opp_fleet_feats"],
        opp_fleet_mask=traj_swapped["opp_fleet_mask"],
        opp_global_feats=traj_swapped["opp_global_feats"],
        opp_my_planet_mask=traj_swapped["opp_my_planet_mask"],
    )


def make_rollout_fn_symmetric(
    env: OrbitWarsEnv,
    model: ActorCritic,
    rollout_length: int,
    num_envs: int,
    episode_steps: int = constants.DEFAULT_EPISODE_STEPS,
):
    """Symmetric self-play: same params drive both players.

    When the model has ``zero_sum_value=True``, opponent observations are
    passed through to the value head so it can see both perspectives.
    """
    use_zero_sum = model.zero_sum_value

    def _one_env_step(carry, _):
        state, rng, params = carry
        rng, r_act0, r_act1, r_reset = jax.random.split(rng, 4)

        obs0 = encode(state, 0, episode_steps)
        obs1 = encode(state, 1, episode_steps)

        sampled0 = model.apply(
            params, obs0, r_act0, state.planet_ships,
            state.planet_x, state.planet_y, state.home_planet_idx[0],
            opp_obs=obs1 if use_zero_sum else None,
        )
        action_0 = _sampled_to_multi(sampled0)

        sampled1 = model.apply(
            params, obs1, r_act1, state.planet_ships,
            state.planet_x, state.planet_y, state.home_planet_idx[1],
            opp_obs=obs0 if use_zero_sum else None,
        )
        action_1 = _sampled_to_multi(sampled1)

        next_state, out = env.step_and_autoreset(
            state, (action_0, action_1), r_reset
        )

        per_step = _per_step_dict_symmetric(obs0, obs1, state, sampled0, out)
        return (next_state, rng, params), per_step

    def _scan_one_env(state, rng, params):
        (final_state, rng_out, _), traj = jax.lax.scan(
            _one_env_step, (state, rng, params), xs=None, length=rollout_length
        )
        final_obs0 = encode(final_state, 0, episode_steps)
        final_obs1 = encode(final_state, 1, episode_steps) if use_zero_sum else None
        sampled_final = model.apply(
            params, final_obs0,
            jax.random.fold_in(rng_out, 1),
            final_state.planet_ships,
            final_state.planet_x, final_state.planet_y,
            final_state.home_planet_idx[0],
            opp_obs=final_obs1,
        )
        return final_state, traj, sampled_final.value

    def rollout_fn(
        params,
        states: EnvState,
        rngs: jnp.ndarray,
    ) -> tuple[EnvState, jnp.ndarray, Rollout]:
        final_states, trajs, last_values = jax.vmap(
            _scan_one_env, in_axes=(0, 0, None)
        )(states, rngs, params)
        traj_swapped = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), trajs
        )
        rollout = _rollout_from_traj_symmetric(traj_swapped, last_values)
        out_rngs = jax.vmap(
            lambda r: jax.random.fold_in(r, rollout_length + 7)
        )(rngs)
        return final_states, out_rngs, rollout

    return jax.jit(rollout_fn)
