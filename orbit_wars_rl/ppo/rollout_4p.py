"""4-player rollout helpers (imported by rollout.py when ORBITWARS_NUM_PLAYERS=4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbit_wars_rl.env import OrbitWarsEnv, constants
from orbit_wars_rl.env.actions import single_to_multi
from orbit_wars_rl.env.state import EnvState
from orbit_wars_rl.features import encode
from orbit_wars_rl.net.model import ActorCritic
from orbit_wars_rl.ppo.rollout import (
    OpponentFn,
    Rollout,
    _per_step_dict_from_sample,
    _rollout_from_traj,
    _sampled_to_multi,
    random_opponent_action,
)


def make_rollout_fn_4p(
    env: OrbitWarsEnv,
    model: ActorCritic,
    rollout_length: int,
    num_envs: int,
    opponent_fn: OpponentFn = random_opponent_action,
    episode_steps: int = constants.DEFAULT_EPISODE_STEPS,
):
    """4-player rollout: player 0 learns; players 1-3 use ``opponent_fn``."""

    def _one_env_step(carry, _):
        state, rng, params = carry
        rng, r_act, r1, r2, r3, r_reset = jax.random.split(rng, 6)

        obs0 = encode(state, 0, episode_steps)
        sampled = model.apply(
            params, obs0, r_act, state.planet_ships,
            state.planet_x, state.planet_y, state.home_planet_idx[0],
        )
        action_0 = _sampled_to_multi(sampled)
        action_1 = single_to_multi(opponent_fn(r1, encode(state, 1, episode_steps)))
        action_2 = single_to_multi(opponent_fn(r2, encode(state, 2, episode_steps)))
        action_3 = single_to_multi(opponent_fn(r3, encode(state, 3, episode_steps)))

        next_state, out = env.step_and_autoreset(
            state, (action_0, action_1, action_2, action_3), r_reset
        )
        per_step = _per_step_dict_from_sample(obs0, state, sampled, out)
        return (next_state, rng, params), per_step

    def _scan_one_env(state: EnvState, rng: jnp.ndarray, params) -> tuple[EnvState, dict, jnp.ndarray]:
        (final_state, rng_out, _), traj = jax.lax.scan(
            _one_env_step, (state, rng, params), xs=None, length=rollout_length
        )
        final_obs0 = encode(final_state, 0, episode_steps)
        sampled_final = model.apply(
            params, final_obs0, jax.random.fold_in(rng_out, 1),
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


def make_rollout_fn_with_frozen_opp_4p(
    env: OrbitWarsEnv,
    model: ActorCritic,
    rollout_length: int,
    num_envs: int,
    episode_steps: int = constants.DEFAULT_EPISODE_STEPS,
):
    """4P self-play: player 0 learns; players 1-3 share one frozen ActorCritic."""

    def _one_env_step(carry, _):
        state, rng, params, fparams = carry
        rng, r_act, r1, r2, r3, r_reset = jax.random.split(rng, 6)

        obs0 = encode(state, 0, episode_steps)
        sampled = model.apply(
            params, obs0, r_act, state.planet_ships,
            state.planet_x, state.planet_y, state.home_planet_idx[0],
        )
        action_0 = _sampled_to_multi(sampled)
        action_1 = _sampled_to_multi(
            model.apply(
                fparams, encode(state, 1, episode_steps), r1, state.planet_ships,
                state.planet_x, state.planet_y, state.home_planet_idx[1],
            )
        )
        action_2 = _sampled_to_multi(
            model.apply(
                fparams, encode(state, 2, episode_steps), r2, state.planet_ships,
                state.planet_x, state.planet_y, state.home_planet_idx[2],
            )
        )
        action_3 = _sampled_to_multi(
            model.apply(
                fparams, encode(state, 3, episode_steps), r3, state.planet_ships,
                state.planet_x, state.planet_y, state.home_planet_idx[3],
            )
        )

        next_state, out = env.step_and_autoreset(
            state, (action_0, action_1, action_2, action_3), r_reset
        )
        per_step = _per_step_dict_from_sample(obs0, state, sampled, out)
        return (next_state, rng, params, fparams), per_step

    def _scan_one_env(state, rng, params, fparams):
        (final_state, rng_out, _, _), traj = jax.lax.scan(
            _one_env_step, (state, rng, params, fparams), xs=None, length=rollout_length
        )
        final_obs0 = encode(final_state, 0, episode_steps)
        sampled_final = model.apply(
            params, final_obs0, jax.random.fold_in(rng_out, 1),
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
