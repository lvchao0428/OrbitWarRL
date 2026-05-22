"""Eval harness: play a batch of episodes vs a baseline; report mean reward + winrate."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbit_wars_rl.env import OrbitWarsEnv, constants
from orbit_wars_rl.env.actions import MultiPlayerAction, PlayerAction, single_to_multi
from orbit_wars_rl.features import encode
from orbit_wars_rl.net.model import ActorCritic
from orbit_wars_rl.ppo.rollout import random_opponent_action


def _sampled_to_multi(sampled) -> MultiPlayerAction:
    return MultiPlayerAction(
        src_idx=sampled.src_idx,
        dst_idx=sampled.dst_idx,
        pct_bin=sampled.pct_bin,
        emit_mask=sampled.emit_mask,
    )


def play_vs_random(
    model: ActorCritic,
    params,
    rng: jnp.ndarray,
    num_envs: int = 16,
    num_groups: int = 5,
    max_episode_steps: int = 200,
) -> dict:
    """Run ``num_envs`` parallel episodes vs the random opponent.

    Returns:
      {win_rate, mean_reward, episodes}
    """
    env = OrbitWarsEnv(num_groups=num_groups, episode_steps=max_episode_steps)
    rng_init, rng_loop = jax.random.split(rng)
    init_rngs = jax.random.split(rng_init, num_envs)
    states = jax.vmap(env.reset)(init_rngs)

    def step_one(carry, t):
        states, rng, rewards_acc, dones_seen = carry
        rng, r_agent, r_opp, r_reset = jax.random.split(rng, 4)
        r_agents = jax.random.split(r_agent, num_envs)
        r_opps = jax.random.split(r_opp, num_envs)
        r_resets = jax.random.split(r_reset, num_envs)

        def _per_env(state, r_a, r_o, r_r):
            obs0 = encode(state, 0, max_episode_steps)
            obs1 = encode(state, 1, max_episode_steps)
            sa = model.apply(params, obs0, r_a, state.planet_ships, deterministic=True)
            a0 = _sampled_to_multi(sa)
            a1 = single_to_multi(random_opponent_action(r_o, obs1))
            new_state, out = env.step_and_autoreset(state, (a0, a1), r_r)
            return new_state, out.reward, out.done

        new_states, rewards_t, dones_t = jax.vmap(_per_env)(states, r_agents, r_opps, r_resets)

        not_seen = jnp.logical_not(dones_seen)
        new_rewards_acc = rewards_acc + jnp.where(not_seen, rewards_t, jnp.float32(0.0))
        new_dones_seen = dones_seen | dones_t
        return (new_states, rng, new_rewards_acc, new_dones_seen), None

    init_carry = (
        states,
        rng_loop,
        jnp.zeros((num_envs,), dtype=jnp.float32),
        jnp.zeros((num_envs,), dtype=jnp.bool_),
    )
    (final_states, _, rewards_acc, dones_seen), _ = jax.lax.scan(
        step_one, init_carry, jnp.arange(max_episode_steps)
    )

    finished = dones_seen
    rewards = jnp.where(finished, rewards_acc, jnp.float32(0.0))
    n_done = jnp.maximum(finished.sum(), jnp.float32(1.0))
    win_rate = ((rewards > 0).astype(jnp.float32) * finished.astype(jnp.float32)).sum() / n_done
    mean_reward = rewards.sum() / n_done

    return dict(
        win_rate=float(win_rate),
        mean_reward=float(mean_reward),
        episodes_completed=int(finished.sum()),
        episodes_total=int(num_envs),
    )


def play_vs_frozen(
    model: ActorCritic,
    params,
    frozen_params,
    rng: jnp.ndarray,
    num_envs: int = 16,
    num_groups: int = 5,
    max_episode_steps: int = 200,
) -> dict:
    """Same as ``play_vs_random`` but the opponent is a frozen learner."""
    env = OrbitWarsEnv(num_groups=num_groups, episode_steps=max_episode_steps)
    rng_init, rng_loop = jax.random.split(rng)
    init_rngs = jax.random.split(rng_init, num_envs)
    states = jax.vmap(env.reset)(init_rngs)

    def step_one(carry, t):
        states, rng, rewards_acc, dones_seen = carry
        rng, r_a, r_o, r_r = jax.random.split(rng, 4)
        r_as = jax.random.split(r_a, num_envs)
        r_os = jax.random.split(r_o, num_envs)
        r_rs = jax.random.split(r_r, num_envs)

        def _per_env(state, r_agent, r_opp, r_reset):
            obs0 = encode(state, 0, max_episode_steps)
            obs1 = encode(state, 1, max_episode_steps)
            sa0 = model.apply(params, obs0, r_agent, state.planet_ships, deterministic=True)
            sa1 = model.apply(frozen_params, obs1, r_opp, state.planet_ships, deterministic=False)
            a0 = _sampled_to_multi(sa0)
            a1 = _sampled_to_multi(sa1)
            new_state, out = env.step_and_autoreset(state, (a0, a1), r_reset)
            return new_state, out.reward, out.done

        new_states, rewards_t, dones_t = jax.vmap(_per_env)(states, r_as, r_os, r_rs)
        not_seen = jnp.logical_not(dones_seen)
        new_rewards_acc = rewards_acc + jnp.where(not_seen, rewards_t, jnp.float32(0.0))
        new_dones_seen = dones_seen | dones_t
        return (new_states, rng, new_rewards_acc, new_dones_seen), None

    init_carry = (
        states,
        rng_loop,
        jnp.zeros((num_envs,), dtype=jnp.float32),
        jnp.zeros((num_envs,), dtype=jnp.bool_),
    )
    (_, _, rewards_acc, dones_seen), _ = jax.lax.scan(
        step_one, init_carry, jnp.arange(max_episode_steps)
    )
    finished = dones_seen
    rewards = jnp.where(finished, rewards_acc, jnp.float32(0.0))
    n_done = jnp.maximum(finished.sum(), jnp.float32(1.0))
    win_rate = ((rewards > 0).astype(jnp.float32) * finished.astype(jnp.float32)).sum() / n_done
    mean_reward = rewards.sum() / n_done
    return dict(
        win_rate=float(win_rate),
        mean_reward=float(mean_reward),
        episodes_completed=int(finished.sum()),
        episodes_total=int(num_envs),
    )
