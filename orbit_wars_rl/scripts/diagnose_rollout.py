"""One-off rollout diagnostic: collect a single rollout with random init params
and print stats about reward/advantage/value/emit distributions.

Helps debug "PPO seems healthy but policy isn't learning" situations by
showing whether the learning signal (advantages) has meaningful magnitude.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from orbit_wars_rl.env import OrbitWarsEnv
from orbit_wars_rl.features import encode
from orbit_wars_rl.net.model import ActorCritic
from orbit_wars_rl.ppo.rollout import make_rollout_fn
from orbit_wars_rl.ppo.update import compute_gae


def describe(name: str, arr: jnp.ndarray) -> None:
    a = np.asarray(arr)
    print(
        f"{name:>30s}  shape={a.shape}  "
        f"mean={a.mean():+.4f}  std={a.std():.4f}  "
        f"min={a.min():+.4f}  max={a.max():+.4f}  "
        f"nonzero={float((a != 0).mean()):.3f}"
    )


def main():
    rng = jax.random.PRNGKey(0)
    rng_init, rng_envs, rng_roll = jax.random.split(rng, 3)

    num_envs = 16
    rollout_length = 256
    episode_steps = 200
    num_groups = 5

    env = OrbitWarsEnv(num_groups=num_groups, episode_steps=episode_steps)
    model = ActorCritic(d_model=64, n_layers=2, n_heads=4, ff_dim=128, max_fleets_per_turn=8)

    env_rngs = jax.random.split(rng_envs, num_envs)
    states = jax.vmap(env.reset)(env_rngs)
    dummy_state = jax.tree_util.tree_map(lambda x: x[0], states)
    dummy_obs = encode(dummy_state, 0, episode_steps)
    params = model.init(
        rng_init, dummy_obs, jax.random.PRNGKey(0), dummy_state.planet_ships,
        dummy_state.planet_x, dummy_state.planet_y, dummy_state.home_planet_idx[0],
    )

    rollout_fn = make_rollout_fn(env, model, rollout_length=rollout_length, num_envs=num_envs, episode_steps=episode_steps)
    states, env_rngs, rollout = rollout_fn(params, states, env_rngs)

    print("=== Raw rollout fields ===")
    describe("reward [T,B]", rollout.reward)
    describe("done [T,B]", rollout.done.astype(jnp.float32))
    describe("value [T,B]", rollout.value)
    describe("last_value [B]", rollout.last_value)
    describe("emit_mask [T,B,K]", rollout.emit_mask.astype(jnp.float32))
    describe("emit_free [T,B,K]", rollout.emit_free_mask.astype(jnp.float32))
    print()

    print("=== Episode statistics ===")
    n_episodes = float(rollout.done.sum())
    if n_episodes > 0:
        winners = jnp.where(rollout.done, rollout.reward, jnp.float32(0.0))
        win_count = float((winners > 0).sum())
        loss_count = float((winners < 0).sum())
        draw_count = float(((winners == 0) & rollout.done).sum())
        print(f"  Episodes finished: {int(n_episodes)}")
        print(f"  Wins: {win_count}  Losses: {loss_count}  Draws: {draw_count}")
        print(f"  Win rate (within rollout): {win_count/n_episodes:.3f}")
    else:
        print(f"  No episodes finished in this rollout. (rollout_length={rollout_length}, episode_steps={episode_steps})")
        print(f"  Steps per env: {rollout_length}, episodes per env: ~{rollout_length/episode_steps:.2f}")
    print()

    print("=== GAE advantages / returns ===")
    advs, rets = compute_gae(
        rollout.reward, rollout.value, rollout.done, rollout.last_value,
        gamma=0.997, gae_lambda=0.95,
    )
    describe("advantages [T,B]", advs)
    describe("returns [T,B]", rets)
    adv_norm_std = float(advs.std())
    print(f"  advantages.std() = {adv_norm_std:.6f}  (if << 0.1, learning signal is weak)")
    print()

    print("=== Per-step emit distribution ===")
    emits_per_turn = rollout.emit_mask.astype(jnp.int32).sum(axis=-1)
    describe("emits_per_turn [T,B]", emits_per_turn.astype(jnp.float32))
    pct_with_zero = float((emits_per_turn == 0).mean())
    pct_with_one = float((emits_per_turn == 1).mean())
    pct_with_8 = float((emits_per_turn == 8).mean())
    print(f"  P(0 emits per turn) = {pct_with_zero:.3f}")
    print(f"  P(1 emit per turn)  = {pct_with_one:.3f}")
    print(f"  P(8 emits per turn) = {pct_with_8:.3f}")


if __name__ == "__main__":
    main()
