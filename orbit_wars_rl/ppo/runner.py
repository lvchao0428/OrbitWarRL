"""Train loop: roll out, update, log, ckpt. Pure-Python wrapper around jit'd pieces."""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp

from orbit_wars_rl.env import OrbitWarsEnv, constants
from orbit_wars_rl.features import encode
from orbit_wars_rl.net.model import ActorCritic
from orbit_wars_rl.ppo.rollout import make_rollout_fn
from orbit_wars_rl.ppo.update import PPOConfig, make_optimizer, make_train_step
from orbit_wars_rl.selfplay.eval import play_vs_random


@dataclass
class TrainConfig:
    num_envs: int = 16
    rollout_length: int = 128
    num_updates: int = 200
    num_groups: int = 5
    episode_steps: int = 200
    eval_every: int = 10
    eval_num_envs: int = 32
    ckpt_dir: str = "./ckpt_mvp"
    ckpt_every: int = 50
    log_every: int = 1
    seed: int = 42
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    ff_dim: int = 128

    ppo: PPOConfig = field(default_factory=PPOConfig)


def save_checkpoint(path: str, params, opt_state, step: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = dict(
        params=jax.tree_util.tree_map(lambda x: jnp.asarray(x), params),
        opt_state=jax.tree_util.tree_map(lambda x: jnp.asarray(x) if hasattr(x, "shape") else x, opt_state),
        step=step,
    )
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


class Logger:
    """Tiny stdout + optional tensorboard logger."""

    def __init__(self, log_dir: Optional[str] = None) -> None:
        self.tb = None
        if log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(log_dir, exist_ok=True)
                self.tb = SummaryWriter(log_dir=log_dir)
            except Exception as e:
                print(f"[logger] tensorboard unavailable ({e}), stdout only")
                self.tb = None

    def log(self, metrics: dict, step: int) -> None:
        if self.tb is not None:
            for k, v in metrics.items():
                try:
                    self.tb.add_scalar(k, float(v), step)
                except Exception:
                    pass

    def close(self) -> None:
        if self.tb is not None:
            try:
                self.tb.close()
            except Exception:
                pass


def train(cfg: TrainConfig, log_dir: Optional[str] = None) -> dict:
    rng = jax.random.PRNGKey(cfg.seed)
    rng_init, rng_envs, rng_train = jax.random.split(rng, 3)

    env = OrbitWarsEnv(num_groups=cfg.num_groups, episode_steps=cfg.episode_steps)
    model = ActorCritic(
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ff_dim=cfg.ff_dim,
    )

    env_rngs = jax.random.split(rng_envs, cfg.num_envs)
    states = jax.vmap(env.reset)(env_rngs)
    dummy_obs = encode(jax.tree_util.tree_map(lambda x: x[0], states), 0, cfg.episode_steps)
    params = model.init(rng_init, dummy_obs, jax.random.PRNGKey(0))
    optimizer = make_optimizer(cfg.ppo)
    opt_state = optimizer.init(params)

    rollout_fn = make_rollout_fn(
        env, model,
        rollout_length=cfg.rollout_length,
        num_envs=cfg.num_envs,
        episode_steps=cfg.episode_steps,
    )
    train_step = make_train_step(model, cfg.ppo, optimizer)

    logger = Logger(log_dir=log_dir)
    history: list[dict] = []

    rng_iter = rng_train
    total_env_steps = 0
    t_start = time.time()

    for update in range(cfg.num_updates):
        rng_iter, r_step = jax.random.split(rng_iter)

        states, env_rngs, rollout = rollout_fn(params, states, env_rngs)
        params, opt_state, metrics = train_step(params, opt_state, rollout, r_step)
        params = jax.tree_util.tree_map(jnp.asarray, params)
        metrics_py = {k: float(v) for k, v in metrics.items()}
        total_env_steps += cfg.num_envs * cfg.rollout_length
        elapsed = time.time() - t_start
        metrics_py["sps"] = total_env_steps / max(elapsed, 1e-6)
        metrics_py["update"] = update
        metrics_py["total_env_steps"] = total_env_steps

        if cfg.eval_every > 0 and (update + 1) % cfg.eval_every == 0:
            eval_metrics = play_vs_random(
                model, params, jax.random.PRNGKey(1000 + update),
                num_envs=cfg.eval_num_envs,
                num_groups=cfg.num_groups,
                max_episode_steps=cfg.episode_steps,
            )
            for k, v in eval_metrics.items():
                metrics_py[f"eval/{k}"] = v

        if update % cfg.log_every == 0:
            print(
                f"upd {update:4d}  steps {total_env_steps:7d}  sps {metrics_py['sps']:.0f}  "
                f"loss {metrics_py['loss']:+.3f}  pg {metrics_py['pg_loss']:+.3f}  v {metrics_py['v_loss']:.3f}  "
                f"ent[s/d/p] {metrics_py['ent_src']:.2f}/{metrics_py['ent_dst']:.2f}/{metrics_py['ent_pct']:.2f}  "
                f"clip {metrics_py['clip_frac']:.2f}  kl {metrics_py['approx_kl']:+.3f}  "
                + (f"WR {metrics_py.get('eval/win_rate', float('nan')):.2f}"
                   if 'eval/win_rate' in metrics_py else "")
            )

        logger.log(metrics_py, update)
        history.append(metrics_py)

        if cfg.ckpt_every > 0 and (update + 1) % cfg.ckpt_every == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"ckpt_{update:06d}.pkl")
            save_checkpoint(ckpt_path, params, opt_state, update)

    logger.close()
    return dict(history=history, final_params=params, final_opt_state=opt_state)
