"""Train loop: roll out, update, log, ckpt. Pure-Python wrapper around jit'd pieces.

Self-play (optional, see ``SelfPlayConfig``):

* For the first ``warmup_updates`` updates the opponent is always ``random``,
  so the policy can climb out of the trivial basin against an easy enemy.
* After warmup, each update flips a coin (``frozen_ratio``) deciding whether
  the whole batch rolls vs ``random`` or vs a snapshot sampled from the pool.
  The pool is refreshed every ``snapshot_every`` updates with the current
  ``params``; the latest snapshot is used as the bootstrap if the pool was
  still empty.
* The same flax model serves both the learner and the opponent; only the
  parameter pytree differs, and ``make_rollout_fn_with_frozen_opp`` ensures
  both share the same jit cache regardless of which snapshot is chosen.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Optional

import chex
import jax
import jax.numpy as jnp

from orbit_wars_rl.env import OrbitWarsEnv, constants
from orbit_wars_rl.env import rewards as env_rewards
from orbit_wars_rl.env.state import EnvState
from orbit_wars_rl.features import encode
from orbit_wars_rl.net.model import ActorCritic
from orbit_wars_rl.env import constants
from orbit_wars_rl.ppo.rollout import (
    make_rollout_fn,
    make_rollout_fn_with_buffer_reset,
    make_rollout_fn_with_frozen_opp,
)
from orbit_wars_rl.ppo.rollout_4p import make_rollout_fn_4p, make_rollout_fn_with_frozen_opp_4p
from orbit_wars_rl.ppo.update import PPOConfig, make_optimizer, make_train_step
from orbit_wars_rl.selfplay.eval import play_vs_random, play_vs_frozen
from orbit_wars_rl.selfplay.pool import FrozenAgentPool


@dataclass
class SelfPlayConfig:
    """Settings for mixing frozen-opponent rollouts into training.

    Defaults are all-off so existing runs are unchanged unless explicitly enabled.

    Buffer-reset curriculum (Plan B):
      buffer_path        : path to .npz produced by collect_states.py;
                           empty string = disabled.
      buffer_reset_ratio : probability in [0, 1] that a done-reset uses a
                           buffer state instead of a random map.  Only active
                           when buffer_path is set and selfplay.enabled=True.
    """
    enabled: bool = False
    warmup_updates: int = 30          # updates spent vs random before any self-play
    snapshot_every: int = 10          # how often to push current params into the pool
    pool_capacity: int = 5            # FIFO ring buffer size
    frozen_ratio: float = 0.5         # fraction of post-warmup updates spent vs frozen
    eval_vs_frozen: bool = True       # additionally compute WR vs latest snapshot
    strong_ckpt_path: str = ""        # C1: fixed strong opponent snapshot (same arch)
    strong_ratio: float = 0.0         # fraction vs strong_ckpt (post-warmup)
    # Plan B: v20 state-buffer curriculum
    buffer_path: str = ""             # path to .npz from collect_states.py
    buffer_reset_ratio: float = 0.30  # fraction of resets that use a buffer state


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
    max_fleets_per_turn: int = constants.MAX_FLEETS_PER_TURN

    ppo: PPOConfig = field(default_factory=PPOConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)


def load_state_buffer(path: str) -> "EnvState":
    """Load a state buffer .npz (from collect_states.py) into a batched EnvState.

    Returns a batched ``EnvState`` with a leading axis of size N (buffer size).
    All arrays are moved to JAX device arrays.
    """
    import numpy as np
    data = np.load(path)
    return EnvState(
        planet_owner=jnp.asarray(data["planet_owner"]),
        planet_x=jnp.asarray(data["planet_x"]),
        planet_y=jnp.asarray(data["planet_y"]),
        planet_radius=jnp.asarray(data["planet_radius"]),
        planet_ships=jnp.asarray(data["planet_ships"]),
        planet_prod=jnp.asarray(data["planet_prod"]),
        planet_mask=jnp.asarray(data["planet_mask"]),
        fleet_owner=jnp.asarray(data["fleet_owner"]),
        fleet_x=jnp.asarray(data["fleet_x"]),
        fleet_y=jnp.asarray(data["fleet_y"]),
        fleet_angle=jnp.asarray(data["fleet_angle"]),
        fleet_ships=jnp.asarray(data["fleet_ships"]),
        fleet_mask=jnp.asarray(data["fleet_mask"]),
        step=jnp.asarray(data["step"]),
        done=jnp.zeros(data["step"].shape, dtype=jnp.bool_),
        rng=jnp.zeros((*data["step"].shape, 2), dtype=jnp.uint32),
        angular_velocity=jnp.asarray(data["angular_velocity"]),
        planet_orbit_radius=jnp.asarray(data["planet_orbit_radius"]),
        planet_orbit_phase=jnp.asarray(data["planet_orbit_phase"]),
        planet_is_orbiting=jnp.asarray(data["planet_is_orbiting"]),
        home_planet_idx=jnp.asarray(data["home_planet_idx"]),
    )


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


def train(
    cfg: TrainConfig,
    log_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> dict:
    """Train PPO. If ``resume_from`` is a path to a pickle written by
    ``save_checkpoint``, overwrite the freshly-initialised params (and
    ``opt_state`` when the optimizer pytree shapes match) with the loaded
    snapshot. Useful for warm-starting an extension run on a new YAML config
    (e.g. v4.2 ckpt -> v4.3 run with refreshed lr schedule and entropy)."""
    rng = jax.random.PRNGKey(cfg.seed)
    rng_init, rng_envs, rng_train = jax.random.split(rng, 3)

    # Reward/term banner: surfaces what would otherwise be invisible config
    # (env vars, defaults) directly into the train log. Anything that affects
    # the reward signal must show up here.
    kaggle_ep = 500
    print(
        "[reward] kaggle-aligned: terminal +1 win/+1 tie>0/-1 loss/-1 double-wipeout; "
        f"is_terminal at step >= episode_steps-2; SHAPING_SCALE={env_rewards.SHAPING_SCALE}; "
        f"v1: KEEP_HOME={env_rewards.SHAPING_KEEP_HOME} "
        f"FLEET_SIZE={env_rewards.SHAPING_FLEET_SIZE}/norm{env_rewards.SHAPING_FLEET_NORM}; "
        f"v2 (expert-replay calibrated): PROD_SHARE={env_rewards.SHAPING_PROD_SHARE} "
        f"PLANET_SHARE={env_rewards.SHAPING_PLANET_SHARE} "
        f"FLEET_LOG={env_rewards.SHAPING_FLEET_LOG}/ref{env_rewards.SHAPING_FLEET_LOG_REF}/"
        f"floor{env_rewards.SHAPING_FLEET_LOG_FLOOR}; "
        f"v3 (top10-2630ep calibrated): "
        f"PROD_SHARE_DELTA={env_rewards.SHAPING_PROD_SHARE_DELTA} "
        f"EMIT_LOG={env_rewards.SHAPING_EMIT_LOG} "
        f"EMIT_GATED={env_rewards.SHAPING_EMIT_GATED} "
        f"RELEASE={env_rewards.SHAPING_RELEASE}/K{env_rewards.SHAPING_RELEASE_K} "
        f"CAPTURE={env_rewards.SHAPING_CAPTURE}; "
        f"episode_steps={cfg.episode_steps} (kaggle={kaggle_ep}, mismatch={cfg.episode_steps != kaggle_ep})",
        flush=True,
    )
    if cfg.selfplay.enabled and cfg.selfplay.strong_ckpt_path:
        print(
            f"[curriculum] strong_ratio={cfg.selfplay.strong_ratio} "
            f"frozen_ratio={cfg.selfplay.frozen_ratio} "
            f"strong_ckpt={cfg.selfplay.strong_ckpt_path}",
            flush=True,
        )
    if cfg.selfplay.enabled and cfg.selfplay.buffer_path:
        print(
            f"[buffer-curriculum] buffer_path={cfg.selfplay.buffer_path} "
            f"buffer_reset_ratio={cfg.selfplay.buffer_reset_ratio}",
            flush=True,
        )

    env = OrbitWarsEnv(num_groups=cfg.num_groups, episode_steps=cfg.episode_steps)
    model = ActorCritic(
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ff_dim=cfg.ff_dim,
        max_fleets_per_turn=cfg.max_fleets_per_turn,
    )

    env_rngs = jax.random.split(rng_envs, cfg.num_envs)
    states = jax.vmap(env.reset)(env_rngs)
    dummy_state = jax.tree_util.tree_map(lambda x: x[0], states)
    dummy_obs = encode(dummy_state, 0, cfg.episode_steps)
    params = model.init(
        rng_init,
        dummy_obs,
        jax.random.PRNGKey(0),
        dummy_state.planet_ships,
    )
    optimizer = make_optimizer(cfg.ppo)
    opt_state = optimizer.init(params)

    if resume_from is not None:
        ckpt = load_checkpoint(resume_from)
        try:
            chex.assert_trees_all_equal_shapes(params, ckpt["params"])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"resume params shape mismatch with {resume_from}: {exc}. "
                "Refusing to silently drop layers."
            ) from exc
        params = ckpt["params"]
        print(f"[resume] loaded params from {resume_from} (step={ckpt.get('step', '?')})")
        # opt_state stays freshly initialised: lr / momentum schedules are
        # re-baked from the new PPOConfig (which is the whole point of an
        # extension run with a tweaked schedule).

    if constants.NUM_PLAYERS == 4:
        rollout_fn = make_rollout_fn_4p(
            env, model,
            rollout_length=cfg.rollout_length,
            num_envs=cfg.num_envs,
            episode_steps=cfg.episode_steps,
        )
        selfplay_rollout_fn = None
        if cfg.selfplay.enabled:
            selfplay_rollout_fn = make_rollout_fn_with_frozen_opp_4p(
                env, model,
                rollout_length=cfg.rollout_length,
                num_envs=cfg.num_envs,
                episode_steps=cfg.episode_steps,
            )
    else:
        rollout_fn = make_rollout_fn(
            env, model,
            rollout_length=cfg.rollout_length,
            num_envs=cfg.num_envs,
            episode_steps=cfg.episode_steps,
        )
        selfplay_rollout_fn = None
        buffer_rollout_fn = None
        if cfg.selfplay.enabled:
            selfplay_rollout_fn = make_rollout_fn_with_frozen_opp(
                env, model,
                rollout_length=cfg.rollout_length,
                num_envs=cfg.num_envs,
                episode_steps=cfg.episode_steps,
            )
            if cfg.selfplay.buffer_path:
                buf_states = load_state_buffer(cfg.selfplay.buffer_path)
                n_buf = jax.tree_util.tree_leaves(buf_states)[0].shape[0]
                print(
                    f"[buffer-curriculum] loaded {n_buf} states from "
                    f"{cfg.selfplay.buffer_path}",
                    flush=True,
                )
                buffer_rollout_fn = make_rollout_fn_with_buffer_reset(
                    env, model,
                    rollout_length=cfg.rollout_length,
                    num_envs=cfg.num_envs,
                    buffer_states=buf_states,
                    reset_prob=cfg.selfplay.buffer_reset_ratio,
                    episode_steps=cfg.episode_steps,
                )

    pool: Optional[FrozenAgentPool] = None
    if cfg.selfplay.enabled:
        pool = FrozenAgentPool(capacity=cfg.selfplay.pool_capacity)

    train_step = make_train_step(model, cfg.ppo, optimizer)

    logger = Logger(log_dir=log_dir)
    history: list[dict] = []

    rng_iter = rng_train
    total_env_steps = 0
    t_start = time.time()
    sp_warmup = cfg.selfplay.warmup_updates if cfg.selfplay.enabled else cfg.num_updates + 1
    sp_frozen = float(cfg.selfplay.frozen_ratio) if cfg.selfplay.enabled else 0.0
    sp_strong = float(cfg.selfplay.strong_ratio) if cfg.selfplay.enabled else 0.0
    sp_buffer = (
        float(cfg.selfplay.buffer_reset_ratio)
        if cfg.selfplay.enabled and cfg.selfplay.buffer_path
        else 0.0
    )

    strong_params = None
    if cfg.selfplay.enabled and cfg.selfplay.strong_ckpt_path:
        strong_ckpt = load_checkpoint(cfg.selfplay.strong_ckpt_path)
        try:
            chex.assert_trees_all_equal_shapes(params, strong_ckpt["params"])
            strong_params = strong_ckpt["params"]
            print(
                f"[curriculum] loaded strong opponent from {cfg.selfplay.strong_ckpt_path}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[curriculum] WARN strong ckpt shape mismatch, disabling: {exc}",
                flush=True,
            )
            sp_strong = 0.0

    for update in range(cfg.num_updates):
        rng_iter, r_step, r_opp_pick, r_pool_sample = jax.random.split(rng_iter, 4)

        opp_tag = "rand"
        if cfg.selfplay.enabled and update >= sp_warmup:
            roll = float(jax.random.uniform(r_opp_pick, ()))
            if (
                strong_params is not None
                and sp_strong > 0.0
                and roll < sp_strong
            ):
                states, env_rngs, rollout = selfplay_rollout_fn(
                    params, strong_params, states, env_rngs,
                )
                opp_tag = "strn"
            elif (
                pool is not None
                and len(pool) > 0
                and roll < sp_strong + sp_frozen
            ):
                frozen_params = pool.sample(r_pool_sample)
                states, env_rngs, rollout = selfplay_rollout_fn(
                    params, frozen_params, states, env_rngs,
                )
                opp_tag = "frzn"
            elif buffer_rollout_fn is not None and sp_buffer > 0.0:
                # Buffer-reset: use frozen params = current params (no frozen opp).
                # The buffer state mixing happens inside the rollout fn itself.
                states, env_rngs, rollout = buffer_rollout_fn(
                    params, params, states, env_rngs,
                )
                opp_tag = "buf"
            else:
                states, env_rngs, rollout = rollout_fn(params, states, env_rngs)
        else:
            states, env_rngs, rollout = rollout_fn(params, states, env_rngs)

        params, opt_state, metrics = train_step(params, opt_state, rollout, r_step)
        params = jax.tree_util.tree_map(jnp.asarray, params)
        metrics_py = {k: float(v) for k, v in metrics.items()}
        total_env_steps += cfg.num_envs * cfg.rollout_length
        elapsed = time.time() - t_start
        metrics_py["sps"] = total_env_steps / max(elapsed, 1e-6)
        metrics_py["update"] = update
        metrics_py["total_env_steps"] = total_env_steps
        metrics_py["opp_frozen"] = 1.0 if opp_tag == "frzn" else 0.0
        metrics_py["opp_strong"] = 1.0 if opp_tag == "strn" else 0.0
        metrics_py["opp_buffer"] = 1.0 if opp_tag == "buf" else 0.0
        if cfg.selfplay.enabled and pool is not None:
            metrics_py["pool_size"] = float(len(pool))

        if cfg.selfplay.enabled and (update + 1) % cfg.selfplay.snapshot_every == 0:
            pool.snapshot(params)  # type: ignore[union-attr]

        if cfg.eval_every > 0 and (update + 1) % cfg.eval_every == 0:
            eval_metrics = play_vs_random(
                model, params, jax.random.PRNGKey(1000 + update),
                num_envs=cfg.eval_num_envs,
                num_groups=cfg.num_groups,
                max_episode_steps=cfg.episode_steps,
            )
            for k, v in eval_metrics.items():
                metrics_py[f"eval/{k}"] = v
            if (cfg.selfplay.enabled and cfg.selfplay.eval_vs_frozen
                and pool is not None and len(pool) > 0):
                frozen_eval = play_vs_frozen(
                    model, params, pool.latest(),
                    jax.random.PRNGKey(2000 + update),
                    num_envs=cfg.eval_num_envs,
                    num_groups=cfg.num_groups,
                    max_episode_steps=cfg.episode_steps,
                )
                for k, v in frozen_eval.items():
                    metrics_py[f"eval_vs_frozen/{k}"] = v

        if update % cfg.log_every == 0:
            wr_rand = metrics_py.get("eval/win_rate")
            wr_frzn = metrics_py.get("eval_vs_frozen/win_rate")
            wr_str = ""
            if wr_rand is not None:
                wr_str += f" WRr {wr_rand:.2f}"
            if wr_frzn is not None:
                wr_str += f" WRf {wr_frzn:.2f}"
            emits = metrics_py.get("mean_emits_per_turn", 0.0)
            ent_emit = metrics_py.get("ent_emit", 0.0)
            adv_std = metrics_py.get("adv_std", 0.0)
            term_r = metrics_py.get("mean_terminal_reward", 0.0)
            ev = metrics_py.get("explained_variance", 0.0)
            spf = metrics_py.get("mean_ships_per_fleet", 0.0)
            z0 = metrics_py.get("zero_emit_rate", 0.0)
            garr = metrics_py.get("mean_garrison_my", 0.0)
            t_garr = metrics_py.get("total_garrison_my", 0.0)
            emit2 = metrics_py.get("emit2_rate", 0.0)
            n_fleets = metrics_py.get("fleets_in_flight", 0.0)
            pshare = metrics_py.get("prod_share", 0.0)
            ptshare = metrics_py.get("planet_share", 0.0)
            flog = metrics_py.get("fleet_log_score", 0.0)
            pdelta = metrics_py.get("prod_share_delta", 0.0)
            pk_ratio = metrics_py.get("peak_over_mean_garr", 0.0)
            print(
                f"upd {update:4d}  steps {total_env_steps:7d}  sps {metrics_py['sps']:.0f}  "
                f"opp {opp_tag}  loss {metrics_py['loss']:+.3f}  "
                f"pg {metrics_py['pg_loss']:+.4f}  v {metrics_py['v_loss']:.3f}  ev {ev:+.2f}  "
                f"adv_std {adv_std:.3f}  tR {term_r:+.2f}  "
                f"ent[s/d/p/e] {metrics_py['ent_src']:.2f}/{metrics_py['ent_dst']:.2f}/{metrics_py['ent_pct']:.2f}/{ent_emit:.2f}  "
                f"emits {emits:.2f}  spf {spf:.1f}  z0 {z0:.2f}  garr {garr:.1f}  "
                f"tG {t_garr:.0f}  e2 {emit2:.2f}  nF {n_fleets:.1f}  "
                f"pS {pshare:.2f}  ptS {ptshare:.2f}  fLog {flog:.2f}  "
                f"pdΔ {pdelta:+.4f}  pkR {pk_ratio:.2f}  "
                f"clip {metrics_py['clip_frac']:.2f}  kl {metrics_py['approx_kl']:+.3f}"
                + wr_str
            )

        logger.log(metrics_py, update)
        history.append(metrics_py)

        if cfg.ckpt_every > 0 and (update + 1) % cfg.ckpt_every == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"ckpt_{update:06d}.pkl")
            save_checkpoint(ckpt_path, params, opt_state, update)

    logger.close()
    return dict(history=history, final_params=params, final_opt_state=opt_state)
