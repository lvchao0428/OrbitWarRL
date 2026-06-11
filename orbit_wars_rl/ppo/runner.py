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

import json
import os
import pickle
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

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
from orbit_wars_rl.ppo.rollout_symmetric import make_rollout_fn_symmetric
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
    pool_seed_paths: list[str] = field(default_factory=list)
    snapshot_current: bool = True     # if false, only externally gated seeds are sampled
    # Plan B: v20 state-buffer curriculum
    buffer_path: str = ""             # path to .npz from collect_states.py
    buffer_reset_ratio: float = 0.30  # fraction of resets that use a buffer state
    buffer_rollout_ratio: float = 1.0 # fraction of post-warmup updates spent in buffer rollout


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
    emit_hard_stop: bool = False
    emit_hard_stop_min_step: int = 1
    flip_hard_mask: bool = False
    # Train/replay alignment (Day11): rolling metrics for strn+frzn only.
    align_roll_window: int = 20
    # Inline mini replay vs v20 on CPU at eval_every (ground-truth gate).
    eval_vs_v20: bool = False
    eval_vs_v20_num_games: int = 3

    # Symmetric self-play (Frog Parade style): same params for both players.
    # When True, ignores all SelfPlayConfig settings (no frozen/strong/buffer).
    # Uses random opponent during warmup, then pure symmetric self-play.
    symmetric_selfplay: bool = False
    symmetric_warmup: int = 100
    # Zero-sum value head: value head sees both players' obs during training.
    zero_sum_value: bool = False
    # v13: allow model to hold (not emit) at step 0; min pct bin floor.
    allow_hold: bool = False
    force_emit_worth_it: bool = False
    min_pct_bin: int = 0
    # Resume from a checkpoint (path to .pkl file).
    resume_ckpt: str = ""

    ppo: PPOConfig = field(default_factory=PPOConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)


# Opponents whose rollout stats correlate better with replay vs v20.
_ALIGN_OPPS = frozenset({"strn", "frzn"})
_CKPT_PREFER_OPPS = frozenset({"strn", "frzn", "rand", "symm"})


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


def _export_meta(cfg: TrainConfig) -> dict[str, Any]:
    return {
        "allow_hold": bool(getattr(cfg, "allow_hold", False)),
        "force_emit_worth_it": bool(getattr(cfg, "force_emit_worth_it", False)),
        "min_pct_bin": int(getattr(cfg, "min_pct_bin", 0)),
        "emit_hard_stop": bool(cfg.emit_hard_stop),
        "emit_hard_stop_min_step": int(cfg.emit_hard_stop_min_step),
        "flip_hard_mask": bool(cfg.flip_hard_mask),
    }


def save_checkpoint(
    path: str,
    params,
    opt_state,
    step: int,
    meta: Optional[dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = dict(
        params=jax.tree_util.tree_map(lambda x: jnp.asarray(x), params),
        opt_state=jax.tree_util.tree_map(
            lambda x: jnp.asarray(x) if hasattr(x, "shape") else x, opt_state
        ),
        step=step,
        meta=meta or {},
    )
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    if meta:
        meta_path = path.replace(".pkl", ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def load_checkpoint(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _adapt_strong_params(target_params, strong_params):
    """Zero-pad strong ckpt weights to match current model when only feature dims differ.

    Handles the case where a strong opponent was trained with fewer feature
    dimensions (e.g. f29: planet=28, global=17, emit_pair=4) and needs to
    run inside a model with more features (e.g. f38: planet=33, global=18,
    emit_pair=6).  New feature dimensions are padded with zeros so they have
    no effect on the opponent's inference.

    Returns adapted params or None if adaptation is not possible.
    """
    import numpy as np
    from orbit_wars_rl.inference.weights import flatten_params

    target_flat = flatten_params(target_params)
    strong_flat = flatten_params(strong_params)

    if set(target_flat.keys()) != set(strong_flat.keys()):
        return None

    adapted = {}
    mismatches = []
    for key in target_flat:
        t_shape = target_flat[key].shape
        s_shape = strong_flat[key].shape
        if t_shape == s_shape:
            adapted[key] = strong_flat[key]
        elif len(t_shape) == len(s_shape):
            # Pad each axis that is smaller in strong than target
            can_pad = all(s <= t for s, t in zip(s_shape, t_shape))
            if not can_pad:
                return None
            pad_widths = [(0, t - s) for s, t in zip(s_shape, t_shape)]
            adapted[key] = np.pad(
                np.asarray(strong_flat[key]),
                pad_widths,
                mode="constant",
                constant_values=0,
            )
            mismatches.append(f"  {key}: {s_shape} -> {t_shape}")
        else:
            return None

    if not mismatches:
        return None

    print("[shape-adapter] padded tensors:", flush=True)
    for m in mismatches:
        print(m, flush=True)

    # Unflatten back into the nested dict structure matching target_params
    def _unflatten(flat_dict):
        root = {}
        for path, arr in flat_dict.items():
            parts = path.split("/")
            node = root
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = jnp.asarray(arr)
        return root

    unflat = _unflatten(adapted)
    # Re-wrap with outer "params" key if target_params has it
    if isinstance(target_params, dict) and set(target_params.keys()) == {"params"}:
        unflat = {"params": unflat}
    return unflat


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
        f"ANTI_HOARD={env_rewards.SHAPING_ANTI_HOARD}/thresh{env_rewards.SHAPING_ANTI_HOARD_THRESH}; "
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
            f"buffer_reset_ratio={cfg.selfplay.buffer_reset_ratio} "
            f"buffer_rollout_ratio={cfg.selfplay.buffer_rollout_ratio}",
            flush=True,
        )
    if cfg.selfplay.enabled and cfg.selfplay.pool_seed_paths:
        print(
            "[gated-pool] seed_paths="
            + ", ".join(cfg.selfplay.pool_seed_paths)
            + f" snapshot_current={cfg.selfplay.snapshot_current}",
            flush=True,
        )

    env = OrbitWarsEnv(num_groups=cfg.num_groups, episode_steps=cfg.episode_steps)
    model = ActorCritic(
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ff_dim=cfg.ff_dim,
        max_fleets_per_turn=cfg.max_fleets_per_turn,
        emit_hard_stop=cfg.emit_hard_stop,
        emit_hard_stop_min_step=cfg.emit_hard_stop_min_step,
        flip_hard_mask=cfg.flip_hard_mask,
        zero_sum_value=cfg.zero_sum_value,
        allow_hold=getattr(cfg, "allow_hold", False),
        force_emit_worth_it=getattr(cfg, "force_emit_worth_it", False),
        min_pct_bin=getattr(cfg, "min_pct_bin", 0),
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
        dummy_state.planet_x,
        dummy_state.planet_y,
        dummy_state.home_planet_idx[0],
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

    symmetric_rollout_fn = None
    if cfg.symmetric_selfplay and constants.NUM_PLAYERS == 2:
        print(
            f"[symmetric] Frog Parade style: same params drive BOTH players, "
            f"no frozen/strong/buffer opponents, "
            f"zero_sum_value={cfg.zero_sum_value}",
            flush=True,
        )
        symmetric_rollout_fn = make_rollout_fn_symmetric(
            env, model,
            rollout_length=cfg.rollout_length,
            num_envs=cfg.num_envs,
            episode_steps=cfg.episode_steps,
        )

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
        float(cfg.selfplay.buffer_rollout_ratio)
        if cfg.selfplay.enabled and cfg.selfplay.buffer_path
        else 0.0
    )

    align_window = max(1, int(cfg.align_roll_window))
    align_roll: dict[str, deque] = {
        k: deque(maxlen=align_window)
        for k in ("spf", "z0", "garr", "e2")
    }

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
        except Exception:  # noqa: BLE001
            strong_params = _adapt_strong_params(params, strong_ckpt["params"])
            if strong_params is not None:
                print(
                    f"[curriculum] shape-adapted strong opponent from "
                    f"{cfg.selfplay.strong_ckpt_path}",
                    flush=True,
                )
            else:
                print(
                    f"[curriculum] WARN strong ckpt shape mismatch and "
                    f"adaptation failed, disabling: {cfg.selfplay.strong_ckpt_path}",
                    flush=True,
                )
                sp_strong = 0.0

    if cfg.selfplay.enabled and pool is not None and cfg.selfplay.pool_seed_paths:
        for seed_path in cfg.selfplay.pool_seed_paths:
            seed_ckpt = load_checkpoint(seed_path)
            try:
                chex.assert_trees_all_equal_shapes(params, seed_ckpt["params"])
                seed_params = seed_ckpt["params"]
                status = "loaded"
            except Exception:  # noqa: BLE001
                seed_params = _adapt_strong_params(params, seed_ckpt["params"])
                status = "shape-adapted" if seed_params is not None else "disabled"
            if seed_params is None:
                print(f"[gated-pool] WARN cannot load seed {seed_path}", flush=True)
                continue
            pool.snapshot(seed_params)
            print(f"[gated-pool] {status} seed opponent {seed_path}", flush=True)

    for update in range(cfg.num_updates):
        rng_iter, r_step, r_opp_pick, r_pool_sample = jax.random.split(rng_iter, 4)

        opp_tag = "rand"

        if cfg.symmetric_selfplay and symmetric_rollout_fn is not None:
            states, env_rngs, rollout = symmetric_rollout_fn(
                params, states, env_rngs,
            )
            opp_tag = "symm"
        elif cfg.selfplay.enabled and update >= sp_warmup:
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
            elif (
                buffer_rollout_fn is not None
                and sp_buffer > 0.0
                and roll < sp_strong + sp_frozen + sp_buffer
            ):
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
        metrics_py["opp_tag"] = opp_tag
        metrics_py["opp_frozen"] = 1.0 if opp_tag == "frzn" else 0.0
        metrics_py["opp_strong"] = 1.0 if opp_tag == "strn" else 0.0
        metrics_py["opp_buffer"] = 1.0 if opp_tag == "buf" else 0.0
        if cfg.selfplay.enabled and pool is not None:
            metrics_py["pool_size"] = float(len(pool))

        spf_step = metrics_py.get("mean_ships_per_fleet", 0.0)
        z0_step = metrics_py.get("zero_emit_rate", 0.0)
        garr_step = metrics_py.get("mean_garrison_my", 0.0)
        e2_step = metrics_py.get("emit2_rate", 0.0)
        if opp_tag in _ALIGN_OPPS:
            align_roll["spf"].append(spf_step)
            align_roll["z0"].append(z0_step)
            align_roll["garr"].append(garr_step)
            align_roll["e2"].append(e2_step)
        for key, roll in align_roll.items():
            if roll:
                metrics_py[f"align/{key}"] = sum(roll) / len(roll)
        metrics_py["align/n"] = float(len(align_roll["spf"]))

        if (
            cfg.selfplay.enabled
            and cfg.selfplay.snapshot_current
            and (update + 1) % cfg.selfplay.snapshot_every == 0
        ):
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

            if cfg.eval_vs_v20:
                eval_ckpt = os.path.join(cfg.ckpt_dir, f"_eval_u{update:06d}.pkl")
                eval_meta = {"update": update, "export": _export_meta(cfg)}
                save_checkpoint(eval_ckpt, params, opt_state, update, meta=eval_meta)
                try:
                    from orbit_wars_rl.eval.v20_mini_gate import run_v20_mini_gate

                    tag = f"train_u{update}"
                    gate = run_v20_mini_gate(
                        eval_ckpt,
                        tag,
                        project_root=os.getcwd(),
                        num_games=cfg.eval_vs_v20_num_games,
                        python=sys.executable,
                    )
                    metrics_py["eval_vs_v20/spf"] = gate["spf"]
                    metrics_py["eval_vs_v20/garr"] = gate["garr"]
                    metrics_py["eval_vs_v20/flip_pct"] = gate["flip_pct"]
                    metrics_py["eval_vs_v20/z0_pct"] = gate["z0_pct"]
                    metrics_py["eval_vs_v20/e2_plus_pct"] = gate["e2_plus_pct"]
                    metrics_py["eval_vs_v20/wins"] = float(gate["wins"])
                    print(
                        f"[eval_vs_v20] u{update} {gate['summary_line']}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[eval_vs_v20] WARN failed: {exc}", flush=True)
                finally:
                    if os.path.isfile(eval_ckpt):
                        os.remove(eval_ckpt)
                    meta_sidecar = eval_ckpt.replace(".pkl", ".meta.json")
                    if os.path.isfile(meta_sidecar):
                        os.remove(meta_sidecar)

        if update % cfg.log_every == 0:
            wr_rand = metrics_py.get("eval/win_rate")
            wr_frzn = metrics_py.get("eval_vs_frozen/win_rate")
            wr_str = ""
            if wr_rand is not None:
                wr_str += f" WRr {wr_rand:.2f}"
            if wr_frzn is not None:
                wr_str += f" WRf {wr_frzn:.2f}"
            v20_spf = metrics_py.get("eval_vs_v20/spf")
            if v20_spf is not None:
                wr_str += (
                    f" v20[spf {v20_spf:.1f} flip {metrics_py.get('eval_vs_v20/flip_pct', 0):.1f}% "
                    f"e2+ {metrics_py.get('eval_vs_v20/e2_plus_pct', 0):.1f}%]"
                )
            align_str = ""
            if align_roll["spf"]:
                align_str = (
                    f" align[spf {metrics_py.get('align/spf', 0):.1f} "
                    f"e2 {metrics_py.get('align/e2', 0):.2f} "
                    f"z0 {metrics_py.get('align/z0', 0):.2f} n={int(metrics_py.get('align/n', 0))}]"
                )
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
                + align_str
                + wr_str
            )

        logger.log(metrics_py, update)
        history.append(metrics_py)

        if cfg.ckpt_every > 0 and (update + 1) % cfg.ckpt_every == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"ckpt_{update:06d}.pkl")
            ckpt_meta = {
                "update": update,
                "opp_tag": opp_tag,
                "export": _export_meta(cfg),
                "spf": spf_step,
                "z0": z0_step,
                "garr": garr_step,
                "emit2_rate": e2_step,
                "align_spf": metrics_py.get("align/spf"),
                "align_e2": metrics_py.get("align/e2"),
                "align_z0": metrics_py.get("align/z0"),
                "eval_vs_v20_spf": metrics_py.get("eval_vs_v20/spf"),
                "eval_vs_v20_e2_plus_pct": metrics_py.get("eval_vs_v20/e2_plus_pct"),
                "eval_vs_v20_flip_pct": metrics_py.get("eval_vs_v20/flip_pct"),
            }
            save_checkpoint(ckpt_path, params, opt_state, update, meta=ckpt_meta)
            warn = ""
            if opp_tag not in _CKPT_PREFER_OPPS:
                warn = (
                    f" WARN opp={opp_tag} (prefer strn/frzn for replay-aligned ckpt; "
                    f"see .meta.json)"
                )
            print(f"[ckpt] saved {ckpt_path} opp={opp_tag}{warn}", flush=True)

    logger.close()
    return dict(history=history, final_params=params, final_opt_state=opt_state)
