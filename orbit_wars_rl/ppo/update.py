"""PPO loss + optax warmup-cosine schedule.

Three independent entropy coefficients (src / dst / pct), value clipping,
advantage normalization, gradient clipping.
"""

from __future__ import annotations

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax

from orbit_wars_rl.features import EncodedObs
from orbit_wars_rl.net.model import ActorCritic
from orbit_wars_rl.ppo.rollout import Rollout


class PPOConfig(NamedTuple):
    lr_peak: float = 3e-4
    lr_warmup_steps: int = 200
    lr_decay_steps: int = 5000
    lr_floor: float = 1e-5
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    value_coef: float = 0.5
    ent_coef_src: float = 0.005
    ent_coef_dst: float = 0.01
    ent_coef_pct: float = 0.02
    max_grad_norm: float = 0.5
    gamma: float = 0.997
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4


def make_optimizer(cfg: PPOConfig) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr_peak,
        warmup_steps=cfg.lr_warmup_steps,
        decay_steps=max(cfg.lr_decay_steps, cfg.lr_warmup_steps + 1),
        end_value=cfg.lr_floor,
    )
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=schedule),
    )


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_values: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Inputs are [T, B] arrays (and ``last_values`` [B,]). Returns adv [T,B], ret [T,B]."""
    T = rewards.shape[0]
    next_values = jnp.concatenate([values[1:], last_values[None, :]], axis=0)
    not_done = 1.0 - dones.astype(jnp.float32)
    gamma_f = jnp.float32(gamma)
    deltas = rewards + gamma_f * next_values * not_done - values

    def scan_body(adv_next, idx):
        adv_t = deltas[idx] + gamma_f * gae_lambda * not_done[idx] * adv_next
        return adv_t, adv_t

    _, advs_rev = jax.lax.scan(scan_body, jnp.zeros_like(rewards[0]), jnp.arange(T - 1, -1, -1))
    advs = advs_rev[::-1]
    returns = advs + values
    return advs, returns


def _stack_obs_from_rollout(r: Rollout) -> EncodedObs:
    """Flatten rollout [T,B,...] into a stack along leading axis for the encoder."""
    return EncodedObs(
        planet_feats=r.obs_planet_feats,
        planet_mask=r.obs_planet_mask,
        fleet_feats=r.obs_fleet_feats,
        fleet_mask=r.obs_fleet_mask,
        global_feats=r.obs_global_feats,
        my_planet_mask=r.obs_my_planet_mask,
        enemy_planet_mask=jnp.zeros_like(r.obs_my_planet_mask),
        neutral_planet_mask=jnp.zeros_like(r.obs_my_planet_mask),
    )


def ppo_loss(
    params,
    model: ActorCritic,
    rollout: Rollout,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    cfg: PPOConfig,
) -> tuple[jnp.ndarray, dict]:
    """Single-minibatch PPO loss. Rollout fields are [N, ...] (flat over T*B)."""
    obs = _stack_obs_from_rollout(rollout)
    out = model.apply(params, obs, rollout.src_idx, rollout.dst_idx, method=ActorCritic.evaluate)

    src_logp_new, src_ent = _logp_entropy(out.src_logits, rollout.src_idx)
    dst_logp_new, dst_ent = _logp_entropy(out.dst_logits, rollout.dst_idx)
    pct_logp_new, pct_ent = _logp_entropy(out.pct_logits, rollout.pct_bin)

    old_logp = rollout.src_logp + rollout.dst_logp + rollout.pct_logp
    new_logp = src_logp_new + dst_logp_new + pct_logp_new
    ratio = jnp.exp(new_logp - old_logp)

    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    adv_norm = (advantages - adv_mean) / adv_std

    pg_loss_1 = -adv_norm * ratio
    pg_loss_2 = -adv_norm * jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
    pg_loss = jnp.maximum(pg_loss_1, pg_loss_2).mean()

    value_pred = out.value
    value_old = rollout.value
    value_clipped = value_old + jnp.clip(value_pred - value_old, -cfg.value_clip_eps, cfg.value_clip_eps)
    v_loss_1 = (value_pred - returns) ** 2
    v_loss_2 = (value_clipped - returns) ** 2
    v_loss = 0.5 * jnp.maximum(v_loss_1, v_loss_2).mean()

    entropy_loss = -(
        cfg.ent_coef_src * src_ent.mean()
        + cfg.ent_coef_dst * dst_ent.mean()
        + cfg.ent_coef_pct * pct_ent.mean()
    )

    total_loss = pg_loss + cfg.value_coef * v_loss + entropy_loss
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > cfg.clip_eps).astype(jnp.float32))
    approx_kl = jnp.mean(old_logp - new_logp)

    metrics = dict(
        pg_loss=pg_loss,
        v_loss=v_loss,
        ent_src=src_ent.mean(),
        ent_dst=dst_ent.mean(),
        ent_pct=pct_ent.mean(),
        clip_frac=clip_frac,
        approx_kl=approx_kl,
        ratio_mean=ratio.mean(),
        value_mean=value_pred.mean(),
        return_mean=returns.mean(),
        adv_mean=adv_mean,
    )
    return total_loss, metrics


def _logp_entropy(logits: jnp.ndarray, idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(idx, logits.shape[-1], dtype=logits.dtype)
    logp = (log_probs * one_hot).sum(axis=-1)
    probs = jax.nn.softmax(logits, axis=-1)
    entropy = -(probs * jnp.where(probs > 0, jnp.log(probs + 1e-20), 0.0)).sum(axis=-1)
    return logp, entropy


def _flatten_rollout(rollout: Rollout, advantages: jnp.ndarray, returns: jnp.ndarray):
    """[T, B, ...] -> [T*B, ...] for all leaves."""
    def f(x):
        return x.reshape((-1,) + x.shape[2:])

    flat_rollout = Rollout(
        obs_planet_feats=f(rollout.obs_planet_feats),
        obs_planet_mask=f(rollout.obs_planet_mask),
        obs_fleet_feats=f(rollout.obs_fleet_feats),
        obs_fleet_mask=f(rollout.obs_fleet_mask),
        obs_global_feats=f(rollout.obs_global_feats),
        obs_my_planet_mask=f(rollout.obs_my_planet_mask),
        src_idx=f(rollout.src_idx),
        dst_idx=f(rollout.dst_idx),
        pct_bin=f(rollout.pct_bin),
        src_logp=f(rollout.src_logp),
        dst_logp=f(rollout.dst_logp),
        pct_logp=f(rollout.pct_logp),
        value=f(rollout.value),
        reward=f(rollout.reward),
        done=f(rollout.done),
        last_value=rollout.last_value,
    )
    return flat_rollout, f(advantages), f(returns)


def _gather_minibatch(rollout: Rollout, advs: jnp.ndarray, rets: jnp.ndarray, idx: jnp.ndarray):
    take = lambda x: x[idx]
    sub = Rollout(
        obs_planet_feats=take(rollout.obs_planet_feats),
        obs_planet_mask=take(rollout.obs_planet_mask),
        obs_fleet_feats=take(rollout.obs_fleet_feats),
        obs_fleet_mask=take(rollout.obs_fleet_mask),
        obs_global_feats=take(rollout.obs_global_feats),
        obs_my_planet_mask=take(rollout.obs_my_planet_mask),
        src_idx=take(rollout.src_idx),
        dst_idx=take(rollout.dst_idx),
        pct_bin=take(rollout.pct_bin),
        src_logp=take(rollout.src_logp),
        dst_logp=take(rollout.dst_logp),
        pct_logp=take(rollout.pct_logp),
        value=take(rollout.value),
        reward=take(rollout.reward),
        done=take(rollout.done),
        last_value=rollout.last_value,
    )
    return sub, take(advs), take(rets)


def make_train_step(model: ActorCritic, cfg: PPOConfig, optimizer: optax.GradientTransformation):
    """Returns ``train_step(params, opt_state, rollout, rng) -> (params, opt_state, metrics)``."""

    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    def train_step(params, opt_state, rollout: Rollout, rng: jnp.ndarray):
        T = rollout.reward.shape[0]
        B = rollout.reward.shape[1]
        advs, rets = compute_gae(
            rollout.reward, rollout.value, rollout.done, rollout.last_value,
            cfg.gamma, cfg.gae_lambda,
        )
        flat_rollout, advs_f, rets_f = _flatten_rollout(rollout, advs, rets)
        n_samples = T * B
        minibatch_size = max(1, n_samples // cfg.num_minibatches)

        def epoch_step(carry, epoch_rng):
            params_e, opt_state_e, accum = carry
            perm = jax.random.permutation(epoch_rng, n_samples)

            def mb_step(carry_mb, mb_idx):
                params_m, opt_state_m, accum_m = carry_mb
                start = mb_idx * minibatch_size
                idx = jax.lax.dynamic_slice_in_dim(perm, start, minibatch_size)
                sub_rollout, sub_advs, sub_rets = _gather_minibatch(flat_rollout, advs_f, rets_f, idx)
                (loss_val, metrics), grads = grad_fn(params_m, model, sub_rollout, sub_advs, sub_rets, cfg)
                updates, opt_state_new = optimizer.update(grads, opt_state_m, params_m)
                params_new = optax.apply_updates(params_m, updates)
                metrics = dict(metrics, loss=loss_val,
                               grad_norm=optax.global_norm(grads))
                summed = jax.tree_util.tree_map(lambda a, b: a + b, accum_m, metrics)
                return (params_new, opt_state_new, summed), None

            zero_metrics = dict(
                pg_loss=jnp.float32(0.0),
                v_loss=jnp.float32(0.0),
                ent_src=jnp.float32(0.0),
                ent_dst=jnp.float32(0.0),
                ent_pct=jnp.float32(0.0),
                clip_frac=jnp.float32(0.0),
                approx_kl=jnp.float32(0.0),
                ratio_mean=jnp.float32(0.0),
                value_mean=jnp.float32(0.0),
                return_mean=jnp.float32(0.0),
                adv_mean=jnp.float32(0.0),
                loss=jnp.float32(0.0),
                grad_norm=jnp.float32(0.0),
            )

            (params_new, opt_state_new, summed), _ = jax.lax.scan(
                mb_step,
                (params_e, opt_state_e, zero_metrics),
                jnp.arange(cfg.num_minibatches),
            )
            summed_combined = jax.tree_util.tree_map(lambda a, b: a + b, accum, summed)
            return (params_new, opt_state_new, summed_combined), None

        rng_epochs = jax.random.split(rng, cfg.update_epochs)
        zero_metrics_outer = dict(
            pg_loss=jnp.float32(0.0),
            v_loss=jnp.float32(0.0),
            ent_src=jnp.float32(0.0),
            ent_dst=jnp.float32(0.0),
            ent_pct=jnp.float32(0.0),
            clip_frac=jnp.float32(0.0),
            approx_kl=jnp.float32(0.0),
            ratio_mean=jnp.float32(0.0),
            value_mean=jnp.float32(0.0),
            return_mean=jnp.float32(0.0),
            adv_mean=jnp.float32(0.0),
            loss=jnp.float32(0.0),
            grad_norm=jnp.float32(0.0),
        )
        (params_out, opt_state_out, accumulated), _ = jax.lax.scan(
            epoch_step, (params, opt_state, zero_metrics_outer), rng_epochs
        )

        denom = jnp.float32(cfg.update_epochs * cfg.num_minibatches)
        avg_metrics = jax.tree_util.tree_map(lambda x: x / denom, accumulated)
        avg_metrics["mean_reward"] = rollout.reward.sum() / jnp.maximum(rollout.done.sum().astype(jnp.float32), jnp.float32(1.0))
        avg_metrics["episodes"] = rollout.done.sum().astype(jnp.float32)
        return params_out, opt_state_out, avg_metrics

    return jax.jit(train_step)
