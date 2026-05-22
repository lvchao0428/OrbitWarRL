"""PPO loss + optax warmup-cosine schedule for the multi-action policy.

Adaptations vs the single-action MVP:
* The behaviour log-prob for a turn is the sum over autoregressive steps
  ``sum_k emit_mask[k] * (src_logp_k + dst_logp_k + pct_logp_k) + emit_logp_k``
  (the emit head's logp is gated on "free choice" inside the sampler, so we
  can sum it unconditionally -- masked steps contributed 0).
* Per-head entropy is averaged across (T*B) and across valid emit steps (so
  ``ent_coef`` keeps the same scale regardless of K).
* PPO ratio is one ratio per turn, not per autoregressive step. Value/clip
  knobs are unchanged.

Three independent entropy coefficients (src / dst / pct), plus a fourth
``ent_coef_emit`` for the emit/stop head, value clipping, advantage
normalization, gradient clipping.
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
    ent_coef_emit: float = 0.04
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
    """Flatten rollout [N,...] into an EncodedObs along leading N=T*B (or T*B already flat)."""
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


def _logp_entropy(logits: jnp.ndarray, idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(idx, logits.shape[-1], dtype=logits.dtype)
    logp = (log_probs * one_hot).sum(axis=-1)
    probs = jax.nn.softmax(logits, axis=-1)
    entropy = -(probs * jnp.where(probs > 0, jnp.log(probs + 1e-20), 0.0)).sum(axis=-1)
    return logp, entropy


def _turn_logp_sum(
    src_logp_k: jnp.ndarray,    # [N, K]
    dst_logp_k: jnp.ndarray,    # [N, K]
    pct_logp_k: jnp.ndarray,    # [N, K]
    emit_logp_k: jnp.ndarray,   # [N, K]
    emit_mask: jnp.ndarray,     # [N, K] bool
) -> jnp.ndarray:
    """Sum per-step logps into a single per-turn logp.

    The (src, dst, pct) heads contribute only when ``emit_mask`` is True for
    that step -- the sampler already zeroed their per-step logp when not
    emitting, but masking here is defensive and lets ``evaluate`` reuse the
    raw logits.
    """
    mask_f = emit_mask.astype(src_logp_k.dtype)
    triple_sum = ((src_logp_k + dst_logp_k + pct_logp_k) * mask_f).sum(axis=-1)
    # emit logp is already zeroed on non-free-choice steps by the sampler.
    emit_sum = emit_logp_k.sum(axis=-1)
    return triple_sum + emit_sum


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
    out = model.apply(
        params,
        obs,
        rollout.src_idx,
        rollout.dst_idx,
        rollout.pct_bin,
        rollout.emit_mask,
        rollout.planet_ships_raw,
        method=ActorCritic.evaluate,
    )

    src_logp_new_k, src_ent_k = _logp_entropy(out.src_logits, rollout.src_idx)
    dst_logp_new_k, dst_ent_k = _logp_entropy(out.dst_logits, rollout.dst_idx)
    pct_logp_new_k, pct_ent_k = _logp_entropy(out.pct_logits, rollout.pct_bin)
    emit_logp_new_k, emit_ent_k = _logp_entropy(
        out.emit_logits, rollout.emit_mask.astype(jnp.int32)
    )

    # Reuse the sampler's exact gating mask so the same logps are credited
    # on both sides (otherwise the PPO ratio drifts away from 1 even at the
    # first epoch and the policy chases its own tail).
    free_f = rollout.emit_free_mask.astype(rollout.emit_logp.dtype)
    src_new_gated = src_logp_new_k * rollout.emit_mask.astype(src_logp_new_k.dtype)
    dst_new_gated = dst_logp_new_k * rollout.emit_mask.astype(dst_logp_new_k.dtype)
    pct_new_gated = pct_logp_new_k * rollout.emit_mask.astype(pct_logp_new_k.dtype)
    emit_new_gated = emit_logp_new_k * free_f

    old_turn_logp = (rollout.src_logp + rollout.dst_logp + rollout.pct_logp).sum(axis=-1) + rollout.emit_logp.sum(axis=-1)
    new_turn_logp = src_new_gated.sum(axis=-1) + dst_new_gated.sum(axis=-1) + pct_new_gated.sum(axis=-1) + emit_new_gated.sum(axis=-1)

    ratio = jnp.exp(new_turn_logp - old_turn_logp)

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

    # Entropies: average over (N, K) but only over steps where the head was an
    # actual free choice. For src/dst/pct that's emit_mask; for emit that's
    # the sampler-stored free_choice mask.
    mask_emit = rollout.emit_mask.astype(jnp.float32)
    n_emit = jnp.maximum(mask_emit.sum(), jnp.float32(1.0))
    ent_src = (src_ent_k * mask_emit).sum() / n_emit
    ent_dst = (dst_ent_k * mask_emit).sum() / n_emit
    ent_pct = (pct_ent_k * mask_emit).sum() / n_emit
    n_free = jnp.maximum(free_f.sum(), jnp.float32(1.0))
    ent_emit = (emit_ent_k * free_f).sum() / n_free

    entropy_loss = -(
        cfg.ent_coef_src * ent_src
        + cfg.ent_coef_dst * ent_dst
        + cfg.ent_coef_pct * ent_pct
        + cfg.ent_coef_emit * ent_emit
    )

    total_loss = pg_loss + cfg.value_coef * v_loss + entropy_loss
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > cfg.clip_eps).astype(jnp.float32))
    approx_kl = jnp.mean(old_turn_logp - new_turn_logp)

    metrics = dict(
        pg_loss=pg_loss,
        v_loss=v_loss,
        ent_src=ent_src,
        ent_dst=ent_dst,
        ent_pct=ent_pct,
        ent_emit=ent_emit,
        clip_frac=clip_frac,
        approx_kl=approx_kl,
        ratio_mean=ratio.mean(),
        value_mean=value_pred.mean(),
        return_mean=returns.mean(),
        adv_mean=adv_mean,
        adv_std=adv_std,
        mean_emits_per_turn=mask_emit.sum(axis=-1).mean(),
    )
    return total_loss, metrics


def _flatten_rollout(rollout: Rollout, advantages: jnp.ndarray, returns: jnp.ndarray):
    """[T, B, ...] -> [T*B, ...] for all per-step leaves."""

    def f(x):
        return x.reshape((-1,) + x.shape[2:])

    flat_rollout = Rollout(
        obs_planet_feats=f(rollout.obs_planet_feats),
        obs_planet_mask=f(rollout.obs_planet_mask),
        obs_fleet_feats=f(rollout.obs_fleet_feats),
        obs_fleet_mask=f(rollout.obs_fleet_mask),
        obs_global_feats=f(rollout.obs_global_feats),
        obs_my_planet_mask=f(rollout.obs_my_planet_mask),
        planet_ships_raw=f(rollout.planet_ships_raw),
        src_idx=f(rollout.src_idx),
        dst_idx=f(rollout.dst_idx),
        pct_bin=f(rollout.pct_bin),
        emit_mask=f(rollout.emit_mask),
        emit_free_mask=f(rollout.emit_free_mask),
        src_logp=f(rollout.src_logp),
        dst_logp=f(rollout.dst_logp),
        pct_logp=f(rollout.pct_logp),
        emit_logp=f(rollout.emit_logp),
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
        planet_ships_raw=take(rollout.planet_ships_raw),
        src_idx=take(rollout.src_idx),
        dst_idx=take(rollout.dst_idx),
        pct_bin=take(rollout.pct_bin),
        emit_mask=take(rollout.emit_mask),
        emit_free_mask=take(rollout.emit_free_mask),
        src_logp=take(rollout.src_logp),
        dst_logp=take(rollout.dst_logp),
        pct_logp=take(rollout.pct_logp),
        emit_logp=take(rollout.emit_logp),
        value=take(rollout.value),
        reward=take(rollout.reward),
        done=take(rollout.done),
        last_value=rollout.last_value,
    )
    return sub, take(advs), take(rets)


_ZERO_METRICS_KEYS = (
    "pg_loss", "v_loss", "ent_src", "ent_dst", "ent_pct", "ent_emit",
    "clip_frac", "approx_kl", "ratio_mean", "value_mean", "return_mean",
    "adv_mean", "adv_std", "mean_emits_per_turn", "loss", "grad_norm",
)


def _zero_metrics_dict():
    return {k: jnp.float32(0.0) for k in _ZERO_METRICS_KEYS}


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

            (params_new, opt_state_new, summed), _ = jax.lax.scan(
                mb_step,
                (params_e, opt_state_e, _zero_metrics_dict()),
                jnp.arange(cfg.num_minibatches),
            )
            summed_combined = jax.tree_util.tree_map(lambda a, b: a + b, accum, summed)
            return (params_new, opt_state_new, summed_combined), None

        rng_epochs = jax.random.split(rng, cfg.update_epochs)
        (params_out, opt_state_out, accumulated), _ = jax.lax.scan(
            epoch_step, (params, opt_state, _zero_metrics_dict()), rng_epochs
        )

        denom = jnp.float32(cfg.update_epochs * cfg.num_minibatches)
        avg_metrics = jax.tree_util.tree_map(lambda x: x / denom, accumulated)
        # Step-level mean reward (includes dense shaping). Useful as a coarse
        # progress signal when shaping is on; for win-rate use eval_every.
        avg_metrics["mean_step_reward"] = rollout.reward.mean()
        # Terminal reward signal: average of reward at done steps only.
        done_f = rollout.done.astype(jnp.float32)
        n_done = jnp.maximum(done_f.sum(), jnp.float32(1.0))
        avg_metrics["mean_terminal_reward"] = (rollout.reward * done_f).sum() / n_done
        avg_metrics["episodes"] = rollout.done.sum().astype(jnp.float32)
        return params_out, opt_state_out, avg_metrics

    return jax.jit(train_step)
