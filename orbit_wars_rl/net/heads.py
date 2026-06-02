"""Action / value heads.

All heads operate on the encoded planet/global embeddings. Padding rows (and
non-owned planets, for the src head) get logits replaced by a large negative
number so softmax mass on them is zero.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


_LOGIT_NEG_INF = -1e9


def _mask_logits(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Set logits to -inf where ``mask`` is False. ``mask`` broadcasts."""
    return jnp.where(mask, logits, jnp.float32(_LOGIT_NEG_INF))


class SrcHead(nn.Module):
    """Score each planet as a launch source; only planets the agent owns survive.

    v7: input includes per-planet ``remaining_ships_norm`` so the K-step loop
    can pick different src planets across steps (in v6 the head saw a static
    planet_emb and would commit to the same src for all K iterations even
    after that src's ships had been reserved).
    """

    hidden: int = 64

    @nn.compact
    def __call__(
        self,
        planet_emb: jnp.ndarray,
        my_planet_mask: jnp.ndarray,
        remaining_norm: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        # ``remaining_norm`` has shape matching planet_emb leading dims + (P,)
        if remaining_norm is not None:
            x = jnp.concatenate(
                [planet_emb, remaining_norm[..., None]], axis=-1,
            )
        else:
            x = planet_emb
        x = nn.Dense(self.hidden, name="fc1")(x)
        x = nn.gelu(x)
        logits = nn.Dense(1, name="src_score")(x)[..., 0]
        return _mask_logits(logits, my_planet_mask)


class DstHead(nn.Module):
    """Score each planet as a target using cross-attention conditioned on src.

    Padding rows are masked out via ``planet_mask``. We additionally mask out
    the ``src`` planet itself so the policy cannot waste probability mass on
    a same-planet launch (which the env silently downgrades to ``ships=0``).

    Top-team feedback (top_players_rl.txt §230 + DAY2 diagnosis): training
    used to allow ``src==dst`` since ``actions.decode_action`` zeros it out,
    but at inference time argmax frequently collapses to the src planet
    (its own embedding dominates the self-attention score), producing an
    empty kaggle action list. Masking src here keeps train/eval aligned.

    ``my_planet_mask`` is kept in the signature for backward-compatible call
    sites; same-owner arrivals are still allowed as reinforcements.
    """

    d_model: int = 64
    n_heads: int = 4

    @nn.compact
    def __call__(
        self,
        planet_emb: jnp.ndarray,
        src_emb: jnp.ndarray,
        planet_mask: jnp.ndarray,
        my_planet_mask: jnp.ndarray,
        src_idx: jnp.ndarray | None = None,
        reserved_norm: jnp.ndarray | None = None,
        pair_feats: jnp.ndarray | None = None,
        sun_block_mask: jnp.ndarray | None = None,
        flip_block_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """v7: ``reserved_norm`` (per-planet reserved ship ratio, same leading
        dims as planet_emb but no embedding axis) is appended so that the head
        can avoid sending the next fleet to a target that is already saturated
        with our incoming ships.

        f26: ``pair_feats`` (..., P, 4) appended to planet rows
        (dist_src_dst, sun_risk, ships_needed, pair_flip_bin5). When
        f29: ``pair_feats`` (..., P, 5) adds pair_margin_norm.
        ``sun_block_mask`` (..., P) is provided, dst whose src->dst path
        crosses the sun guard are hard-masked to ``-inf`` logits *unless*
        every candidate is blocked (fallback to standard mask). That
        prevents catastrophic empty-action turns near the sun.

        f31: ``flip_block_mask`` (..., P) blocks enemy/neutral dst that
        cannot be flipped with floor(rem[src]*0.7); same fallback rule.
        """
        del my_planet_mask  # kept for backward-compatible call sites
        is_batched = planet_emb.ndim == 3
        if not is_batched:
            planet_emb = planet_emb[None, ...]
            src_emb = src_emb[None, ...]
            planet_mask = planet_mask[None, ...]
            if src_idx is not None:
                src_idx = src_idx[None, ...] if src_idx.ndim == 0 else src_idx
            if reserved_norm is not None:
                reserved_norm = reserved_norm[None, ...]
            if pair_feats is not None:
                pair_feats = pair_feats[None, ...]
            if sun_block_mask is not None:
                sun_block_mask = sun_block_mask[None, ...]
            if flip_block_mask is not None:
                flip_block_mask = flip_block_mask[None, ...]

        q = src_emb[:, None, :]
        attn_mask = planet_mask[:, None, None, :]
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            name="cross_attn",
        )(q, planet_emb, mask=attn_mask)
        cond = attended[:, 0, :]

        # planet rows: include reserved_norm + pair_feats as extra scalars
        extras = []
        if reserved_norm is not None:
            extras.append(reserved_norm[..., None])
        if pair_feats is not None:
            extras.append(pair_feats)
        if extras:
            planet_rows = jnp.concatenate([planet_emb] + extras, axis=-1)
        else:
            planet_rows = planet_emb
        joined = jnp.concatenate(
            [planet_rows, jnp.broadcast_to(cond[:, None, :], planet_rows.shape[:-1] + (cond.shape[-1],))],
            axis=-1,
        )
        x = nn.Dense(self.d_model, name="dst_fc1")(joined)
        x = nn.gelu(x)
        logits = nn.Dense(1, name="dst_score")(x)[..., 0]

        # Build effective mask: planet_mask AND NOT src (one-hot).
        eff_mask = planet_mask
        if src_idx is not None:
            P = planet_emb.shape[1]
            src_one_hot = jax.nn.one_hot(src_idx, P, dtype=jnp.bool_)  # (B, P)
            eff_mask = planet_mask & jnp.logical_not(src_one_hot)
        # Apply sun-block mask if any non-blocked candidate exists; otherwise
        # fall back to the standard mask so the head still produces a valid
        # categorical (downstream code may still skip via emit head).
        if sun_block_mask is not None:
            allowed_after_sun = eff_mask & jnp.logical_not(sun_block_mask)
            any_allowed = allowed_after_sun.any(axis=-1, keepdims=True)
            eff_mask = jnp.where(any_allowed, allowed_after_sun, eff_mask)
        if flip_block_mask is not None:
            allowed_after_flip = eff_mask & jnp.logical_not(flip_block_mask)
            any_allowed = allowed_after_flip.any(axis=-1, keepdims=True)
            eff_mask = jnp.where(any_allowed, allowed_after_flip, eff_mask)
        logits = _mask_logits(logits, eff_mask)
        if not is_batched:
            logits = jnp.squeeze(logits, axis=0)
        return logits


class PctHead(nn.Module):
    """Bin head over (src_emb, dst_emb, global_emb) -> num_pct_bins logits."""

    num_bins: int
    hidden: int = 64

    @nn.compact
    def __call__(
        self,
        src_emb: jnp.ndarray,
        dst_emb: jnp.ndarray,
        global_emb: jnp.ndarray,
        src_remaining_norm: jnp.ndarray | None = None,
        pair_feats: jnp.ndarray | None = None,
        pct_low_bin_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """v7: ``src_remaining_norm`` is a scalar = log1p(remaining_at_src)/8
        so the pct head knows how many ships are available right now.

        f26: ``pair_feats`` (..., 2) =
            [min_bin_norm = smallest bin index that flips garr_dst / 7,
             pair_flip_bin5 = float(floor(rem*0.7) > garr_dst)]
        Direct index signal so pct head learns "pick at least this bin".

        f29: ``pct_low_bin_mask`` (..., num_bins) bool -- bins below
        min_flip_bin are hard-masked to ``-inf`` unless all bins blocked
        (fallback to unmasked logits).
        """
        feats = [src_emb, dst_emb, global_emb]
        if src_remaining_norm is not None:
            feats.append(src_remaining_norm[..., None])
        if pair_feats is not None:
            feats.append(pair_feats)
        x = jnp.concatenate(feats, axis=-1)
        x = nn.Dense(self.hidden, name="fc1")(x)
        x = nn.gelu(x)
        logits = nn.Dense(self.num_bins, name="logits")(x)
        if pct_low_bin_mask is not None:
            allowed = pct_low_bin_mask
            any_allowed = allowed.any(axis=-1, keepdims=True)
            eff_mask = jnp.where(any_allowed, allowed, jnp.ones_like(allowed))
            logits = _mask_logits(logits, eff_mask)
        return logits


class EmitHead(nn.Module):
    """Binary head: continue emitting (1) vs stop (0) at this autoregressive step.

    Conditioned on the encoded global state, the running planet/fleet pools,
    and the current autoregressive step index (one-hot). Used by the
    multi-action policy to decide when to stop launching fleets in a turn.

    Output: logits of shape (..., 2) ordered as [stop_logit, continue_logit]
    so ``argmax==1`` means "keep emitting".

    The bias on the final Dense layer is initialised so that the network
    starts biased toward "continue" (logit ``+continue_bias``) -- otherwise
    the random init lands close to a 50/50 split and the policy quickly
    collapses to "emit 1 fleet and stop" (a local optimum identical to the
    single-action MVP). We let the network learn its way back to a balanced
    distribution via the entropy regulariser + reward signal.
    """

    hidden: int = 64
    max_steps: int = 8
    continue_bias: float = 0.0

    @nn.compact
    def __call__(
        self,
        global_emb: jnp.ndarray,
        planet_pool: jnp.ndarray,
        step_idx: jnp.ndarray,
        total_remaining_norm: jnp.ndarray | None = None,
        pair_feats_g: jnp.ndarray | None = None,
        emit_force_stop: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """v7: ``total_remaining_norm`` (scalar) = log1p(sum of remaining ships
        across my planets) / 8.  Tells the emit head how much firepower we
        still have available this turn -- prevents the policy from emitting
        a 6th fleet when the source planet is already empty.

        f26: ``pair_feats_g`` (..., 6) per-step global pair signals:
            [emit_worth_it, best_pair_margin_norm,
             home_remain_ratio, total_remain_ratio,
             feasible_target_count_norm, surplus_ratio]
        directly answers "should I emit one more fleet from this state?".

        f29: ``emit_force_stop`` (...,) bool -- when True (step>0 and no
        worth-it pair), continue logit is set to ``-inf``.
        """
        step_oh = jax.nn.one_hot(step_idx, self.max_steps, dtype=global_emb.dtype)
        step_oh = jnp.broadcast_to(step_oh, global_emb.shape[:-1] + (self.max_steps,))
        feats = [global_emb, planet_pool, step_oh]
        if total_remaining_norm is not None:
            feats.append(total_remaining_norm[..., None])
        if pair_feats_g is not None:
            feats.append(pair_feats_g)
        x = jnp.concatenate(feats, axis=-1)
        x = nn.Dense(self.hidden, name="fc1")(x)
        x = nn.gelu(x)
        bias_init = lambda key, shape, dtype=jnp.float32: jnp.array(
            [0.0, self.continue_bias], dtype=dtype
        )
        logits = nn.Dense(2, name="logits", bias_init=bias_init)(x)
        if emit_force_stop is not None:
            # logits[..., 0]=stop, logits[..., 1]=continue
            logits = jnp.where(
                emit_force_stop[..., None],
                jnp.array([0.0, _LOGIT_NEG_INF], dtype=logits.dtype),
                logits,
            )
        return logits


class ValueHead(nn.Module):
    """Scalar value from a multi-query cross-attention over per-entity embeddings.

    Top-team feedback (top_players_rl.txt §65: "multi-query ValueHead") was
    flagged as one of the F12 improvements. Compared to the original
    ``concat(global, planet_pool, fleet_pool) -> MLP -> scalar`` design, this
    version uses N learned query tokens that cross-attend to all valid
    planet+fleet embeddings, then pools the queries into a scalar.

    This buys us:
      * The value head can attend to *specific* planets / fleets rather than
        a mean-pooled summary -- crucial when one planet (e.g., the enemy
        home with 200 ships) dominates the outcome but is averaged away.
      * More capacity proportional to (n_queries * d_model) so value can
        learn faster without value_coef blowing up policy advantage signal.
    """

    d_model: int = 64
    n_queries: int = 4
    n_heads: int = 4
    hidden: int = 64

    @nn.compact
    def __call__(
        self,
        global_emb: jnp.ndarray,
        planet_emb: jnp.ndarray,
        planet_mask: jnp.ndarray,
        fleet_emb: jnp.ndarray,
        fleet_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        is_batched = planet_emb.ndim == 3
        if not is_batched:
            global_emb = global_emb[None, :]
            planet_emb = planet_emb[None, ...]
            planet_mask = planet_mask[None, ...]
            fleet_emb = fleet_emb[None, ...]
            fleet_mask = fleet_mask[None, ...]
        B = planet_emb.shape[0]

        queries = self.param(
            "queries",
            nn.initializers.normal(stddev=0.02),
            (self.n_queries, self.d_model),
            jnp.float32,
        )
        q = jnp.broadcast_to(queries[None, :, :], (B, self.n_queries, self.d_model))
        # Condition queries on global state so they're not just static slots.
        g_proj = nn.Dense(self.d_model, name="q_cond")(global_emb)[:, None, :]
        q = q + g_proj

        kv = jnp.concatenate([planet_emb, fleet_emb], axis=1)
        kv_mask = jnp.concatenate([planet_mask, fleet_mask], axis=1)
        attn_mask = kv_mask[:, None, None, :]
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            name="value_attn",
        )(q, kv, mask=attn_mask)

        # Concat all attended query outputs + global, then MLP to scalar.
        pooled = attended.reshape(B, self.n_queries * self.d_model)
        joined = jnp.concatenate([pooled, global_emb], axis=-1)
        x = nn.Dense(self.hidden, name="fc1")(joined)
        x = nn.gelu(x)
        value = nn.Dense(1, name="value")(x)[..., 0]

        if not is_batched:
            value = jnp.squeeze(value, axis=0)
        return value
