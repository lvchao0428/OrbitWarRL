"""Action / value heads.

All heads operate on the encoded planet/global embeddings. Padding rows (and
non-owned planets, for the src head) get logits replaced by a large negative
number so softmax mass on them is zero.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


_LOGIT_NEG_INF = -1e9


def _mask_logits(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Set logits to -inf where ``mask`` is False. ``mask`` broadcasts."""
    return jnp.where(mask, logits, jnp.float32(_LOGIT_NEG_INF))


class SrcHead(nn.Module):
    """Score each planet as a launch source; only planets the agent owns survive."""

    @nn.compact
    def __call__(self, planet_emb: jnp.ndarray, my_planet_mask: jnp.ndarray) -> jnp.ndarray:
        logits = nn.Dense(1, name="src_score")(planet_emb)[..., 0]
        return _mask_logits(logits, my_planet_mask)


class DstHead(nn.Module):
    """Score each planet as a target using cross-attention conditioned on src."""

    d_model: int = 64
    n_heads: int = 4

    @nn.compact
    def __call__(
        self,
        planet_emb: jnp.ndarray,
        src_emb: jnp.ndarray,
        planet_mask: jnp.ndarray,
        my_planet_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        is_batched = planet_emb.ndim == 3
        if not is_batched:
            planet_emb = planet_emb[None, ...]
            src_emb = src_emb[None, ...]
            planet_mask = planet_mask[None, ...]
            my_planet_mask = my_planet_mask[None, ...]

        q = src_emb[:, None, :]
        attn_mask = planet_mask[:, None, None, :]
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            name="cross_attn",
        )(q, planet_emb, mask=attn_mask)
        cond = attended[:, 0, :]

        joined = jnp.concatenate([planet_emb, jnp.broadcast_to(cond[:, None, :], planet_emb.shape)], axis=-1)
        logits = nn.Dense(1, name="dst_score")(joined)[..., 0]

        valid = planet_mask & jnp.logical_not(my_planet_mask)
        logits = _mask_logits(logits, valid)
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
    ) -> jnp.ndarray:
        x = jnp.concatenate([src_emb, dst_emb, global_emb], axis=-1)
        x = nn.Dense(self.hidden, name="fc1")(x)
        x = nn.gelu(x)
        return nn.Dense(self.num_bins, name="logits")(x)


class ValueHead(nn.Module):
    """Scalar value from pooled global + planet + fleet embeddings."""

    hidden: int = 64

    @nn.compact
    def __call__(
        self,
        global_emb: jnp.ndarray,
        planet_pool: jnp.ndarray,
        fleet_pool: jnp.ndarray,
    ) -> jnp.ndarray:
        x = jnp.concatenate([global_emb, planet_pool, fleet_pool], axis=-1)
        x = nn.Dense(self.hidden, name="fc1")(x)
        x = nn.gelu(x)
        return nn.Dense(1, name="value")(x)[..., 0]
