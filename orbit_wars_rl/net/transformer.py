"""Tiny entity transformer encoder.

Input streams (each [B, N_i, F_i]) are projected to ``d_model``, tagged with a
learned type embedding (global / planet / fleet) and concatenated into a single
[B, 1 + P + F, d] sequence. A pre-norm self-attention stack consumes that with
a global key padding mask. Global token (index 0) is always valid.
"""

from __future__ import annotations

from typing import Callable

import flax.linen as nn
import jax.numpy as jnp


class _MLPBlock(nn.Module):
    d_model: int
    ff_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.ff_dim, name="fc1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model, name="fc2")(x)
        return x


class _EncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    ff_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        h = nn.LayerNorm(name="ln1")(x)
        attn = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            name="attn",
        )(h, mask=mask)
        x = x + attn

        h = nn.LayerNorm(name="ln2")(x)
        x = x + _MLPBlock(self.d_model, self.ff_dim, name="mlp")(h)
        return x


class EntityTransformer(nn.Module):
    """Encode (global, planets, fleets) into per-entity embeddings + global emb."""

    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    ff_dim: int = 128

    @nn.compact
    def __call__(
        self,
        planet_feats: jnp.ndarray,
        planet_mask: jnp.ndarray,
        fleet_feats: jnp.ndarray,
        fleet_mask: jnp.ndarray,
        global_feats: jnp.ndarray,
    ) -> dict:
        is_batched = planet_feats.ndim == 3
        if not is_batched:
            planet_feats = planet_feats[None, ...]
            planet_mask = planet_mask[None, ...]
            fleet_feats = fleet_feats[None, ...]
            fleet_mask = fleet_mask[None, ...]
            global_feats = global_feats[None, ...]

        B, P, _ = planet_feats.shape
        F = fleet_feats.shape[1]

        type_embed = self.param(
            "type_embed",
            nn.initializers.normal(stddev=0.02),
            (3, self.d_model),
            jnp.float32,
        )

        planet_tokens = nn.Dense(self.d_model, name="planet_proj")(planet_feats)
        fleet_tokens = nn.Dense(self.d_model, name="fleet_proj")(fleet_feats)
        global_token = nn.Dense(self.d_model, name="global_proj")(global_feats)[:, None, :]

        planet_tokens = planet_tokens + type_embed[1][None, None, :]
        fleet_tokens = fleet_tokens + type_embed[2][None, None, :]
        global_token = global_token + type_embed[0][None, None, :]

        tokens = jnp.concatenate([global_token, planet_tokens, fleet_tokens], axis=1)

        global_valid = jnp.ones((B, 1), dtype=jnp.bool_)
        key_mask = jnp.concatenate([global_valid, planet_mask, fleet_mask], axis=1)
        attn_mask = key_mask[:, None, None, :].astype(jnp.bool_)
        attn_mask = jnp.broadcast_to(attn_mask, (B, 1, tokens.shape[1], tokens.shape[1]))

        x = tokens
        for i in range(self.n_layers):
            x = _EncoderBlock(self.d_model, self.n_heads, self.ff_dim, name=f"block{i}")(x, attn_mask)
        x = nn.LayerNorm(name="ln_out")(x)

        global_emb = x[:, 0, :]
        planet_emb = x[:, 1:1 + P, :]
        fleet_emb = x[:, 1 + P:, :]

        mask_f = planet_mask[:, :, None].astype(jnp.float32)
        planet_pool = (planet_emb * mask_f).sum(axis=1) / jnp.maximum(mask_f.sum(axis=1), 1.0)
        fleet_mask_f = fleet_mask[:, :, None].astype(jnp.float32)
        fleet_pool = (fleet_emb * fleet_mask_f).sum(axis=1) / jnp.maximum(fleet_mask_f.sum(axis=1), 1.0)

        out = dict(
            global_emb=global_emb,
            planet_emb=planet_emb,
            fleet_emb=fleet_emb,
            planet_pool=planet_pool,
            fleet_pool=fleet_pool,
        )
        if not is_batched:
            out = {k: jnp.squeeze(v, axis=0) for k, v in out.items()}
        return out
