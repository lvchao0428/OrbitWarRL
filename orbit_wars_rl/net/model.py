"""ActorCritic: encoder + 3 policy heads + value head; supports sample / argmax."""

from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.features import EncodedObs
from orbit_wars_rl.net.transformer import EntityTransformer
from orbit_wars_rl.net.heads import SrcHead, DstHead, PctHead, ValueHead


class ActorCriticOutput(NamedTuple):
    src_logits: jnp.ndarray
    dst_logits: jnp.ndarray
    pct_logits: jnp.ndarray
    value: jnp.ndarray


class SampledAction(NamedTuple):
    src_idx: jnp.ndarray
    dst_idx: jnp.ndarray
    pct_bin: jnp.ndarray
    src_logp: jnp.ndarray
    dst_logp: jnp.ndarray
    pct_logp: jnp.ndarray
    src_entropy: jnp.ndarray
    dst_entropy: jnp.ndarray
    pct_entropy: jnp.ndarray
    value: jnp.ndarray


def _categorical_logp_entropy(
    logits: jnp.ndarray, idx: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(idx, logits.shape[-1], dtype=logits.dtype)
    logp = (log_probs * one_hot).sum(axis=-1)
    probs = jax.nn.softmax(logits, axis=-1)
    entropy = -(probs * jnp.where(probs > 0, jnp.log(probs + 1e-20), 0.0)).sum(axis=-1)
    return logp, entropy


def _gather_planet_emb(planet_emb: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """Pick the embedding for planet ``idx``. Supports batched leading dim."""
    if planet_emb.ndim == 3:
        B = planet_emb.shape[0]
        return planet_emb[jnp.arange(B), idx]
    return planet_emb[idx]


class ActorCritic(nn.Module):
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    ff_dim: int = 128
    num_pct_bins: int = constants.NUM_PCT_BINS

    def setup(self) -> None:
        self.encoder = EntityTransformer(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            ff_dim=self.ff_dim,
        )
        self.src_head = SrcHead()
        self.dst_head = DstHead(d_model=self.d_model, n_heads=self.n_heads)
        self.pct_head = PctHead(num_bins=self.num_pct_bins)
        self.value_head = ValueHead()

    def _encode(self, obs: EncodedObs) -> dict:
        return self.encoder(
            obs.planet_feats,
            obs.planet_mask,
            obs.fleet_feats,
            obs.fleet_mask,
            obs.global_feats,
        )

    def evaluate(
        self,
        obs: EncodedObs,
        src_idx: jnp.ndarray,
        dst_idx: jnp.ndarray,
    ) -> ActorCriticOutput:
        """Compute all logits + value given pre-chosen src/dst (for PPO update)."""
        enc = self._encode(obs)
        src_logits = self.src_head(enc["planet_emb"], obs.my_planet_mask)
        src_emb = _gather_planet_emb(enc["planet_emb"], src_idx)
        dst_logits = self.dst_head(enc["planet_emb"], src_emb, obs.planet_mask, obs.my_planet_mask)
        dst_emb = _gather_planet_emb(enc["planet_emb"], dst_idx)
        pct_logits = self.pct_head(src_emb, dst_emb, enc["global_emb"])
        value = self.value_head(enc["global_emb"], enc["planet_pool"], enc["fleet_pool"])
        return ActorCriticOutput(src_logits, dst_logits, pct_logits, value)

    def __call__(self, obs: EncodedObs, rng: jnp.ndarray, *, deterministic: bool = False) -> SampledAction:
        """Sample (src, dst, pct) and return logps + entropies + value.

        ``deterministic=True`` picks argmax for all heads (use at eval time).
        Falls back to slot 0 if a player has no valid source planet.
        """
        enc = self._encode(obs)
        src_logits = self.src_head(enc["planet_emb"], obs.my_planet_mask)

        any_src = obs.my_planet_mask.any(axis=-1)

        r_src, r_dst, r_pct = jax.random.split(rng, 3)

        if deterministic:
            src_idx = jnp.argmax(src_logits, axis=-1)
        else:
            src_idx = jax.random.categorical(r_src, src_logits, axis=-1)
        src_idx = src_idx.astype(jnp.int32)

        src_logp, src_entropy = _categorical_logp_entropy(src_logits, src_idx)
        src_logp = jnp.where(any_src, src_logp, jnp.float32(0.0))
        src_entropy = jnp.where(any_src, src_entropy, jnp.float32(0.0))

        src_emb = _gather_planet_emb(enc["planet_emb"], src_idx)
        dst_logits = self.dst_head(enc["planet_emb"], src_emb, obs.planet_mask, obs.my_planet_mask)
        if deterministic:
            dst_idx = jnp.argmax(dst_logits, axis=-1)
        else:
            dst_idx = jax.random.categorical(r_dst, dst_logits, axis=-1)
        dst_idx = dst_idx.astype(jnp.int32)
        dst_logp, dst_entropy = _categorical_logp_entropy(dst_logits, dst_idx)

        dst_emb = _gather_planet_emb(enc["planet_emb"], dst_idx)
        pct_logits = self.pct_head(src_emb, dst_emb, enc["global_emb"])
        if deterministic:
            pct_idx = jnp.argmax(pct_logits, axis=-1)
        else:
            pct_idx = jax.random.categorical(r_pct, pct_logits, axis=-1)
        pct_idx = pct_idx.astype(jnp.int32)
        pct_logp, pct_entropy = _categorical_logp_entropy(pct_logits, pct_idx)

        value = self.value_head(enc["global_emb"], enc["planet_pool"], enc["fleet_pool"])

        return SampledAction(
            src_idx=src_idx,
            dst_idx=dst_idx,
            pct_bin=pct_idx,
            src_logp=src_logp,
            dst_logp=dst_logp,
            pct_logp=pct_logp,
            src_entropy=src_entropy,
            dst_entropy=dst_entropy,
            pct_entropy=pct_entropy,
            value=value,
        )
