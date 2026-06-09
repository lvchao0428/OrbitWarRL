"""ActorCritic with autoregressive multi-fleet head.

Each turn the policy emits up to ``K = MAX_FLEETS_PER_TURN`` fleets via an
autoregressive scan:

  for t in [0, K):
      emit_logits = EmitHead(global, planet_pool, step=t)
      if t == 0: force emit (else the turn is empty)
      emit_t = bernoulli(emit_logits)  # 1 = continue, 0 = stop
      if not still_emitting: replay zeros (logp_t = 0)
      src_logits = SrcHead(planet_emb) masked by (my_planets & remaining_ships > 0)
      src_t = categorical(src_logits)
      dst_logits = DstHead(planet_emb, src_emb)
      dst_t = categorical(dst_logits)
      pct_logits = PctHead(src_emb, dst_emb, global)
      pct_t = categorical(pct_logits)
      reserved += ships(src, pct)        # for the next iteration's mask

A "soft" reserved-ships book-keeping mirrors what ``dynamics.launch_fleets``
actually does, so the policy never confidently emits an action the env will
silently drop.

The encoder runs **once** per turn -- the encoded planet embeddings are reused
across all K autoregressive steps. Only the EmitHead/SrcHead/PctHead are
re-run per step. ``DstHead`` is also re-run but it's cheap.
"""

from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.features import EncodedObs
from orbit_wars_rl.features.pair import (
    dst_pair_features_batched,
    dst_flip_block_mask,
    emit_pair_globals,
    pct_pair_features,
    pct_min_bin_index,
    pct_low_bin_mask,
)
from orbit_wars_rl.net.transformer import EntityTransformer
from orbit_wars_rl.net.heads import SrcHead, DstHead, PctHead, ValueHead, ZeroSumValueHead, EmitHead


_PCT_BIN_TABLE = jnp.array(constants.PCT_BIN_VALUES, dtype=jnp.float32)


class ActorCriticOutput(NamedTuple):
    """Logits arrays from ``evaluate``. Each [..., K, ...]."""

    src_logits: jnp.ndarray         # (..., K, P)
    dst_logits: jnp.ndarray         # (..., K, P)
    pct_logits: jnp.ndarray         # (..., K, num_pct_bins)
    emit_logits: jnp.ndarray        # (..., K, 2)
    value: jnp.ndarray              # (...,)


class SampledMultiAction(NamedTuple):
    """Sampled K-step action with per-step logp/entropy and overall value."""

    src_idx: jnp.ndarray          # (..., K) int32
    dst_idx: jnp.ndarray          # (..., K) int32
    pct_bin: jnp.ndarray          # (..., K) int32
    emit_mask: jnp.ndarray        # (..., K) bool   -- True iff this step actually launches
    emit_free_mask: jnp.ndarray   # (..., K) bool   -- True iff EmitHead was a free choice (not forced)

    src_logp: jnp.ndarray         # (..., K) float -- per-step, 0 where not emitting
    dst_logp: jnp.ndarray         # (..., K)
    pct_logp: jnp.ndarray         # (..., K)
    emit_logp: jnp.ndarray        # (..., K)

    src_entropy: jnp.ndarray      # (..., K) float -- per-step, 0 where not emitting
    dst_entropy: jnp.ndarray      # (..., K)
    pct_entropy: jnp.ndarray      # (..., K)
    emit_entropy: jnp.ndarray     # (..., K)

    value: jnp.ndarray            # (...,) float


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


def _ships_to_send_for_step(
    planet_ships: jnp.ndarray,   # (P,) int32 -- raw garrison
    reserved: jnp.ndarray,       # (P,) int32 -- already reserved this turn
    src_idx: jnp.ndarray,        # () int32
    pct_bin: jnp.ndarray,        # () int32
) -> jnp.ndarray:
    """Mirrors ``actions.decode_action`` ships logic (without owner/dst checks)."""
    avail = jnp.maximum(planet_ships[src_idx] - reserved[src_idx], jnp.int32(0))
    pct_idx = jnp.clip(pct_bin, 0, constants.NUM_PCT_BINS - 1)
    pct = _PCT_BIN_TABLE[pct_idx]
    raw = jnp.maximum(
        jnp.int32(1),
        jnp.floor(avail.astype(jnp.float32) * pct).astype(jnp.int32),
    )
    return jnp.minimum(raw, avail)


def _ships_to_send_for_step_batched(
    planet_ships: jnp.ndarray,   # (..., P) int32
    reserved: jnp.ndarray,       # (..., P) int32
    src_idx: jnp.ndarray,        # (...,) int32
    pct_bin: jnp.ndarray,        # (...,) int32
) -> jnp.ndarray:
    """Same as ``_ships_to_send_for_step`` but supports a leading batch dim.

    Used by the multi-action sample/evaluate loops which run with leading
    ``[B]`` (from vmap or rollout scan over T*B).
    """
    if planet_ships.ndim == 1:
        return _ships_to_send_for_step(planet_ships, reserved, src_idx, pct_bin)
    # Gather along the last (planet) axis with src_idx.
    src_ships = jnp.take_along_axis(planet_ships, src_idx[..., None], axis=-1)[..., 0]
    src_reserved = jnp.take_along_axis(reserved, src_idx[..., None], axis=-1)[..., 0]
    avail = jnp.maximum(src_ships - src_reserved, jnp.int32(0))
    pct_idx = jnp.clip(pct_bin, 0, constants.NUM_PCT_BINS - 1)
    pct = _PCT_BIN_TABLE[pct_idx]
    raw = jnp.maximum(
        jnp.int32(1),
        jnp.floor(avail.astype(jnp.float32) * pct).astype(jnp.int32),
    )
    return jnp.minimum(raw, avail)


def _scatter_add(buf: jnp.ndarray, idx: jnp.ndarray, val: jnp.ndarray) -> jnp.ndarray:
    """Add ``val`` into ``buf`` at index ``idx`` along the last axis.

    Supports both 1D and batched buffers (broadcasting idx/val accordingly).
    Equivalent to ``buf[..., idx] += val`` but jit-friendly.
    """
    if buf.ndim == 1:
        return buf.at[idx].add(val)
    # Build a one-hot scatter add along the last axis.
    P = buf.shape[-1]
    one_hot = jax.nn.one_hot(idx, P, dtype=buf.dtype)  # (..., P)
    return buf + val[..., None] * one_hot


def _remaining_features(
    ships_raw: jnp.ndarray,    # (..., P) int32 -- start-of-turn garrison
    reserved: jnp.ndarray,     # (..., P) int32 -- already promised this turn
    my_mask: jnp.ndarray,      # (..., P) bool  -- our planets
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the per-step reserved-aware features fed into the heads.

    Returns
    -------
    remaining_norm   : (..., P) float32  log1p(remaining)/8
    reserved_norm    : (..., P) float32  log1p(reserved)/8
    total_remaining_norm : (...,)  float32  log1p(sum_of_my_remaining)/8

    All three use log1p / 8 normalisation -- same scaling as the planet
    feature ``log_ships`` in features/encode.py so the heads see comparable
    magnitudes.
    """
    remaining = jnp.maximum(ships_raw - reserved, jnp.int32(0)).astype(jnp.float32)
    reserved_f = jnp.maximum(reserved, jnp.int32(0)).astype(jnp.float32)
    remaining_norm = jnp.log1p(remaining) / jnp.float32(8.0)
    reserved_norm = jnp.log1p(reserved_f) / jnp.float32(8.0)
    my_remaining_total = (remaining * my_mask.astype(remaining.dtype)).sum(axis=-1)
    total_remaining_norm = jnp.log1p(my_remaining_total) / jnp.float32(8.0)
    return remaining_norm, reserved_norm, total_remaining_norm


class ActorCritic(nn.Module):
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    ff_dim: int = 128
    num_pct_bins: int = constants.NUM_PCT_BINS
    max_fleets_per_turn: int = constants.MAX_FLEETS_PER_TURN
    emit_hard_stop: bool = False
    emit_hard_stop_min_step: int = 1
    flip_hard_mask: bool = False
    zero_sum_value: bool = False
    allow_hold: bool = False
    min_pct_bin: int = 0

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
        self.emit_head = EmitHead(max_steps=self.max_fleets_per_turn)
        if self.zero_sum_value:
            self.value_head = ZeroSumValueHead(
                d_model=self.d_model, n_heads=self.n_heads
            )
        else:
            self.value_head = ValueHead(d_model=self.d_model, n_heads=self.n_heads)

    def _encode(self, obs: EncodedObs) -> dict:
        return self.encoder(
            obs.planet_feats,
            obs.planet_mask,
            obs.fleet_feats,
            obs.fleet_mask,
            obs.global_feats,
        )

    # ---- evaluate (PPO): replay pre-sampled K-step actions and recompute logits

    def evaluate(
        self,
        obs: EncodedObs,
        src_idx: jnp.ndarray,    # (..., K) int32
        dst_idx: jnp.ndarray,    # (..., K) int32
        pct_bin: jnp.ndarray,    # (..., K) int32  -- needed to recompute reserved buffer
        emit_mask: jnp.ndarray,  # (..., K) bool   -- which steps actually emitted
        planet_ships_raw: jnp.ndarray,  # (..., P) int32 -- garrison at start of turn (for reserved)
        planet_x: jnp.ndarray,           # (..., P) float -- f26 pair feats
        planet_y: jnp.ndarray,           # (..., P) float
        home_idx: jnp.ndarray,           # (...,) int32
        opp_obs: EncodedObs | None = None,
    ) -> ActorCriticOutput:
        """Compute per-step logits + value given pre-chosen K-step actions.

        Implementation note: we unroll K steps with a Python ``for`` loop because
        ``nn.Module.__call__`` is not allowed inside ``jax.lax.scan`` (flax would
        re-init params each iter). K is static (``self.max_fleets_per_turn``)
        so the unroll is cheap and JIT-friendly.
        """
        enc = self._encode(obs)
        planet_emb = enc["planet_emb"]
        global_emb = enc["global_emb"]
        planet_pool = enc["planet_pool"]
        fleet_emb = enc["fleet_emb"]

        is_batched = planet_emb.ndim == 3
        if not is_batched:
            obs_my = obs.my_planet_mask
            obs_pmask = obs.planet_mask
            ships_raw_b = planet_ships_raw
        else:
            obs_my = obs.my_planet_mask
            obs_pmask = obs.planet_mask
            ships_raw_b = planet_ships_raw

        K = self.max_fleets_per_turn

        # f26: shared signals for pair features.
        # ``target_mask`` = enemy or neutral (anything not mine, that is a real planet).
        target_mask = obs_pmask & jnp.logical_not(obs_my)
        ships_my_f = jnp.where(obs_my, ships_raw_b, jnp.int32(0)).astype(jnp.float32)
        total_init = ships_my_f.sum(axis=-1)  # (...,)

        if self.zero_sum_value and opp_obs is not None:
            opp_enc = self._encode(opp_obs)
            value = self.value_head(
                global_emb, planet_emb, obs.planet_mask, fleet_emb, obs.fleet_mask,
                opp_global_emb=opp_enc["global_emb"],
                opp_planet_emb=opp_enc["planet_emb"],
                opp_planet_mask=opp_obs.planet_mask,
                opp_fleet_emb=opp_enc["fleet_emb"],
                opp_fleet_mask=opp_obs.fleet_mask,
            )
        else:
            value = self.value_head(
                global_emb, planet_emb, obs.planet_mask, fleet_emb, obs.fleet_mask
            )

        # We pre-compute src/dst/pct/emit logits for all K steps via a python
        # for-loop so flax can build the heads exactly once. The running
        # ``reserved`` buffer is updated each iter using the *given* action.
        reserved = jnp.zeros_like(ships_raw_b)  # (..., P)

        src_logits_list = []
        dst_logits_list = []
        pct_logits_list = []
        emit_logits_list = []

        for t in range(K):
            # Take per-step inputs along the K axis (works batched and unbatched).
            src_t = src_idx[..., t]
            dst_t = dst_idx[..., t]
            pct_t = pct_bin[..., t]

            # masked src head: my planets with remaining ships > 0
            remaining = ships_raw_b - reserved
            avail_mask = obs_my & (remaining > 0)
            fallback = jnp.logical_not(avail_mask.any(axis=-1, keepdims=True))
            eff_mask = jnp.where(fallback, obs_my, avail_mask)
            remaining_norm, reserved_norm, total_remaining_norm = _remaining_features(
                ships_raw_b, reserved, obs_my
            )
            # f26: per-step pair features (depend on reserved + chosen src/dst).
            home_init = jnp.take_along_axis(
                ships_my_f, home_idx[..., None], axis=-1
            )[..., 0]
            emit_pair_g = emit_pair_globals(
                planet_x, planet_y, ships_raw_b, obs_pmask,
                obs_my, target_mask, remaining,
                home_idx, home_init, total_init,
            )
            dst_pair, sun_block = dst_pair_features_batched(
                planet_x, planet_y, ships_raw_b, obs_pmask,
                target_mask, remaining, src_t,
            )
            flip_block = (
                dst_flip_block_mask(
                    ships_raw_b, obs_pmask, target_mask, remaining, src_t,
                )
                if self.flip_hard_mask
                else None
            )

            src_logits_t = self.src_head(planet_emb, eff_mask, remaining_norm)
            emit_worth_it = emit_pair_g[..., 0] > 0
            emit_force_stop = (
                jnp.bool_(t >= self.emit_hard_stop_min_step) & jnp.logical_not(emit_worth_it)
                if self.emit_hard_stop
                else None
            )
            emit_logits_t = self.emit_head(
                global_emb, planet_pool, jnp.int32(t), total_remaining_norm,
                pair_feats_g=emit_pair_g,
                emit_force_stop=emit_force_stop,
            )
            src_emb_t = _gather_planet_emb(planet_emb, src_t)
            dst_logits_t = self.dst_head(
                planet_emb, src_emb_t, obs_pmask, obs_my,
                src_idx=src_t, reserved_norm=reserved_norm,
                pair_feats=dst_pair, sun_block_mask=sun_block,
                flip_block_mask=flip_block,
            )
            dst_emb_t = _gather_planet_emb(planet_emb, dst_t)
            # src_remaining_norm: scalar (per leading batch) = remaining at src
            src_remaining_norm = jnp.take_along_axis(
                remaining_norm, src_t[..., None], axis=-1
            )[..., 0]
            # pct pair features: (..., 2) [min_bin_norm, pair_flip_bin5]
            garr_dst_chosen = jnp.take_along_axis(
                ships_raw_b.astype(jnp.float32), dst_t[..., None], axis=-1
            )[..., 0]
            rem_src_chosen = jnp.take_along_axis(
                remaining.astype(jnp.float32), src_t[..., None], axis=-1
            )[..., 0]
            pct_pair = pct_pair_features(garr_dst_chosen, rem_src_chosen)
            min_bin = pct_min_bin_index(garr_dst_chosen, rem_src_chosen)
            min_bin = jnp.maximum(min_bin, jnp.int32(self.min_pct_bin))
            pct_mask = pct_low_bin_mask(min_bin, self.num_pct_bins)
            pct_logits_t = self.pct_head(
                src_emb_t, dst_emb_t, global_emb, src_remaining_norm,
                pair_feats=pct_pair,
                pct_low_bin_mask=pct_mask,
            )

            src_logits_list.append(src_logits_t)
            dst_logits_list.append(dst_logits_t)
            pct_logits_list.append(pct_logits_t)
            emit_logits_list.append(emit_logits_t)

            ships_t = _ships_to_send_for_step_batched(ships_raw_b, reserved, src_t, pct_t)
            ships_eff = jnp.where(emit_mask[..., t], ships_t, jnp.int32(0))
            reserved = _scatter_add(reserved, src_t, ships_eff)

        # Stack along a new K axis.
        s_logits = jnp.stack(src_logits_list, axis=-2)  # (..., K, P)
        d_logits = jnp.stack(dst_logits_list, axis=-2)  # (..., K, P)
        p_logits = jnp.stack(pct_logits_list, axis=-2)  # (..., K, num_pct_bins)
        e_logits = jnp.stack(emit_logits_list, axis=-2)  # (..., K, 2)

        return ActorCriticOutput(
            src_logits=s_logits,
            dst_logits=d_logits,
            pct_logits=p_logits,
            emit_logits=e_logits,
            value=value,
        )

    # ---- sample (rollout): produce K-step action stochastically

    def __call__(
        self,
        obs: EncodedObs,
        rng: jnp.ndarray,
        planet_ships_raw: jnp.ndarray,
        planet_x: jnp.ndarray,
        planet_y: jnp.ndarray,
        home_idx: jnp.ndarray,
        *,
        deterministic: bool = False,
        opp_obs: EncodedObs | None = None,
    ) -> SampledMultiAction:
        """Sample a K-step multi-fleet action.

        ``planet_ships_raw`` (..., P,) is the per-planet garrison at the start
        of the turn -- needed for the running ``reserved_ships`` mask.

        ``deterministic=True`` picks argmax everywhere (use at eval time).

        Like ``evaluate``, the K-step loop is a Python ``for`` (flax doesn't
        allow ``nn.Module.__call__`` inside ``jax.lax.scan``).
        """
        enc = self._encode(obs)
        planet_emb = enc["planet_emb"]
        global_emb = enc["global_emb"]
        planet_pool = enc["planet_pool"]
        fleet_emb = enc["fleet_emb"]

        obs_my = obs.my_planet_mask
        obs_pmask = obs.planet_mask
        ships_raw = planet_ships_raw

        K = self.max_fleets_per_turn

        # f26 shared
        target_mask = obs_pmask & jnp.logical_not(obs_my)
        ships_my_f = jnp.where(obs_my, ships_raw, jnp.int32(0)).astype(jnp.float32)
        total_init = ships_my_f.sum(axis=-1)

        if self.zero_sum_value and opp_obs is not None:
            opp_enc = self._encode(opp_obs)
            value = self.value_head(
                global_emb, planet_emb, obs.planet_mask, fleet_emb, obs.fleet_mask,
                opp_global_emb=opp_enc["global_emb"],
                opp_planet_emb=opp_enc["planet_emb"],
                opp_planet_mask=opp_obs.planet_mask,
                opp_fleet_emb=opp_enc["fleet_emb"],
                opp_fleet_mask=opp_obs.fleet_mask,
            )
        else:
            value = self.value_head(
                global_emb, planet_emb, obs.planet_mask, fleet_emb, obs.fleet_mask
            )

        reserved = jnp.zeros_like(ships_raw)  # (..., P) int32
        # still_emitting is a per-batch bool; if rank=0 keep scalar.
        leading_shape = ships_raw.shape[:-1]
        still_emitting = jnp.ones(leading_shape, dtype=jnp.bool_) if leading_shape else jnp.bool_(True)

        src_list, dst_list, pct_list, emit_list = [], [], [], []
        free_list = []  # store the sampler's free_choice mask per step for evaluate parity
        sl_list, dl_list, pl_list, el_list = [], [], [], []
        se_list, de_list, pe_list, ee_list = [], [], [], []

        # Per-step rngs deterministically derived from the outer rng + t.
        for t in range(K):
            rng = jax.random.fold_in(rng, t + 1)
            r_emit, r_src, r_dst, r_pct = jax.random.split(rng, 4)

            remaining = ships_raw - reserved
            avail_mask = obs_my & (remaining > 0)
            any_avail = avail_mask.any(axis=-1, keepdims=True)
            no_options = jnp.logical_not(jnp.squeeze(any_avail, axis=-1)) if leading_shape else jnp.logical_not(any_avail[0])
            # broadcast the "no_options" fallback against the K-axis structure
            eff_mask = jnp.where(any_avail, avail_mask, obs_my)
            remaining_norm, reserved_norm, total_remaining_norm = _remaining_features(
                ships_raw, reserved, obs_my
            )
            # f26 emit pair globals (no src/dst dep -- safe to compute first).
            home_init = jnp.take_along_axis(
                ships_my_f, home_idx[..., None], axis=-1
            )[..., 0]
            emit_pair_g = emit_pair_globals(
                planet_x, planet_y, ships_raw, obs_pmask,
                obs_my, target_mask, remaining,
                home_idx, home_init, total_init,
            )
            src_logits_t = self.src_head(planet_emb, eff_mask, remaining_norm)
            emit_worth_it = emit_pair_g[..., 0] > 0
            emit_force_stop = (
                jnp.bool_(t >= self.emit_hard_stop_min_step) & jnp.logical_not(emit_worth_it)
                if self.emit_hard_stop
                else None
            )
            emit_logits_t = self.emit_head(
                global_emb, planet_pool, jnp.int32(t), total_remaining_norm,
                pair_feats_g=emit_pair_g,
                emit_force_stop=emit_force_stop,
            )

            # Sample emit (1 = continue, 0 = stop).
            # allow_hold=True: t==0 is also a free choice (model can hold/skip turn).
            # allow_hold=False (legacy): t==0 forced to emit if options exist.
            if deterministic:
                emit_pred = jnp.argmax(emit_logits_t, axis=-1).astype(jnp.int32)
            else:
                emit_pred = jax.random.categorical(r_emit, emit_logits_t).astype(jnp.int32)
            emit_pred_bool = emit_pred == 1
            force_first = bool(t == 0) and not self.allow_hold
            if force_first:
                decision = jnp.logical_not(no_options)
            else:
                decision = emit_pred_bool & jnp.logical_not(no_options)
            emit_t = decision & still_emitting

            emit_idx = emit_t.astype(jnp.int32)
            emit_logp_full, emit_ent_full = _categorical_logp_entropy(emit_logits_t, emit_idx)
            already_stopped = jnp.logical_not(still_emitting)
            free_choice = jnp.logical_not(already_stopped | jnp.bool_(force_first) | no_options)
            emit_logp = jnp.where(free_choice, emit_logp_full, jnp.float32(0.0))
            emit_entropy = jnp.where(free_choice, emit_ent_full, jnp.float32(0.0))

            if deterministic:
                src_t = jnp.argmax(src_logits_t, axis=-1).astype(jnp.int32)
            else:
                src_t = jax.random.categorical(r_src, src_logits_t).astype(jnp.int32)
            src_logp_full, src_ent_full = _categorical_logp_entropy(src_logits_t, src_t)
            src_logp = jnp.where(emit_t, src_logp_full, jnp.float32(0.0))
            src_entropy = jnp.where(emit_t, src_ent_full, jnp.float32(0.0))

            src_emb_t = _gather_planet_emb(planet_emb, src_t)
            # f26: dst pair feats depend on chosen src.
            dst_pair, sun_block = dst_pair_features_batched(
                planet_x, planet_y, ships_raw, obs_pmask,
                target_mask, remaining, src_t,
            )
            flip_block = (
                dst_flip_block_mask(
                    ships_raw, obs_pmask, target_mask, remaining, src_t,
                )
                if self.flip_hard_mask
                else None
            )
            dst_logits_t = self.dst_head(
                planet_emb, src_emb_t, obs_pmask, obs_my,
                src_idx=src_t, reserved_norm=reserved_norm,
                pair_feats=dst_pair, sun_block_mask=sun_block,
                flip_block_mask=flip_block,
            )
            if deterministic:
                dst_t = jnp.argmax(dst_logits_t, axis=-1).astype(jnp.int32)
            else:
                dst_t = jax.random.categorical(r_dst, dst_logits_t).astype(jnp.int32)
            dst_logp_full, dst_ent_full = _categorical_logp_entropy(dst_logits_t, dst_t)
            dst_logp = jnp.where(emit_t, dst_logp_full, jnp.float32(0.0))
            dst_entropy = jnp.where(emit_t, dst_ent_full, jnp.float32(0.0))

            dst_emb_t = _gather_planet_emb(planet_emb, dst_t)
            src_remaining_norm = jnp.take_along_axis(
                remaining_norm, src_t[..., None], axis=-1
            )[..., 0]
            # f26: pct pair feats use the chosen (src, dst).
            garr_dst_chosen = jnp.take_along_axis(
                ships_raw.astype(jnp.float32), dst_t[..., None], axis=-1
            )[..., 0]
            rem_src_chosen = jnp.take_along_axis(
                remaining.astype(jnp.float32), src_t[..., None], axis=-1
            )[..., 0]
            pct_pair = pct_pair_features(garr_dst_chosen, rem_src_chosen)
            min_bin = pct_min_bin_index(garr_dst_chosen, rem_src_chosen)
            min_bin = jnp.maximum(min_bin, jnp.int32(self.min_pct_bin))
            pct_mask = pct_low_bin_mask(min_bin, self.num_pct_bins)
            pct_logits_t = self.pct_head(
                src_emb_t, dst_emb_t, global_emb, src_remaining_norm,
                pair_feats=pct_pair,
                pct_low_bin_mask=pct_mask,
            )
            if deterministic:
                pct_t = jnp.argmax(pct_logits_t, axis=-1).astype(jnp.int32)
            else:
                pct_t = jax.random.categorical(r_pct, pct_logits_t).astype(jnp.int32)
            pct_logp_full, pct_ent_full = _categorical_logp_entropy(pct_logits_t, pct_t)
            pct_logp = jnp.where(emit_t, pct_logp_full, jnp.float32(0.0))
            pct_entropy = jnp.where(emit_t, pct_ent_full, jnp.float32(0.0))

            ships_t = _ships_to_send_for_step_batched(ships_raw, reserved, src_t, pct_t)
            ships_eff = jnp.where(emit_t, ships_t, jnp.int32(0))
            reserved = _scatter_add(reserved, src_t, ships_eff)
            still_emitting = still_emitting & emit_t

            src_list.append(src_t)
            dst_list.append(dst_t)
            pct_list.append(pct_t)
            emit_list.append(emit_t)
            free_list.append(free_choice)
            sl_list.append(src_logp); dl_list.append(dst_logp); pl_list.append(pct_logp); el_list.append(emit_logp)
            se_list.append(src_entropy); de_list.append(dst_entropy); pe_list.append(pct_entropy); ee_list.append(emit_entropy)

        stack = lambda lst: jnp.stack(lst, axis=-1)
        return SampledMultiAction(
            src_idx=stack(src_list),
            dst_idx=stack(dst_list),
            pct_bin=stack(pct_list),
            emit_mask=stack(emit_list),
            emit_free_mask=stack(free_list),
            src_logp=stack(sl_list),
            dst_logp=stack(dl_list),
            pct_logp=stack(pl_list),
            emit_logp=stack(el_list),
            src_entropy=stack(se_list),
            dst_entropy=stack(de_list),
            pct_entropy=stack(pe_list),
            emit_entropy=stack(ee_list),
            value=value,
        )
