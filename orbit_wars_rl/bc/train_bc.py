"""Behavior-clone the v20 heuristic into the current (v23) ActorCritic.

Usage
-----
  python -m orbit_wars_rl.bc.train_bc \
      --data data/bc_v20_self.npz \
      --epochs 5 --batch-size 256 --lr 3e-4 \
      --out ckpt_bc_v20/ckpt_final.pkl

Network defaults match the v23 Route A spec (d_model=256, n_layers=4,
n_heads=8, ff_dim=1024, allow_hold=1, zero_sum_value=1) so the resulting
ckpt can warm-start v24 PPO with zero shape adaptation. Inference-time
masks (emit_hard_stop / min_pct_bin) stay OFF during BC: we clone v20's
raw distribution; the masks are static flags that don't touch parameter
shapes and can be re-enabled at PPO/export time.

What this does
--------------
1. Loads the .npz produced by ``collect_data.py``.
2. Splits 90/10 into train/val (deterministic shuffle).
3. Initialises an ActorCritic with the current multi-action architecture
   (heads accept reserved-aware pair inputs).
4. Trains by minimising a masked cross-entropy over each head:

       L = w_src * CE(src) + w_dst * CE(dst) + w_pct * CE(pct)
           + w_emit * CE(emit)

   where:
     - CE(src/dst/pct) is summed only over steps where ``emit_t = True``
       (no fleet -> no src/dst/pct labels to learn)
     - CE(emit) is summed over steps where ``loss_mask & emit_free`` is True
       (the forced-first-emit decision is excluded; v20-issued-zero turns
       contribute exactly one "stop" example at step 0)
     - ``w_emit = 2.0`` to overweight stop decisions (v20 turns are 64%
       no-emit; raw CE would underweight this signal)

5. Computes per-head accuracy on val every epoch:
     - acc_src/dst/pct: argmax over masked-only steps (where emit=True)
     - acc_emit: argmax matches target over (loss_mask & emit_free) steps
     - emit_calibration: fraction of val turns where greedy emit-count
       matches v20's exact emit-count

6. Saves a ckpt in the same pickled-dict format as PPO ckpts, so
   ``export_submission.py`` accepts it.

Decision: we use ``ActorCritic.evaluate()`` (not ``__call__``). The eval
path takes pre-chosen (src, dst, pct, emit) tuples and recomputes the K
logits with the correct ``reserved`` running buffer. That's exactly what
BC needs: we hand it v20's choices and minimize CE on those choices.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Any

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from orbit_wars_rl.env import constants
from orbit_wars_rl.features.encode import EncodedObs
from orbit_wars_rl.net.model import ActorCritic


_K = constants.MAX_FLEETS_PER_TURN


# --------------------------------------------------------------------
# Data
# --------------------------------------------------------------------

def _load_dataset(path: str) -> dict[str, np.ndarray]:
    """Load a .npz dataset into a dict of numpy arrays."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _split_train_val(data: dict[str, np.ndarray], val_frac: float = 0.1,
                     seed: int = 0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Deterministic shuffle then split."""
    n = data["src"].shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return (
        {k: v[train_idx] for k, v in data.items()},
        {k: v[val_idx] for k, v in data.items()},
    )


def _batch_iter(data: dict[str, np.ndarray], batch_size: int, rng_seed: int):
    """Yield contiguous shuffled batches; final partial batch dropped."""
    n = data["src"].shape[0]
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(n)
    n_batches = n // batch_size
    for b in range(n_batches):
        idx = perm[b * batch_size:(b + 1) * batch_size]
        yield {k: v[idx] for k, v in data.items()}


def _to_encoded_obs(batch: dict[str, np.ndarray]) -> EncodedObs:
    return EncodedObs(
        planet_feats=jnp.asarray(batch["planet_feats"]),
        planet_mask=jnp.asarray(batch["planet_mask"]),
        fleet_feats=jnp.asarray(batch["fleet_feats"]),
        fleet_mask=jnp.asarray(batch["fleet_mask"]),
        global_feats=jnp.asarray(batch["global_feats"]),
        my_planet_mask=jnp.asarray(batch["my_planet_mask"]),
        enemy_planet_mask=jnp.asarray(batch["enemy_planet_mask"]),
        neutral_planet_mask=jnp.asarray(batch["neutral_planet_mask"]),
    )


# --------------------------------------------------------------------
# Loss / metrics
# --------------------------------------------------------------------

def _masked_ce(logits: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy averaged over True entries of ``mask`` (per-batch sum / count).

    logits : (B, K, C)
    target : (B, K)  int
    mask   : (B, K)  bool

    Returns scalar.
    """
    target_logit = jnp.take_along_axis(logits, target[..., None], axis=-1)[..., 0]
    effective_mask = mask & (target_logit > jnp.float32(-1.0e8))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(target, logits.shape[-1], dtype=logits.dtype)
    per_step_ce = -(log_probs * one_hot).sum(axis=-1)  # (B, K)
    masked = jnp.where(effective_mask, per_step_ce, jnp.float32(0.0))
    n = jnp.maximum(jnp.float32(effective_mask.sum()), jnp.float32(1.0))
    return masked.sum() / n


def _masked_weighted_ce(
    logits: jnp.ndarray,
    target: jnp.ndarray,
    mask: jnp.ndarray,
    class_weights: jnp.ndarray,
) -> jnp.ndarray:
    target_logit = jnp.take_along_axis(logits, target[..., None], axis=-1)[..., 0]
    effective_mask = mask & (target_logit > jnp.float32(-1.0e8))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(target, logits.shape[-1], dtype=logits.dtype)
    per_step_ce = -(log_probs * one_hot).sum(axis=-1)
    weights = jnp.take(class_weights.astype(logits.dtype), target)
    weighted = per_step_ce * weights
    masked = jnp.where(effective_mask, weighted, jnp.float32(0.0))
    denom = jnp.where(effective_mask, weights, jnp.float32(0.0)).sum()
    return masked.sum() / jnp.maximum(denom, jnp.float32(1.0))


def _masked_acc(logits: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    target_logit = jnp.take_along_axis(logits, target[..., None], axis=-1)[..., 0]
    effective_mask = mask & (target_logit > jnp.float32(-1.0e8))
    pred = jnp.argmax(logits, axis=-1)
    hit = (pred == target).astype(jnp.float32)
    n = jnp.maximum(jnp.float32(effective_mask.sum()), jnp.float32(1.0))
    return (jnp.where(effective_mask, hit, jnp.float32(0.0))).sum() / n


def _bc_loss_and_metrics(
    params: dict,
    apply_fn,
    batch: dict[str, jnp.ndarray],
    w_emit: float,
    emit_pos_weight: float,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Compute BC loss + per-head accuracy on a single batch.

    Calls ``ActorCritic.evaluate`` with the same raw geometry fields PPO uses.
    """
    enc = _to_encoded_obs(batch)
    src = jnp.asarray(batch["src"])
    dst = jnp.asarray(batch["dst"])
    pct = jnp.asarray(batch["pct"])
    emit = jnp.asarray(batch["emit"])
    loss_mask = jnp.asarray(batch["loss_mask"])
    emit_free = jnp.asarray(batch["emit_free"])
    ships_raw = jnp.asarray(batch["planet_ships_raw"])
    # Current pair features depend on true geometry and the player's home slot.
    px = jnp.asarray(batch["planet_x_raw"])
    py = jnp.asarray(batch["planet_y_raw"])
    home = jnp.asarray(batch["home_idx_raw"])
    # v23: orbital fields for ETA-lead pair features. Legacy datasets miss
    # them; fall back to "nothing orbits" which degrades lead to plain dist.
    leading = ships_raw.shape[:-1]
    orbit_phase = jnp.asarray(
        batch.get("planet_orbit_phase_raw", jnp.zeros_like(px)))
    orbit_radius = jnp.asarray(
        batch.get("planet_orbit_radius_raw", jnp.zeros_like(px)))
    is_orbiting = jnp.asarray(
        batch.get("planet_is_orbiting_raw",
                  jnp.zeros(px.shape, dtype=jnp.bool_)))
    ang_vel = jnp.asarray(
        batch.get("angular_velocity_raw", jnp.zeros(leading, dtype=jnp.float32)))

    out = apply_fn(
        {"params": params["params"]},
        enc, src, dst, pct, emit, ships_raw, px, py, home,
        orbit_phase, orbit_radius, is_orbiting, ang_vel,
        method=ActorCritic.evaluate,
    )

    # src / dst / pct supervised only where emit=True (fleet was actually issued)
    emit_mask = emit & loss_mask
    src_ce = _masked_ce(out.src_logits, src, emit_mask)
    dst_ce = _masked_ce(out.dst_logits, dst, emit_mask)
    pct_ce = _masked_ce(out.pct_logits, pct, emit_mask)

    # emit head supervised on loss_mask & emit_free
    emit_target = emit.astype(jnp.int32)
    emit_supervise = loss_mask & emit_free
    emit_class_weights = jnp.asarray([1.0, emit_pos_weight], dtype=jnp.float32)
    emit_ce = _masked_weighted_ce(
        out.emit_logits, emit_target, emit_supervise, emit_class_weights
    )

    total = src_ce + dst_ce + pct_ce + w_emit * emit_ce

    # accuracies for monitoring
    src_acc = _masked_acc(out.src_logits, src, emit_mask)
    dst_acc = _masked_acc(out.dst_logits, dst, emit_mask)
    pct_acc = _masked_acc(out.pct_logits, pct, emit_mask)
    emit_acc = _masked_acc(out.emit_logits, emit_target, emit_supervise)

    metrics = {
        "loss/total": total,
        "loss/src": src_ce,
        "loss/dst": dst_ce,
        "loss/pct": pct_ce,
        "loss/emit": emit_ce,
        "acc/src": src_acc,
        "acc/dst": dst_acc,
        "acc/pct": pct_acc,
        "acc/emit": emit_acc,
        "stats/n_emit_steps": jnp.float32(emit_mask.sum()),
        "stats/n_supervised_emit": jnp.float32(emit_supervise.sum()),
    }
    return total, metrics


# --------------------------------------------------------------------
# Train
# --------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help=".npz dataset produced by collect_data.py")
    parser.add_argument("--out", type=str, default="ckpt_bc_v0/ckpt_final.pkl")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--w-emit", type=float, default=2.0,
                        help="loss weight for emit head (v20 is 64% stop)")
    parser.add_argument("--emit-pos-weight", type=float, default=1.0,
                        help="class weight for emit=True continuation labels")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--emit-hard-stop", action="store_true",
                        help="train/evaluate with the inference hard stop mask")
    parser.add_argument("--flip-hard-mask", action="store_true",
                        help="train/evaluate with the dst flip hard mask")
    parser.add_argument("--allow-hold", type=int, choices=(0, 1), default=1,
                        help="t=0 emit is a free choice (v23 spec; required "
                             "to clone v20's ~60% hold rate)")
    parser.add_argument("--zero-sum-value", type=int, choices=(0, 1), default=1,
                        help="use ZeroSumValueHead (v23 spec); BC never "
                             "trains value but the param tree must match v24")
    parser.add_argument("--min-pct-bin", type=int, default=0,
                        help="pct low-bin hard mask during BC (default off)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=50,
                        help="print train metrics every N steps")
    args = parser.parse_args()

    print(f"loading dataset: {args.data}")
    data = _load_dataset(args.data)
    required = ["planet_x_raw", "planet_y_raw", "home_idx_raw"]
    missing = [k for k in required if k not in data]
    if missing:
        raise SystemExit(
            "dataset is missing required raw geometry fields: "
            + ", ".join(missing) + ". Recollect with the current collect_data.py."
        )
    orbit_fields = ["planet_orbit_phase_raw", "planet_orbit_radius_raw",
                    "planet_is_orbiting_raw", "angular_velocity_raw"]
    missing_orbit = [k for k in orbit_fields if k not in data]
    if missing_orbit:
        print(
            "WARNING: dataset is missing orbital fields: "
            + ", ".join(missing_orbit)
            + ". ETA-lead degrades to plain distance; recollect for v23 runs.",
            flush=True,
        )
    n = data["src"].shape[0]
    print(f"  {n} samples, shape: src={data['src'].shape}, "
          f"planet_feats={data['planet_feats'].shape}")

    train_data, val_data = _split_train_val(data, args.val_frac, seed=args.seed)
    print(f"  train: {train_data['src'].shape[0]}  val: {val_data['src'].shape[0]}")

    rng = jax.random.PRNGKey(args.seed)
    rng_init, rng_train = jax.random.split(rng, 2)

    model = ActorCritic(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        max_fleets_per_turn=_K,
        emit_hard_stop=args.emit_hard_stop,
        flip_hard_mask=args.flip_hard_mask,
        allow_hold=bool(args.allow_hold),
        zero_sum_value=bool(args.zero_sum_value),
        min_pct_bin=args.min_pct_bin,
    )

    # Build a dummy single sample for init
    print("initialising model...")
    dummy_enc = EncodedObs(
        planet_feats=jnp.asarray(train_data["planet_feats"][0]),
        planet_mask=jnp.asarray(train_data["planet_mask"][0]),
        fleet_feats=jnp.asarray(train_data["fleet_feats"][0]),
        fleet_mask=jnp.asarray(train_data["fleet_mask"][0]),
        global_feats=jnp.asarray(train_data["global_feats"][0]),
        my_planet_mask=jnp.asarray(train_data["my_planet_mask"][0]),
        enemy_planet_mask=jnp.asarray(train_data["enemy_planet_mask"][0]),
        neutral_planet_mask=jnp.asarray(train_data["neutral_planet_mask"][0]),
    )
    dummy_ships = jnp.asarray(train_data["planet_ships_raw"][0])
    dummy_px = jnp.asarray(train_data["planet_x_raw"][0])
    dummy_py = jnp.asarray(train_data["planet_y_raw"][0])
    dummy_home = jnp.asarray(train_data["home_idx_raw"][0])
    params = model.init(rng_init, dummy_enc, rng_init, dummy_ships,
                        dummy_px, dummy_py, dummy_home)

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  model params: {n_params:,}")

    # Optimizer (no schedule; BC is short, fixed lr is fine).
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.lr),
    )
    opt_state = optimizer.init(params)

    # Compiled train step. We jit per-batch grad.
    def loss_fn(params, batch):
        loss, metrics = _bc_loss_and_metrics(
            params, model.apply, batch, args.w_emit, args.emit_pos_weight
        )
        return loss, metrics

    @jax.jit
    def train_step(params, opt_state, batch):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        # report grad norm for diagnostics
        gnorm = optax.global_norm(grads)
        metrics = {**metrics, "stats/grad_norm": gnorm}
        return new_params, new_opt_state, loss, metrics

    @jax.jit
    def eval_step(params, batch):
        _, metrics = _bc_loss_and_metrics(
            params, model.apply, batch, args.w_emit, args.emit_pos_weight
        )
        return metrics

    # Move data to jnp once (small enough for local val; train slices each iter)
    val_batch_jnp = {k: jnp.asarray(v) for k, v in val_data.items()}

    def eval_loop(params, val_batch_size: int = 256) -> dict[str, float]:
        out_acc = {}
        n = val_batch_jnp["src"].shape[0]
        # average over batches
        sums = {}
        counts = 0
        for s in range(0, n, val_batch_size):
            e = min(s + val_batch_size, n)
            batch = {k: v[s:e] for k, v in val_batch_jnp.items()}
            m = eval_step(params, batch)
            counts += 1
            for k, v in m.items():
                sums[k] = sums.get(k, 0.0) + float(v)
        for k, v in sums.items():
            out_acc[k] = v / max(counts, 1)
        return out_acc

    # Pre-eval at epoch 0 (random init) for baseline.
    print("\nepoch 0 (random init):")
    eval_metrics = eval_loop(params)
    for k in ["loss/total", "loss/emit", "acc/src", "acc/dst", "acc/pct", "acc/emit"]:
        print(f"  {k}: {eval_metrics[k]:.4f}")

    n_train = train_data["src"].shape[0]
    step = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_losses = []
        for batch in _batch_iter(train_data, args.batch_size,
                                 rng_seed=args.seed * 1000 + epoch):
            batch_jnp = {k: jnp.asarray(v) for k, v in batch.items()}
            params, opt_state, loss, metrics = train_step(params, opt_state, batch_jnp)
            train_losses.append(float(loss))
            if step % args.log_every == 0:
                print(
                    f"  step {step} | loss {float(loss):.4f}  "
                    f"src {float(metrics['loss/src']):.3f}  "
                    f"dst {float(metrics['loss/dst']):.3f}  "
                    f"pct {float(metrics['loss/pct']):.3f}  "
                    f"emit {float(metrics['loss/emit']):.3f}  "
                    f"gn {float(metrics['stats/grad_norm']):.3f}"
                )
            step += 1

        # epoch-end val
        eval_metrics = eval_loop(params)
        ep_dt = time.time() - epoch_start
        print(
            f"\nepoch {epoch+1}/{args.epochs} ({ep_dt:.1f}s) "
            f"train_loss {np.mean(train_losses):.4f}  "
            f"| val acc src {eval_metrics['acc/src']:.3f}  "
            f"dst {eval_metrics['acc/dst']:.3f}  "
            f"pct {eval_metrics['acc/pct']:.3f}  "
            f"emit {eval_metrics['acc/emit']:.3f}\n"
        )

    total_dt = time.time() - t_start
    print(f"\ntraining done in {total_dt:.1f}s ({args.epochs} epochs, {step} steps)")

    # Save in the same format as PPO ckpts so export_submission and the PPO
    # runner's resume_ckpt both accept it. The .meta.json sidecar carries
    # the export flags quick_replay.sh applies when building a submission.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "bc": {"data": args.data, "epochs": args.epochs, "lr": args.lr},
        "export": {
            "allow_hold": bool(args.allow_hold),
            "force_emit_worth_it": False,
            "min_pct_bin": int(args.min_pct_bin),
            "emit_hard_stop": bool(args.emit_hard_stop),
            "emit_hard_stop_min_step": 0,
            "flip_hard_mask": bool(args.flip_hard_mask),
        },
    }
    payload = dict(
        params=jax.tree_util.tree_map(lambda x: jnp.asarray(x), params),
        opt_state=jax.tree_util.tree_map(
            lambda x: jnp.asarray(x) if hasattr(x, "shape") else x, opt_state,
        ),
        step=step,
        meta=meta,
    )
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    with open(str(out).replace(".pkl", ".meta.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2)
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # Final eval summary
    final = eval_loop(params)
    print("\nFINAL val metrics:")
    for k in ["loss/total", "loss/src", "loss/dst", "loss/pct", "loss/emit",
              "acc/src", "acc/dst", "acc/pct", "acc/emit"]:
        print(f"  {k}: {final[k]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
