"""Parity test: numpy forward must match flax ActorCritic to ~1e-4.

Usage:
    python -m orbit_wars_rl.inference.test_parity
    python -m orbit_wars_rl.inference.test_parity --ckpt ckpt_mvp/ckpt_000049.pkl

Without ``--ckpt`` it uses freshly-initialized flax params (so no training run
is required to verify the implementation is correct). With ``--ckpt`` it
verifies a real checkpoint round-trips through pickle → numpy and still
produces matching logits.
"""

from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np

from orbit_wars_rl.env import OrbitWarsEnv
from orbit_wars_rl.features import encode
from orbit_wars_rl.inference import numpy_forward as nf
from orbit_wars_rl.inference.weights import (
    flatten_params,
    load_flat_params,
    assert_expected_keys,
)
from orbit_wars_rl.net.model import ActorCritic


def _build_obs(seed: int = 7) -> tuple:
    env = OrbitWarsEnv(num_groups=5, episode_steps=60)
    state = env.reset(jax.random.PRNGKey(seed))
    obs = encode(state, 0, 60)
    return state, obs


def _flax_logits(model: ActorCritic, params, obs, src_idx_jax: jnp.ndarray, dst_idx_jax: jnp.ndarray):
    """Call the deterministic eval path of ActorCritic; return logits + value."""
    out = model.apply(params, obs, src_idx_jax, dst_idx_jax, method=ActorCritic.evaluate)
    return (
        np.asarray(out.src_logits),
        np.asarray(out.dst_logits),
        np.asarray(out.pct_logits),
        float(np.asarray(out.value)),
    )


def run_parity(ckpt_path: str | None = None, tol: float = 1e-3) -> int:
    state, obs = _build_obs()
    model = ActorCritic()
    init_params = model.init(jax.random.PRNGKey(0), obs, jax.random.PRNGKey(1))

    if ckpt_path:
        flat = load_flat_params(ckpt_path)
        # Build a jax-side params tree from the same flat dict by reading the existing tree's leaves;
        # easiest is: just reuse pickle file's params directly.
        import pickle
        with open(ckpt_path, "rb") as f:
            payload = pickle.load(f)
        params = payload["params"]
    else:
        flat = flatten_params(init_params)
        params = init_params

    assert_expected_keys(flat, n_layers=2)

    planet_feats = np.asarray(obs.planet_feats)
    planet_mask = np.asarray(obs.planet_mask)
    fleet_feats = np.asarray(obs.fleet_feats)
    fleet_mask = np.asarray(obs.fleet_mask)
    global_feats = np.asarray(obs.global_feats)
    my_planet_mask = np.asarray(obs.my_planet_mask)

    g_emb_np, p_emb_np, _f_emb_np, p_pool_np, f_pool_np = nf.encode_tokens(
        flat, planet_feats, planet_mask, fleet_feats, fleet_mask, global_feats, n_layers=2,
    )
    src_logits_np = nf.src_head(flat, p_emb_np, my_planet_mask)
    src_idx_np = int(np.argmax(src_logits_np))
    dst_logits_np = nf.dst_head(flat, p_emb_np, p_emb_np[src_idx_np], planet_mask, my_planet_mask)
    dst_idx_np = int(np.argmax(dst_logits_np))
    pct_logits_np = nf.pct_head(flat, p_emb_np[src_idx_np], p_emb_np[dst_idx_np], g_emb_np)
    value_np = nf.value_head(flat, g_emb_np, p_pool_np, f_pool_np)

    src_idx_jax = jnp.int32(src_idx_np)
    dst_idx_jax = jnp.int32(dst_idx_np)
    s_jax, d_jax, p_jax, v_jax = _flax_logits(model, params, obs, src_idx_jax, dst_idx_jax)

    failed = False

    def _report(name: str, a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> bool:
        if mask is not None:
            diff = np.where(mask, np.abs(a - b), 0.0)
        else:
            diff = np.abs(a - b)
        mx = float(diff.max())
        mean = float(diff.mean())
        ok = mx < tol
        marker = "OK " if ok else "FAIL"
        print(f"  [{marker}] {name:<14} max={mx:.3e} mean={mean:.3e}")
        return ok

    print(f"parity_check (ckpt={ckpt_path or 'fresh-init'}; tol={tol})")
    failed |= not _report("src_logits", src_logits_np, s_jax, mask=my_planet_mask)
    dst_valid = planet_mask & ~my_planet_mask
    failed |= not _report("dst_logits", dst_logits_np, d_jax, mask=dst_valid)
    failed |= not _report("pct_logits", pct_logits_np, p_jax)
    failed |= not _report("value     ", np.array([value_np], dtype=np.float32), np.array([v_jax], dtype=np.float32))

    print(f"  src_idx jax={int(jnp.argmax(s_jax))}  np={src_idx_np}")
    print(f"  dst_idx jax={int(jnp.argmax(d_jax))}  np={dst_idx_np}")
    print(f"  pct_idx jax={int(jnp.argmax(p_jax))}  np={int(np.argmax(pct_logits_np))}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    return run_parity(args.ckpt, tol=args.tol)


if __name__ == "__main__":
    sys.exit(main())
