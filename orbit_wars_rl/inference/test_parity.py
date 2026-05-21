"""Parity test: numpy forward must agree with flax ActorCritic.

The contract we ship is **argmax-equivalence**: for any obs, the deterministic
action ``(src_idx, dst_idx, pct_bin)`` chosen by the numpy path must match the
jax path. Float32 accumulation differences can push absolute logit deltas into
the 1e-3 range on a well-trained net, which is fine as long as argmax is
preserved.

By default we sample 16 random states and:

1. Report max/mean logit deltas across all states (informational, not a gate).
2. Pass iff argmax agrees on every state (the only real gate).

Usage:
    python -m orbit_wars_rl.inference.test_parity
    python -m orbit_wars_rl.inference.test_parity --ckpt ckpt_mvp/ckpt_000049.pkl
    python -m orbit_wars_rl.inference.test_parity --num-states 32 --tol 5e-3
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


def _build_obs(seed: int):
    env = OrbitWarsEnv(num_groups=5, episode_steps=60)
    state = env.reset(jax.random.PRNGKey(seed))
    obs = encode(state, 0, 60)
    return obs


def _flax_logits(model, params, obs, src_idx_jax, dst_idx_jax):
    out = model.apply(params, obs, src_idx_jax, dst_idx_jax, method=ActorCritic.evaluate)
    return (
        np.asarray(out.src_logits),
        np.asarray(out.dst_logits),
        np.asarray(out.pct_logits),
        float(np.asarray(out.value)),
    )


def _argmax_eq(logits_a: np.ndarray, logits_b: np.ndarray) -> bool:
    return int(np.argmax(logits_a)) == int(np.argmax(logits_b))


def run_parity(
    ckpt_path: str | None = None,
    tol: float = 1e-3,
    num_states: int = 16,
    fail_on_logit_drift: bool = False,
) -> int:
    """Returns 0 on pass, non-zero on real disagreement.

    A "real disagreement" means argmax differs on any of the sampled states; a
    numeric drift above ``tol`` is reported as WARN but not a failure unless
    ``fail_on_logit_drift`` is set.
    """
    model = ActorCritic()
    init_obs = _build_obs(seed=0)
    init_params = model.init(jax.random.PRNGKey(0), init_obs, jax.random.PRNGKey(1))

    if ckpt_path:
        flat = load_flat_params(ckpt_path)
        import pickle
        with open(ckpt_path, "rb") as f:
            payload = pickle.load(f)
        params = payload["params"]
    else:
        flat = flatten_params(init_params)
        params = init_params

    assert_expected_keys(flat, n_layers=2)

    print(f"parity_check (ckpt={ckpt_path or 'fresh-init'}; tol={tol}; states={num_states})")

    src_match = 0
    dst_match = 0
    pct_match = 0
    src_max = dst_max = pct_max = val_max = 0.0
    src_mean = dst_mean = pct_mean = val_mean = 0.0

    for s in range(num_states):
        obs = _build_obs(seed=1000 + s)
        planet_feats = np.asarray(obs.planet_feats)
        planet_mask = np.asarray(obs.planet_mask)
        fleet_feats = np.asarray(obs.fleet_feats)
        fleet_mask = np.asarray(obs.fleet_mask)
        global_feats = np.asarray(obs.global_feats)
        my_planet_mask = np.asarray(obs.my_planet_mask)
        if not bool(my_planet_mask.any()):
            continue  # no owned planet -> trivial; skip

        g_emb, p_emb, _, p_pool, f_pool = nf.encode_tokens(
            flat, planet_feats, planet_mask, fleet_feats, fleet_mask, global_feats, n_layers=2,
        )
        s_np = nf.src_head(flat, p_emb, my_planet_mask)
        src_np = int(np.argmax(s_np))
        d_np = nf.dst_head(flat, p_emb, p_emb[src_np], planet_mask, my_planet_mask)
        dst_np = int(np.argmax(d_np))
        p_np = nf.pct_head(flat, p_emb[src_np], p_emb[dst_np], g_emb)
        v_np = nf.value_head(flat, g_emb, p_pool, f_pool)

        s_jx, d_jx, p_jx, v_jx = _flax_logits(
            model, params, obs, jnp.int32(src_np), jnp.int32(dst_np),
        )

        src_match += int(_argmax_eq(s_np, s_jx))
        dst_match += int(_argmax_eq(d_np, d_jx))
        pct_match += int(_argmax_eq(p_np, p_jx))

        d_src = np.where(my_planet_mask, np.abs(s_np - s_jx), 0.0)
        d_dst = np.where(planet_mask & ~my_planet_mask, np.abs(d_np - d_jx), 0.0)
        d_pct = np.abs(p_np - p_jx)
        d_val = abs(v_np - v_jx)
        src_max = max(src_max, float(d_src.max())); src_mean += float(d_src.mean())
        dst_max = max(dst_max, float(d_dst.max())); dst_mean += float(d_dst.mean())
        pct_max = max(pct_max, float(d_pct.max())); pct_mean += float(d_pct.mean())
        val_max = max(val_max, d_val); val_mean += d_val

    total = max(1, num_states)
    src_mean /= total
    dst_mean /= total
    pct_mean /= total
    val_mean /= total

    def _line(name: str, mx: float, mean: float, matched: int, denom: int, head: str = "logits") -> bool:
        warn_drift = mx >= tol
        argmax_ok = matched == denom
        marker = "OK " if argmax_ok else "FAIL"
        suffix = ""
        if warn_drift and argmax_ok:
            suffix = "  WARN(drift)"
        print(f"  [{marker}] {name:<14} argmax {matched}/{denom}  drift max={mx:.3e} mean={mean:.3e}{suffix}")
        return argmax_ok

    src_ok = _line("src_logits", src_max, src_mean, src_match, total)
    dst_ok = _line("dst_logits", dst_max, dst_mean, dst_match, total)
    pct_ok = _line("pct_logits", pct_max, pct_mean, pct_match, total)
    print(f"  [INFO] value         drift max={val_max:.3e} mean={val_mean:.3e}")

    argmax_pass = src_ok and dst_ok and pct_ok
    drift_pass = max(src_max, dst_max, pct_max) < tol
    failed = (not argmax_pass) or (fail_on_logit_drift and not drift_pass)
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--tol", type=float, default=5e-3,
                    help="logit drift warning threshold (informational; not a gate by default)")
    ap.add_argument("--num-states", type=int, default=16)
    ap.add_argument("--strict", action="store_true",
                    help="also fail if logit drift exceeds --tol on any state")
    args = ap.parse_args()
    return run_parity(args.ckpt, tol=args.tol, num_states=args.num_states,
                      fail_on_logit_drift=args.strict)


if __name__ == "__main__":
    sys.exit(main())
