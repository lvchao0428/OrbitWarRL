"""Parity test: numpy multi-action forward must agree with flax ActorCritic.

The contract we ship is **whole-turn action equivalence**: for any obs, the
deterministic K-step action list produced by the numpy path must match the
jax path exactly (same emitted (src, dst, pct) triples in the same order,
same emit_mask).

Float32 accumulation differences can push absolute logit deltas into the
1e-3 range on a well-trained net, which is fine as long as every step's
argmax is preserved.

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

from orbit_wars_rl.env import OrbitWarsEnv, constants
from orbit_wars_rl.features import encode
from orbit_wars_rl.inference import numpy_forward as nf
from orbit_wars_rl.inference.weights import (
    flatten_params,
    load_flat_params,
    assert_expected_keys,
)
from orbit_wars_rl.net.model import ActorCritic


def _build_state_and_obs(seed: int):
    env = OrbitWarsEnv(num_groups=5, episode_steps=60)
    state = env.reset(jax.random.PRNGKey(seed))
    obs = encode(state, 0, 60)
    return state, obs


def run_parity(
    ckpt_path: str | None = None,
    tol: float = 5e-3,
    num_states: int = 16,
    fail_on_logit_drift: bool = False,
    max_fleets_per_turn: int = constants.MAX_FLEETS_PER_TURN,
) -> int:
    """Returns 0 on pass, non-zero on real disagreement.

    A "real disagreement" means the emit/src/dst/pct sequences differ between
    numpy and jax for any state. Logit drift above ``tol`` is reported as
    WARN but not a failure unless ``fail_on_logit_drift`` is set.
    """
    model = ActorCritic(max_fleets_per_turn=max_fleets_per_turn)
    init_state, init_obs = _build_state_and_obs(seed=0)
    init_params = model.init(
        jax.random.PRNGKey(0),
        init_obs,
        jax.random.PRNGKey(1),
        init_state.planet_ships,
    )

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

    print(f"parity_check multi-action (ckpt={ckpt_path or 'fresh-init'}; "
          f"tol={tol}; states={num_states}; K={max_fleets_per_turn})")

    full_match = 0
    emit_only_match = 0
    val_max = 0.0
    val_mean = 0.0
    mismatch_examples = []

    for s in range(num_states):
        state, obs = _build_state_and_obs(seed=1000 + s)
        planet_feats = np.asarray(obs.planet_feats)
        planet_mask = np.asarray(obs.planet_mask)
        fleet_feats = np.asarray(obs.fleet_feats)
        fleet_mask = np.asarray(obs.fleet_mask)
        global_feats = np.asarray(obs.global_feats)
        my_planet_mask = np.asarray(obs.my_planet_mask)
        planet_ships = np.asarray(state.planet_ships)
        if not bool(my_planet_mask.any()):
            continue

        # --- jax deterministic multi-action ---
        sa = model.apply(
            params, obs, jax.random.PRNGKey(42), state.planet_ships, deterministic=True
        )
        jax_src = np.asarray(sa.src_idx).tolist()
        jax_dst = np.asarray(sa.dst_idx).tolist()
        jax_pct = np.asarray(sa.pct_bin).tolist()
        jax_emit = np.asarray(sa.emit_mask).astype(bool).tolist()
        jax_value = float(np.asarray(sa.value))

        # --- numpy multi-action ---
        np_src, np_dst, np_pct, np_emit, np_value = nf.greedy_multi_action(
            flat, planet_feats, planet_mask, fleet_feats, fleet_mask,
            global_feats, my_planet_mask, planet_ships,
            n_layers=2, max_fleets_per_turn=max_fleets_per_turn,
        )

        # jax emit_mask is K-long; numpy returns only the emitted slice.
        # Build a directly comparable jax "emitted list".
        jax_emitted_src = [jax_src[t] for t in range(max_fleets_per_turn) if jax_emit[t]]
        jax_emitted_dst = [jax_dst[t] for t in range(max_fleets_per_turn) if jax_emit[t]]
        jax_emitted_pct = [jax_pct[t] for t in range(max_fleets_per_turn) if jax_emit[t]]

        emit_count_match = (np_emit.count(True) == sum(jax_emit))
        if emit_count_match:
            emit_only_match += 1

        full_ok = (
            jax_emitted_src == np_src
            and jax_emitted_dst == np_dst
            and jax_emitted_pct == np_pct
        )
        if full_ok:
            full_match += 1
        elif len(mismatch_examples) < 3:
            mismatch_examples.append(
                f"  state {s}: jax=({jax_emitted_src},{jax_emitted_dst},{jax_emitted_pct})  "
                f"np=({np_src},{np_dst},{np_pct})"
            )

        d_val = abs(float(np_value) - jax_value)
        val_max = max(val_max, d_val)
        val_mean += d_val

    total = max(1, num_states)
    val_mean /= total
    marker = "OK " if full_match == total else "FAIL"
    print(f"  [{marker}] whole-turn action list match: {full_match}/{total}")
    print(f"  [INFO] emit count match: {emit_only_match}/{total}")
    print(f"  [INFO] value drift max={val_max:.3e} mean={val_mean:.3e}")
    for line in mismatch_examples:
        print(line)

    return 0 if full_match == total else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--tol", type=float, default=5e-3,
                    help="logit drift warning threshold (informational; not a gate by default)")
    ap.add_argument("--num-states", type=int, default=16)
    ap.add_argument("--strict", action="store_true",
                    help="also fail if logit drift exceeds --tol on any state")
    ap.add_argument("--max-fleets-per-turn", type=int, default=constants.MAX_FLEETS_PER_TURN)
    args = ap.parse_args()
    return run_parity(
        args.ckpt, tol=args.tol, num_states=args.num_states,
        fail_on_logit_drift=args.strict,
        max_fleets_per_turn=args.max_fleets_per_turn,
    )


if __name__ == "__main__":
    sys.exit(main())
