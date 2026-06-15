"""v30c smoke: capture_ready dims + 1 PPO update from v30b u1599."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.env import OrbitWarsEnv
from orbit_wars_rl.features.encode import encode
from orbit_wars_rl.features.pair import DST_PAIR_DIM, EMIT_PAIR_DIM
from orbit_wars_rl.net.model import ActorCritic
from orbit_wars_rl.ppo.runner import SelfPlayConfig, TrainConfig, train
from orbit_wars_rl.ppo.update import PPOConfig


def _check_dims() -> None:
    assert DST_PAIR_DIM == 7, DST_PAIR_DIM
    assert EMIT_PAIR_DIM == 12, EMIT_PAIR_DIM
    print(f"[OK ] DST_PAIR_DIM={DST_PAIR_DIM} EMIT_PAIR_DIM={EMIT_PAIR_DIM}")


def _check_eval_path() -> None:
    model = ActorCritic(d_model=64, n_layers=2, n_heads=4, ff_dim=128, flip_hard_mask=True)
    env = OrbitWarsEnv(num_groups=5, episode_steps=60)
    key = jax.random.PRNGKey(0)
    s = env.reset(key)
    enc = encode(s, player=0, episode_steps=60)
    params = model.init(
        key, enc, jax.random.PRNGKey(1), s.planet_ships, s.planet_x, s.planet_y, jnp.int32(0),
    )
    K = constants.MAX_FLEETS_PER_TURN
    src = jnp.zeros((K,), dtype=jnp.int32)
    dst = jnp.zeros((K,), dtype=jnp.int32)
    pct = jnp.zeros((K,), dtype=jnp.int32)
    emit = jnp.zeros((K,), dtype=jnp.bool_).at[0].set(True)
    hi = s.home_planet_idx[0]
    out = model.apply(
        params, enc, src, dst, pct, emit, s.planet_ships, s.planet_x,
        s.planet_y, hi, None, None, None, None, None,
        freeze_dst_attn=True, method=ActorCritic.evaluate,
    )
    assert out.dst_logits.shape[-1] == constants.MAX_PLANETS
    print("[OK ] evaluate path with v30c features")


def _check_ppo_one_update(ckpt_path: Path) -> None:
    if not ckpt_path.is_file():
        print(f"[SKIP] PPO smoke — ckpt not found: {ckpt_path}")
        return
    cfg = TrainConfig(
        num_envs=4,
        rollout_length=16,
        num_updates=1,
        episode_steps=60,
        num_groups=constants.MIN_PLANET_GROUPS,
        eval_every=0,
        ckpt_dir=str(tempfile.mkdtemp(prefix="smoke_v30c_")),
        ckpt_every=0,
        log_every=1,
        d_model=256,
        n_layers=4,
        n_heads=8,
        ff_dim=1024,
        flip_hard_mask=True,
        allow_hold=True,
        force_emit_worth_it=True,
        zero_sum_value=True,
        resume_ckpt=str(ckpt_path),
        ppo=PPOConfig(
            lr_warmup_steps=1,
            lr_decay_steps=10,
            update_epochs=1,
            num_minibatches=1,
            roi_aux_coef=0.4,
            freeze_dst_attn_updates=800,
        ),
        selfplay=SelfPlayConfig(enabled=False),
    )
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    train(cfg, log_dir=None, resume_from=str(ckpt_path))
    print(f"[OK ] v30c 1 update from {ckpt_path.name}")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    print("=== smoke_v30c ===")
    _check_dims()
    _check_eval_path()
    v30b = root / "ckpt_multi_action_v30b_econ/ckpt_001599.pkl"
    v29 = root / "ckpt_multi_action_v29_aim/ckpt_003999.pkl"
    ckpt = v30b if v30b.is_file() else v29
    _check_ppo_one_update(ckpt)
    print("\n[ALL PASS] smoke_v30c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
