"""v30b smoke: warmstart + freeze attn dst + high roi aux."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp

from orbit_wars_rl.env import constants
from orbit_wars_rl.env.env import OrbitWarsEnv
from orbit_wars_rl.features.encode import encode
from orbit_wars_rl.net.model import ActorCritic
from orbit_wars_rl.ppo.runner import SelfPlayConfig, TrainConfig, train
from orbit_wars_rl.ppo.update import PPOConfig


def _check_freeze_eval() -> None:
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
    out_free = model.apply(
        params, enc, src, dst, pct, emit, s.planet_ships, s.planet_x,
        s.planet_y, hi, None, None, None, None, None,
        freeze_dst_attn=False, method=ActorCritic.evaluate,
    )
    out_frz = model.apply(
        params, enc, src, dst, pct, emit, s.planet_ships, s.planet_x,
        s.planet_y, hi, None, None, None, None, None,
        freeze_dst_attn=True, method=ActorCritic.evaluate,
    )
    assert out_free.dst_logits.shape == out_frz.dst_logits.shape
    print("[OK ] evaluate freeze_dst_attn path")


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
        ckpt_dir=str(tempfile.mkdtemp(prefix="smoke_v30b_")),
        ckpt_every=0,
        log_every=1,
        d_model=256,
        n_layers=4,
        n_heads=8,
        ff_dim=1024,
        flip_hard_mask=True,
        allow_hold=True,
        zero_sum_value=True,
        resume_ckpt=str(ckpt_path),
        ppo=PPOConfig(
            lr_warmup_steps=1,
            lr_decay_steps=10,
            update_epochs=1,
            num_minibatches=1,
            roi_aux_coef=0.4,
            freeze_dst_attn_updates=1000,
        ),
        selfplay=SelfPlayConfig(enabled=False),
    )
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    train(cfg, log_dir=None, resume_from=str(ckpt_path))
    print(f"[OK ] v30b 1 update from {ckpt_path.name}")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    print("=== smoke_v30b ===")
    _check_freeze_eval()
    warm = root / "ckpt_multi_action_v30b_warm/ckpt_warm.pkl"
    init = root / "ckpt_multi_action_v29_aim/ckpt_003999.pkl"
    ckpt = warm if warm.is_file() else init
    _check_ppo_one_update(ckpt)
    print("\n[ALL PASS] smoke_v30b")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
