"""v30 integration smoke: econ head, ROI aux, resume from v29.

Run: ``python -m orbit_wars_rl.scripts.smoke_v30``
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from orbit_wars_rl.env import constants, state
from orbit_wars_rl.env.actions import noop_multi_action
from orbit_wars_rl.env.env import OrbitWarsEnv
from orbit_wars_rl.features.pair import ECON_DST_DIM
from orbit_wars_rl.net.model import ActorCritic
from orbit_wars_rl.ppo.runner import SelfPlayConfig, TrainConfig, train
from orbit_wars_rl.ppo.update import PPOConfig


def _check_econ_dim() -> None:
    assert ECON_DST_DIM == 7, ECON_DST_DIM
    print(f"[OK ] ECON_DST_DIM={ECON_DST_DIM}")


def _check_env_step() -> None:
    os.environ["ORBITWARS_SHAPING_CAPTURE_ROI"] = "0.025"
    os.environ["ORBITWARS_SHAPING_FRIENDLY_SHUFFLE"] = "0.01"
    os.environ["ORBITWARS_SHAPING_CAPTURE"] = "0"
    os.environ["ORBITWARS_SKIP_PARITY"] = "1"
    from orbit_wars_rl.env import rewards as rewards_mod

    rewards_mod.SHAPING_CAPTURE_ROI = 0.025
    rewards_mod.SHAPING_FRIENDLY_SHUFFLE = 0.01
    rewards_mod.SHAPING_CAPTURE = 0.0

    env = OrbitWarsEnv(num_groups=5, episode_steps=60)
    key = jax.random.PRNGKey(0)
    s = env.reset(key)
    hold = noop_multi_action()
    _s2, out = env.step(s, (hold, hold))
    r_f = float(jnp.asarray(out.reward))
    assert np.isfinite(r_f), f"non-finite reward: {r_f}"
    print(f"[OK ] v30 env step reward finite ({r_f:+.4f})")


def _check_model_init() -> None:
    model = ActorCritic(d_model=64, n_layers=2, n_heads=4, ff_dim=128, flip_hard_mask=True)
    key = jax.random.PRNGKey(0)
    from orbit_wars_rl.features.encode import encode

    env = OrbitWarsEnv(num_groups=5, episode_steps=60)
    s = env.reset(key)
    enc = encode(s, player=0, episode_steps=60)
    params = model.init(
        key,
        enc,
        jax.random.PRNGKey(1),
        s.planet_ships,
        s.planet_x,
        s.planet_y,
        jnp.int32(0),
    )
    flat_keys = jax.tree_util.tree_leaves_with_path(params)
    names = ["/".join(str(k) for k in p) for p, _ in flat_keys]
    assert any("dst_economics_head" in n for n in names), names[:5]
    print("[OK ] ActorCritic init includes dst_economics_head")


def _check_resume(ckpt_path: Path) -> None:
    if not ckpt_path.is_file():
        print(f"[SKIP] resume smoke — ckpt not found: {ckpt_path}")
        return
    cfg = TrainConfig(
        num_envs=4,
        rollout_length=16,
        num_updates=1,
        episode_steps=60,
        num_groups=constants.MIN_PLANET_GROUPS,
        eval_every=0,
        ckpt_dir=str(tempfile.mkdtemp(prefix="smoke_v30_")),
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
            roi_aux_coef=0.08,
        ),
        selfplay=SelfPlayConfig(enabled=False),
    )
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    train(cfg, log_dir=None, resume_from=str(ckpt_path))
    print(f"[OK ] resume 1 update from {ckpt_path.name} (v29->v30 econ head adapt)")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    for name in (
        "ckpt_multi_action_v29_aim/ckpt_003999.pkl",
        "ckpt_multi_action_v28_roi/ckpt_003999.pkl",
    ):
        ckpt = Path(os.environ.get("V30_RESUME_CKPT", name))
        if not ckpt.is_absolute():
            ckpt = root / ckpt
        if ckpt.is_file():
            break
    else:
        ckpt = root / "ckpt_multi_action_v29_aim/ckpt_003999.pkl"

    print("=== smoke_v30 ===")
    _check_econ_dim()
    _check_env_step()
    _check_model_init()
    _check_resume(ckpt)
    print("\n[ALL PASS] smoke_v30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
