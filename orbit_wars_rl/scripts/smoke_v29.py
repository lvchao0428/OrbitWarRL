"""v29 integration smoke: pair ROI dim6, dedup, resume from v28, export parity.

Run: ``python -m orbit_wars_rl.scripts.smoke_v29``
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
from orbit_wars_rl.features.encode import encode
from orbit_wars_rl.features.pair import DST_PAIR_DIM
from orbit_wars_rl.ppo.runner import SelfPlayConfig, TrainConfig, train
from orbit_wars_rl.ppo.update import PPOConfig


def _check_pair_dim() -> None:
    assert DST_PAIR_DIM == 6, DST_PAIR_DIM
    print(f"[OK ] DST_PAIR_DIM={DST_PAIR_DIM}")


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
    print(f"[OK ] v29 env step reward finite ({r_f:+.4f})")


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
        ckpt_dir=str(tempfile.mkdtemp(prefix="smoke_v29_")),
        ckpt_every=0,
        log_every=1,
        d_model=256,
        n_layers=4,
        n_heads=8,
        ff_dim=1024,
        allow_hold=True,
        zero_sum_value=True,
        resume_ckpt=str(ckpt_path),
        ppo=PPOConfig(lr_warmup_steps=1, lr_decay_steps=10, update_epochs=1, num_minibatches=1),
        selfplay=SelfPlayConfig(enabled=False),
    )
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    train(cfg, log_dir=None, resume_from=str(ckpt_path))
    print(f"[OK ] resume 1 update from {ckpt_path.name} (shape-adapt dim6)")


def _check_export(ckpt_path: Path, root: Path) -> None:
    template = root / "submission_rl_v21.py"
    if not ckpt_path.is_file():
        print("[SKIP] export smoke — no ckpt")
        return
    if not template.is_file():
        print("[SKIP] export smoke — no submission_rl_v21.py")
        return
    from orbit_wars_rl.inference.weights import infer_arch_from_flat, load_flat_params

    arch = infer_arch_from_flat(load_flat_params(str(ckpt_path)))
    if int(arch.get("dst_pair_dim", 0)) != DST_PAIR_DIM:
        print(
            f"[SKIP] export smoke — ckpt dst_pair_dim={arch.get('dst_pair_dim')} "
            f"!= template {DST_PAIR_DIM} (v28 init ckpt; export after v29 train)"
        )
        return
    out = root / "tmp_smoke_v29_submission.py"
    cmd = [
        sys.executable,
        "-m",
        "orbit_wars_rl.scripts.export_submission",
        "--ckpt",
        str(ckpt_path),
        "--template",
        str(template),
        "--out",
        str(out),
        "--skip-parity",
    ]
    subprocess.run(cmd, check=True, cwd=str(root), capture_output=True, text=True)
    if out.is_file():
        out.unlink(missing_ok=True)
    print("[OK ] export_submission 1-step CPU")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    for name in (
        "ckpt_multi_action_v28_roi/ckpt_003999.pkl",
        "ckpt_multi_action_v28_roi/ckpt_latest.pkl",
    ):
        ckpt = Path(os.environ.get("V29_RESUME_CKPT", name))
        if not ckpt.is_absolute():
            ckpt = root / ckpt
        if ckpt.is_file():
            break
    else:
        ckpt = root / "ckpt_multi_action_v28_roi/ckpt_003999.pkl"

    print("=== smoke_v29 ===")
    _check_pair_dim()
    _check_env_step()
    _check_resume(ckpt)
    _check_export(ckpt, root)
    print("\n[ALL PASS] smoke_v29")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
