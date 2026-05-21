"""Run a tiny training loop to confirm everything is wired and JIT compiles.

Used as both a sanity check and an example of how to drive the trainer.

  python -m orbit_wars_rl.scripts.smoke_test
"""

from __future__ import annotations

import argparse
import time

from orbit_wars_rl.ppo.runner import SelfPlayConfig, TrainConfig, train
from orbit_wars_rl.ppo.update import PPOConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--rollout-length", type=int, default=32)
    ap.add_argument("--num-updates", type=int, default=10)
    ap.add_argument("--episode-steps", type=int, default=60)
    ap.add_argument("--num-groups", type=int, default=4)
    ap.add_argument("--selfplay", action="store_true",
                    help="exercise the frozen-opponent rollout path too")
    args = ap.parse_args()

    sp = SelfPlayConfig(
        enabled=args.selfplay,
        # short schedule so smoke actually hits the frozen path
        warmup_updates=2 if args.selfplay else 1000,
        snapshot_every=2,
        pool_capacity=3,
        frozen_ratio=0.5,
        eval_vs_frozen=args.selfplay,
    )

    cfg = TrainConfig(
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        num_updates=args.num_updates,
        episode_steps=args.episode_steps,
        num_groups=args.num_groups,
        eval_every=max(1, args.num_updates // 2),
        eval_num_envs=8,
        ckpt_dir="./tmp_smoke_ckpt",
        ckpt_every=0,
        log_every=1,
        ppo=PPOConfig(
            lr_warmup_steps=4,
            lr_decay_steps=max(args.num_updates * 2, 20),
            update_epochs=2,
            num_minibatches=2,
        ),
        selfplay=sp,
    )

    t0 = time.time()
    result = train(cfg, log_dir=None)
    elapsed = time.time() - t0

    history = result["history"]
    print()
    print(f"smoke run finished in {elapsed:.1f}s ({args.num_updates} updates)")
    print(f"final SPS: {history[-1]['sps']:.0f}")
    has_nan = any(
        any(v != v for k, v in m.items() if isinstance(v, float))
        for m in history
    )
    print(f"any NaN in metrics? {has_nan}")
    if "eval/win_rate" in history[-1]:
        print(f"final eval win_rate vs random: {history[-1]['eval/win_rate']:.2f}")
    return 1 if has_nan else 0


if __name__ == "__main__":
    raise SystemExit(main())
