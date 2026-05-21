"""Full training entrypoint. Reads YAML config and dispatches to runner.

Usage:
  python -m orbit_wars_rl.scripts.train --config orbit_wars_rl/configs/mvp.yaml \
      --log-dir ./logs/mvp_run1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from orbit_wars_rl.ppo.runner import TrainConfig, train
from orbit_wars_rl.ppo.update import PPOConfig


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "configs" / "mvp.yaml"),
    )
    ap.add_argument("--log-dir", type=str, default=None,
                    help="tensorboard log dir; set None to disable tb")
    ap.add_argument("--num-updates", type=int, default=None,
                    help="override config num_updates (for quick experiments)")
    args = ap.parse_args()

    cfg_dict = _load_yaml(args.config)
    train_cfg_dict = cfg_dict.get("train", {})
    ppo_cfg_dict = cfg_dict.get("ppo", {})

    if args.num_updates is not None:
        train_cfg_dict["num_updates"] = args.num_updates

    ppo_cfg = PPOConfig(**ppo_cfg_dict)
    train_cfg = TrainConfig(**train_cfg_dict, ppo=ppo_cfg)

    log_dir = args.log_dir
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    result = train(train_cfg, log_dir=log_dir)
    h = result["history"]
    if h:
        last = h[-1]
        if "eval/win_rate" in last:
            print(f"final eval win_rate vs random: {last['eval/win_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
