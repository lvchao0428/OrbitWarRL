"""PPO training: rollout collection, loss/update, runner."""

from orbit_wars_rl.ppo.rollout import (
    Rollout,
    make_rollout_fn,
    random_opponent_action,
)

__all__ = ["Rollout", "make_rollout_fn", "random_opponent_action"]
