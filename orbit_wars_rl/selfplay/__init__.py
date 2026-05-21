"""Self-play utilities and eval harness."""

from orbit_wars_rl.selfplay.eval import play_vs_random, play_vs_frozen
from orbit_wars_rl.selfplay.pool import FrozenAgentPool

__all__ = ["play_vs_random", "play_vs_frozen", "FrozenAgentPool"]
