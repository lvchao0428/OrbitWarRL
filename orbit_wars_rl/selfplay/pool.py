"""Frozen-agent pool: keep K recent snapshots of the learner for self-play.

The pool stores **jax-array pytrees** so they can be passed straight into the
jit'd rollout. ``snapshot()`` always wraps incoming params in ``jnp.asarray``
to avoid keeping references to mutable numpy memory across train steps.
"""

from __future__ import annotations

from typing import List, Optional

import jax
import jax.numpy as jnp


class FrozenAgentPool:
    """FIFO ring buffer of param pytrees. Newest snapshot is at index -1."""

    def __init__(self, capacity: int = 5) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >=1, got {capacity}")
        self.capacity = int(capacity)
        self._items: List = []

    def __len__(self) -> int:
        return len(self._items)

    def snapshot(self, params) -> None:
        frozen = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)
        self._items.append(frozen)
        if len(self._items) > self.capacity:
            self._items.pop(0)

    def latest(self):
        """Return the most recent snapshot or ``None`` if the pool is empty."""
        return self._items[-1] if self._items else None

    def sample(self, rng: jnp.ndarray):
        """Uniform random snapshot. ``None`` when the pool is empty."""
        if not self._items:
            return None
        idx = int(jax.random.randint(rng, (), 0, len(self._items)))
        return self._items[idx]

    def all(self) -> List:
        """Read-only view of the current pool (newest last)."""
        return list(self._items)
