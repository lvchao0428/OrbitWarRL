"""Frozen-agent pool: keep K recent snapshots of the learner for self-play."""

from __future__ import annotations

import copy
from typing import List, Optional

import jax
import jax.numpy as jnp


class FrozenAgentPool:
    """LIFO ring buffer of param dicts. ``snapshot()`` adds, ``sample()`` picks one."""

    def __init__(self, capacity: int = 5) -> None:
        self.capacity = int(capacity)
        self._items: List = []

    def __len__(self) -> int:
        return len(self._items)

    def snapshot(self, params) -> None:
        self._items.append(jax.tree_util.tree_map(lambda x: jnp.asarray(x), params))
        if len(self._items) > self.capacity:
            self._items.pop(0)

    def sample(self, rng: jnp.ndarray):
        if not self._items:
            return None
        idx = int(jax.random.randint(rng, (), 0, len(self._items)))
        return self._items[idx]
