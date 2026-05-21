"""Numpy-only inference layer for Kaggle single-file submission.

This package is kept dependency-free aside from numpy so it can be inlined into
a Kaggle ``submission_rl_v1.py`` without dragging jax/flax along. The training
code under ``orbit_wars_rl.net`` is the source of truth; this layer mirrors it
parameter-for-parameter.
"""
