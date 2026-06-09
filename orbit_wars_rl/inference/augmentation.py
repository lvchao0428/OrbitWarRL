"""Test-time data augmentation for inference.

The Orbit Wars board has 4-fold symmetry (center at 50,50):
  * Identity
  * 180° rotation: (x,y) → (100-x, 100-y), angle → angle + π
  * Mirror-X: (x,y) → (100-x, y), angle → π - angle
  * Mirror-Y: (x,y) → (x, 100-y), angle → -angle

At inference time we run the model under all 4 augmentations, average the
logits, then sample from the averaged policy. This smooths out positional
biases and exploits the geometric symmetry of the game.

For each augmentation we must:
  1. Transform the observation (planet positions, fleet positions/angles)
  2. Run the model forward to get logits
  3. Inverse-transform the action logits (src/dst logits are per-planet,
     so if planet indices didn't change, no inverse is needed)

Since planet indices stay the same (we transform coordinates in-place,
not re-sort planets), no inverse transform on logits is needed. The model
sees the same planet IDs with different spatial features and produces
logits that can be directly averaged.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

BOARD = 100.0


class Augmentation(NamedTuple):
    """Defines a spatial augmentation as a transform on (x, y, angle)."""
    name: str
    flip_x: bool   # x → BOARD - x
    flip_y: bool   # y → BOARD - y


AUGMENTATIONS = [
    Augmentation("identity", flip_x=False, flip_y=False),
    Augmentation("rot180",   flip_x=True,  flip_y=True),
    Augmentation("mirror_x", flip_x=True,  flip_y=False),
    Augmentation("mirror_y", flip_x=False, flip_y=True),
]


def transform_positions(
    planet_x: np.ndarray,
    planet_y: np.ndarray,
    fleet_x: np.ndarray,
    fleet_y: np.ndarray,
    fleet_angle: np.ndarray,
    aug: Augmentation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply spatial augmentation to positions and angles.

    Returns transformed (planet_x, planet_y, fleet_x, fleet_y, fleet_angle).
    All inputs/outputs are numpy arrays.
    """
    px = np.float32(BOARD) - planet_x if aug.flip_x else planet_x.copy()
    py = np.float32(BOARD) - planet_y if aug.flip_y else planet_y.copy()
    fx = np.float32(BOARD) - fleet_x if aug.flip_x else fleet_x.copy()
    fy = np.float32(BOARD) - fleet_y if aug.flip_y else fleet_y.copy()

    fa = fleet_angle.copy()
    if aug.flip_x and aug.flip_y:
        fa = fa + np.float32(np.pi)
    elif aug.flip_x:
        fa = np.float32(np.pi) - fa
    elif aug.flip_y:
        fa = -fa

    # Normalize angle to [-π, π]
    fa = np.arctan2(np.sin(fa), np.cos(fa))

    return px, py, fx, fy, fa


def average_logits(
    logits_list: list[np.ndarray],
) -> np.ndarray:
    """Average logits from multiple augmentations.

    Each element in logits_list is an array of the same shape.
    Returns the element-wise mean.
    """
    return np.mean(np.stack(logits_list, axis=0), axis=0)
