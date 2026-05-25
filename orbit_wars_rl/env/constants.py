"""Global env constants. Shapes MUST stay fixed for jit/vmap to be happy."""

from __future__ import annotations

BOARD: float = 100.0
SUN_X: float = 50.0
SUN_Y: float = 50.0
SUN_RADIUS: float = 10.0
SUN_PATH_MARGIN: float = 1.25

MAX_PLANETS: int = 40
MAX_FLEETS: int = 128

import os as _os

NUM_PLAYERS: int = int(_os.environ.get("ORBITWARS_NUM_PLAYERS", "2"))
if NUM_PLAYERS not in (2, 4):
    raise ValueError(f"ORBITWARS_NUM_PLAYERS must be 2 or 4, got {NUM_PLAYERS}")
NEUTRAL_OWNER: int = -1
PADDING_OWNER: int = -2

DEFAULT_MAX_SHIP_SPEED: float = 6.0
DEFAULT_EPISODE_STEPS: int = 500

MIN_PLANET_GROUPS: int = 3
MAX_PLANET_GROUPS: int = 10
PLANETS_PER_GROUP: int = 4

NUM_PCT_BINS: int = 8
PCT_BIN_VALUES = (0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00)

# Max number of fleets a single player can launch per turn (autoregressive head).
# v20 uses 26; we start small to keep the unroll cheap and grow if needed.
MAX_FLEETS_PER_TURN: int = 8

HOME_PLANET_SHIPS: int = 10
NEUTRAL_SHIPS_MIN: int = 5
NEUTRAL_SHIPS_MAX: int = 50

PROD_MIN: int = 1
PROD_MAX: int = 5

# Orbital motion (matches Kaggle orbit_wars).
# Each episode samples one angular_velocity uniformly from this range; all
# orbiting planets share the same omega. Sign convention: positive = CCW
# (matches Kaggle env, confirmed via parity test seed=42 step 5 displacements).
ORBIT_OMEGA_MIN: float = 0.025
ORBIT_OMEGA_MAX: float = 0.05
# A planet orbits iff (distance_to_sun + planet_radius) < this threshold.
ORBIT_RADIUS_LIMIT: float = 50.0
