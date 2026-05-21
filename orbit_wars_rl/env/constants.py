"""Global env constants. Shapes MUST stay fixed for jit/vmap to be happy."""

from __future__ import annotations

BOARD: float = 100.0
SUN_X: float = 50.0
SUN_Y: float = 50.0
SUN_RADIUS: float = 10.0
SUN_PATH_MARGIN: float = 1.25

MAX_PLANETS: int = 40
MAX_FLEETS: int = 128

NUM_PLAYERS: int = 2
NEUTRAL_OWNER: int = -1
PADDING_OWNER: int = -2

DEFAULT_MAX_SHIP_SPEED: float = 6.0
DEFAULT_EPISODE_STEPS: int = 500

MIN_PLANET_GROUPS: int = 3
MAX_PLANET_GROUPS: int = 10
PLANETS_PER_GROUP: int = 4

NUM_PCT_BINS: int = 4
PCT_BIN_VALUES = (0.25, 0.5, 0.75, 1.0)

HOME_PLANET_SHIPS: int = 10
NEUTRAL_SHIPS_MIN: int = 5
NEUTRAL_SHIPS_MAX: int = 50

PROD_MIN: int = 1
PROD_MAX: int = 5
