"""Feature encoding for the entity transformer."""

from orbit_wars_rl.features.encode import (
    EncodedObs,
    PLANET_FEAT_DIM,
    FLEET_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    BASE_GLOBAL_FEAT_DIM,
    encode,
)
from orbit_wars_rl.features.history import HIST_LEN, TEMPORAL_GLOBAL_DIM
from orbit_wars_rl.features.pair import (
    DST_PAIR_DIM,
    EMIT_PAIR_DIM,
    PCT_PAIR_DIM,
    dst_pair_features_batched,
    emit_pair_globals,
    pct_pair_features,
)

__all__ = [
    "EncodedObs",
    "PLANET_FEAT_DIM",
    "FLEET_FEAT_DIM",
    "GLOBAL_FEAT_DIM",
    "encode",
    "DST_PAIR_DIM",
    "EMIT_PAIR_DIM",
    "PCT_PAIR_DIM",
    "dst_pair_features_batched",
    "emit_pair_globals",
    "pct_pair_features",
]
