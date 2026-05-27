import numpy as np

from orbit_wars_rl.bc.state_buffer_util import flip_owners, remap_4p_to_2p


def _sample(owner_vals):
    n = len(owner_vals)
    return {
        "planet_owner": np.asarray(owner_vals, dtype=np.int8),
        "fleet_owner": np.asarray([-1, 0, 2, 3][:n], dtype=np.int8),
        "home_planet_idx": np.zeros(2, dtype=np.int32),
        "step": np.int32(42),
    }


def test_remap_4p_to_2p():
    d = _sample([-1, 0, 1, 2, 3])
    out = remap_4p_to_2p(d, perspective=2)
    assert out["planet_owner"].tolist() == [-1, 1, 1, 0, 1]
    assert out["fleet_owner"].tolist() == [-1, 1, 0, 1]


def test_flip_owners():
    d = _sample([0, 1, -1])
    out = flip_owners(d)
    assert out["planet_owner"].tolist() == [1, 0, -1]
    assert out["fleet_owner"].tolist() == [-1, 1, 2]
