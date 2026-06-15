"""Reward / termination unit tests.

These directly mirror the rules in
``/opt/anaconda3/lib/python3.12/site-packages/kaggle_environments/envs/orbit_wars/orbit_wars.py``
lines 684-715. Any change to those rules must update both files.

Run with: ``python -m orbit_wars_rl.env.test_rewards``.
"""

from __future__ import annotations

import jax.numpy as jnp

from orbit_wars_rl.env import constants, state, rewards


def _two_planet_state(p0_ships: int, p1_ships: int):
    """Build a minimal EnvState with one planet per player + given ship counts."""
    s = state.empty_state(constants.MAX_PLANETS, constants.MAX_FLEETS)
    # Mutate two planet slots
    s = s.replace(
        planet_owner=s.planet_owner.at[0].set(0).at[1].set(1),
        planet_mask=s.planet_mask.at[0].set(True).at[1].set(True),
        planet_ships=s.planet_ships.at[0].set(p0_ships).at[1].set(p1_ships),
    )
    return s


def _both_dead_state():
    """No planets, no fleets → both alive_count == 0 → done by elim."""
    return state.empty_state(constants.MAX_PLANETS, constants.MAX_FLEETS)


def _expect(name: str, got: float, want: float) -> None:
    ok = float(got) == float(want)
    flag = "OK " if ok else "FAIL"
    print(f"[{flag}] {name}: got={float(got):+.1f} want={float(want):+.1f}")
    if not ok:
        raise AssertionError(f"{name}: got={got} want={want}")


def main() -> int:
    # --- terminal_reward: regular win / loss ----------------------------
    s_pwin = _two_planet_state(p0_ships=100, p1_ships=50)
    _expect("p0 beats p1: p0=+1", rewards.terminal_reward(s_pwin, 0), 1.0)
    _expect("p0 beats p1: p1=-1", rewards.terminal_reward(s_pwin, 1), -1.0)

    s_ploss = _two_planet_state(p0_ships=50, p1_ships=100)
    _expect("p0 loses to p1: p0=-1", rewards.terminal_reward(s_ploss, 0), -1.0)
    _expect("p0 loses to p1: p1=+1", rewards.terminal_reward(s_ploss, 1), 1.0)

    # --- TIE: kaggle gives +1 to BOTH ----------------------------------
    # Pre-Day-3 bug: we gave 0/0. Kaggle's rule
    # ``scores[i] == max_score and max_score > 0`` makes both winners.
    s_tie = _two_planet_state(p0_ships=75, p1_ships=75)
    _expect("ship-count tie: p0=+1 (was 0)", rewards.terminal_reward(s_tie, 0), 1.0)
    _expect("ship-count tie: p1=+1 (was 0)", rewards.terminal_reward(s_tie, 1), 1.0)

    # --- DOUBLE WIPEOUT: max=0, kaggle gives -1 to BOTH ---------------
    # Pre-Day-3 bug: we gave 0/0. Kaggle's rule clamps to -1 when max==0.
    s_dead = _both_dead_state()
    _expect("both dead: p0=-1 (was 0)", rewards.terminal_reward(s_dead, 0), -1.0)
    _expect("both dead: p1=-1 (was 0)", rewards.terminal_reward(s_dead, 1), -1.0)

    # --- one player wiped, other has 1 ship ---------------------------
    s_thin_win = _two_planet_state(p0_ships=1, p1_ships=0)
    _expect("p0=1 ship, p1=0: p0=+1", rewards.terminal_reward(s_thin_win, 0), 1.0)
    _expect("p0=1 ship, p1=0: p1=-1", rewards.terminal_reward(s_thin_win, 1), -1.0)

    # --- is_terminal: step cap matches kaggle (episodeSteps - 2) -----
    # Kaggle terminates when step >= episodeSteps - 2.
    # i.e. with default 500, termination at step 498.
    s = _two_planet_state(100, 100)
    s_step497 = s.replace(step=jnp.int32(497))
    s_step498 = s.replace(step=jnp.int32(498))
    s_step499 = s.replace(step=jnp.int32(499))
    _expect("step=497, ep=500: not done", float(rewards.is_terminal(s_step497, 500)), 0.0)
    _expect("step=498, ep=500: done    ", float(rewards.is_terminal(s_step498, 500)), 1.0)
    _expect("step=499, ep=500: done    ", float(rewards.is_terminal(s_step499, 500)), 1.0)

    # --- is_terminal: elimination overrides step cap -----------------
    s_alone = _two_planet_state(0, 100)  # p0 dead, p1 alive
    # Re-zero p0 slot fully (planet_mask still True with 0 ships still counts as alive
    # by our rule; but matching kaggle, ``alive == has any planets OR any fleet'').
    # Use both-dead state to test elim.
    s_dead0 = s_dead.replace(step=jnp.int32(0))
    _expect("both dead at step=0: done", float(rewards.is_terminal(s_dead0, 500)), 1.0)

    # --- SHAPING_SCALE default = 0.0 (Day 3 audit) ------------------
    _expect("SHAPING_SCALE default = 0.0", rewards.SHAPING_SCALE, 0.0)

    # --- Day 4 shaping family defaults (must be 0 for backward compat) --
    _expect("SHAPING_KEEP_HOME default = 0.0",  rewards.SHAPING_KEEP_HOME,  0.0)
    _expect("SHAPING_FLEET_SIZE default = 0.0", rewards.SHAPING_FLEET_SIZE, 0.0)

    # --- keep_home_reward: with default 0 coefficient must return 0 ---
    s_home = _two_planet_state(p0_ships=50, p1_ships=50)
    s_home = s_home.replace(home_planet_idx=jnp.array([0, 1], dtype=jnp.int32))
    _expect("keep_home p0 (coef=0)", float(rewards.keep_home_reward(s_home, 0)), 0.0)
    _expect("keep_home p1 (coef=0)", float(rewards.keep_home_reward(s_home, 1)), 0.0)

    # --- keep_home_reward: with coef=1, home owned -> tanh(log1p(N)/8) ---
    rewards.SHAPING_KEEP_HOME = 1.0
    try:
        got = float(rewards.keep_home_reward(s_home, 0))
        want = float(jnp.tanh(jnp.log1p(jnp.float32(50.0)) / 8.0))
        _expect("keep_home p0 (coef=1, owned)", got, want)

        # Home captured -> reward = 0 regardless of garrison.
        s_lost = s_home.replace(planet_owner=s_home.planet_owner.at[0].set(1))
        _expect("keep_home p0 (home lost)", float(rewards.keep_home_reward(s_lost, 0)), 0.0)
    finally:
        rewards.SHAPING_KEEP_HOME = 0.0

    # --- fleet_size_reward: with default 0 coefficient must return 0 ---
    valid = jnp.array([True, True, False, False, False, False, False, False])
    ships = jnp.array([10, 5, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    _expect("fleet_size (coef=0)", float(rewards.fleet_size_reward(valid, ships)), 0.0)

    # --- fleet_size_reward: with coef=1, ships=10 (NORM=20, FLOOR=0.2) ---
    rewards.SHAPING_FLEET_SIZE = 1.0
    try:
        # ships=10 -> 10/20 - 0.2 = 0.3; ships=5 -> 5/20 - 0.2 = 0.05
        got = float(rewards.fleet_size_reward(valid, ships))
        want = 0.3 + 0.05
        _expect("fleet_size (coef=1, 10+5 ships)", round(got, 5), round(want, 5))

        # Invalid launches contribute nothing even when ships>0.
        valid_none = jnp.zeros_like(valid)
        _expect("fleet_size (no valid)", float(rewards.fleet_size_reward(valid_none, ships)), 0.0)

        # Tiny launch (1 ship) -> 1/20 - 0.2 = -0.15: negative penalty.
        v1 = jnp.array([True, False, False, False, False, False, False, False])
        s1 = jnp.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
        _expect("fleet_size (1 ship -> -0.15)",
                round(float(rewards.fleet_size_reward(v1, s1)), 4), -0.15)
    finally:
        rewards.SHAPING_FLEET_SIZE = 0.0

    # ============ Day 4 §12 v2 shaping (post-expert-replay) ==============
    _expect("SHAPING_PROD_SHARE default = 0.0",   rewards.SHAPING_PROD_SHARE,   0.0)
    _expect("SHAPING_PLANET_SHARE default = 0.0", rewards.SHAPING_PLANET_SHARE, 0.0)
    _expect("SHAPING_FLEET_LOG default = 0.0",    rewards.SHAPING_FLEET_LOG,    0.0)

    # --- prod_share_reward: build a state where p0 has 2 planets prod=3+5,
    # p1 has 1 planet prod=4. total = 12. my_share = 8/12 = 0.667.
    # reward (coef=1) = 0.667 - 0.5 = +0.167.
    s_prod = state.empty_state(constants.MAX_PLANETS, constants.MAX_FLEETS)
    s_prod = s_prod.replace(
        planet_owner=s_prod.planet_owner.at[0].set(0).at[1].set(0).at[2].set(1),
        planet_mask=s_prod.planet_mask.at[0].set(True).at[1].set(True).at[2].set(True),
        planet_prod=s_prod.planet_prod.at[0].set(3).at[1].set(5).at[2].set(4),
    )
    _expect("prod_share p0 (coef=0)", float(rewards.prod_share_reward(s_prod, 0)), 0.0)
    rewards.SHAPING_PROD_SHARE = 1.0
    try:
        _expect("prod_share p0 (coef=1)",
                round(float(rewards.prod_share_reward(s_prod, 0)), 3), 0.167)
        _expect("prod_share p1 (coef=1)",
                round(float(rewards.prod_share_reward(s_prod, 1)), 3), -0.167)
    finally:
        rewards.SHAPING_PROD_SHARE = 0.0

    # --- planet_share_reward: same state, p0 has 2/3 planets.
    # reward (coef=1) = 0.667 - 0.5 = +0.167
    _expect("planet_share p0 (coef=0)", float(rewards.planet_share_reward(s_prod, 0)), 0.0)
    rewards.SHAPING_PLANET_SHARE = 1.0
    try:
        _expect("planet_share p0 (coef=1)",
                round(float(rewards.planet_share_reward(s_prod, 0)), 3), 0.167)
        _expect("planet_share p1 (coef=1)",
                round(float(rewards.planet_share_reward(s_prod, 1)), 3), -0.167)
    finally:
        rewards.SHAPING_PLANET_SHARE = 0.0

    # --- fleet_size_log_reward: log scale, REF=500, FLOOR=0.3 ---
    rewards.SHAPING_FLEET_LOG = 1.0
    try:
        # ships=1 -> log1p(1)/log1p(500)=0.111 -> -0.189
        v1m = jnp.array([True, False, False, False, False, False, False, False])
        s1 = jnp.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
        _expect("fleet_log (1 ship -> -0.19)",
                round(float(rewards.fleet_size_log_reward(v1m, s1)), 2), -0.19)

        # ships=500 -> 1.000 - 0.3 = +0.7
        s500 = jnp.array([500, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
        _expect("fleet_log (500 ship -> +0.70)",
                round(float(rewards.fleet_size_log_reward(v1m, s500)), 2), 0.70)

        # ships=2000 (above REF) -> clipped to 1.000 - 0.3 = +0.7 (no over-reward)
        s2000 = jnp.array([2000, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
        _expect("fleet_log (2000 ship -> capped +0.70)",
                round(float(rewards.fleet_size_log_reward(v1m, s2000)), 2), 0.70)

        # invalid launch contributes nothing
        v0 = jnp.zeros_like(v1m)
        _expect("fleet_log (no valid)",
                float(rewards.fleet_size_log_reward(v0, s500)), 0.0)
    finally:
        rewards.SHAPING_FLEET_LOG = 0.0

    # ============ Day 5 (post-top10) shaping ===============================
    _expect("SHAPING_PROD_SHARE_DELTA default = 0.0",
            rewards.SHAPING_PROD_SHARE_DELTA, 0.0)
    _expect("SHAPING_EMIT_LOG default = 0.0", rewards.SHAPING_EMIT_LOG, 0.0)
    _expect("SHAPING_RELEASE default = 0.0",  rewards.SHAPING_RELEASE,  0.0)

    # --- R1 prod_share_delta_reward: state where we go from p0_share=0.5
    # (3/3+3) to p0_share=0.667 (8/8+4); delta = +0.167.
    s_prev = state.empty_state(constants.MAX_PLANETS, constants.MAX_FLEETS)
    s_prev = s_prev.replace(
        planet_owner=s_prev.planet_owner.at[0].set(0).at[1].set(1),
        planet_mask=s_prev.planet_mask.at[0].set(True).at[1].set(True),
        planet_prod=s_prev.planet_prod.at[0].set(3).at[1].set(3),
    )
    s_next = s_prev.replace(
        planet_owner=s_prev.planet_owner.at[0].set(0).at[1].set(0).at[2].set(1),
        planet_mask=s_prev.planet_mask.at[0].set(True).at[1].set(True).at[2].set(True),
        planet_prod=s_prev.planet_prod.at[0].set(3).at[1].set(5).at[2].set(4),
    )
    _expect("prod_share_delta (coef=0)",
            float(rewards.prod_share_delta_reward(s_prev, s_next, 0)), 0.0)
    rewards.SHAPING_PROD_SHARE_DELTA = 1.0
    try:
        # share_prev = 3/6 = 0.5; share_next = 8/12 = 0.667; delta = +0.167
        _expect("prod_share_delta p0 (+0.167)",
                round(float(rewards.prod_share_delta_reward(s_prev, s_next, 0)), 3),
                0.167)
        # p1 sees the opposite sign.
        _expect("prod_share_delta p1 (-0.167)",
                round(float(rewards.prod_share_delta_reward(s_prev, s_next, 1)), 3),
                -0.167)
        # zero delta when state is unchanged.
        _expect("prod_share_delta zero on no-op",
                float(rewards.prod_share_delta_reward(s_next, s_next, 0)), 0.0)
    finally:
        rewards.SHAPING_PROD_SHARE_DELTA = 0.0

    # --- R4 emit_log_reward: log1p of number of valid launches ---
    v0 = jnp.zeros((8,), dtype=jnp.bool_)
    v2 = jnp.array([True, True, False, False, False, False, False, False])
    v8 = jnp.ones((8,), dtype=jnp.bool_)
    _expect("emit_log (coef=0)", float(rewards.emit_log_reward(v2)), 0.0)
    rewards.SHAPING_EMIT_LOG = 1.0
    try:
        _expect("emit_log (0 valid)", float(rewards.emit_log_reward(v0)), 0.0)
        # log1p(2) = ln(3) ~= 1.0986
        _expect("emit_log (2 valid -> log1p(2))",
                round(float(rewards.emit_log_reward(v2)), 4),
                round(float(jnp.log1p(jnp.float32(2.0))), 4))
        # log1p(8) ~= 2.197
        _expect("emit_log (8 valid -> log1p(8))",
                round(float(rewards.emit_log_reward(v8)), 4),
                round(float(jnp.log1p(jnp.float32(8.0))), 4))
    finally:
        rewards.SHAPING_EMIT_LOG = 0.0

    # --- R2 release_bonus_reward: src_prod=5, src_garr=200 → 200/5/20=2.0
    #     -> tanh(2.0 - 1) = tanh(1) ~= 0.762
    #     ships=100 -> log_size = log1p(100)/log1p(500) ~= 0.741
    #     per_launch = 0.741 * 0.762 = 0.565
    s_rel = state.empty_state(constants.MAX_PLANETS, constants.MAX_FLEETS)
    s_rel = s_rel.replace(
        planet_owner=s_rel.planet_owner.at[0].set(0),
        planet_mask=s_rel.planet_mask.at[0].set(True),
        planet_ships=s_rel.planet_ships.at[0].set(200),
        planet_prod=s_rel.planet_prod.at[0].set(5),
    )
    src_idx = jnp.zeros((8,), dtype=jnp.int32)
    ships = jnp.array([100, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    valid = jnp.array([True, False, False, False, False, False, False, False])
    _expect("release_bonus (coef=0)",
            float(rewards.release_bonus_reward(s_rel, src_idx, valid, ships)), 0.0)
    rewards.SHAPING_RELEASE = 1.0
    try:
        log_size = float(jnp.log1p(jnp.float32(100.0))
                         / jnp.log1p(jnp.float32(rewards.SHAPING_FLEET_LOG_REF)))
        release_factor = float(jnp.tanh(jnp.float32(2.0 - 1.0)))
        want = log_size * release_factor
        got = float(rewards.release_bonus_reward(s_rel, src_idx, valid, ships))
        _expect("release_bonus stockpiled launch",
                round(got, 4), round(want, 4))

        # Same launch but the source has only 5 ships (=1 turn of prod);
        # generations = 5/5/20 = 0.05 -> tanh(0.05-1) = tanh(-0.95) ~= -0.74
        # -> negative bonus (penalty for releasing too early).
        s_low = s_rel.replace(planet_ships=s_rel.planet_ships.at[0].set(5))
        got_low = float(rewards.release_bonus_reward(s_low, src_idx, valid, ships))
        assert got_low < 0.0, f"expected negative early release, got {got_low}"
        print(f"[OK ] release_bonus early launch < 0 (got {got_low:+.3f})")

        # Invalid launches contribute nothing.
        v_none = jnp.zeros_like(valid)
        _expect("release_bonus (no valid)",
                float(rewards.release_bonus_reward(s_rel, src_idx, v_none, ships)),
                0.0)
    finally:
        rewards.SHAPING_RELEASE = 0.0

    # --- R3: prod_share / planet_share use constants.NUM_PLAYERS, not
    #     a hard-coded 2.0. Verify the live formula matches our derivation
    #     for the 2P case (matches §141-§174 tests above) AND symbolically
    #     references NUM_PLAYERS so changing constants.NUM_PLAYERS would
    #     update the baseline automatically.
    import inspect as _inspect
    src_p = _inspect.getsource(rewards.prod_share_reward)
    src_pl = _inspect.getsource(rewards.planet_share_reward)
    assert "constants.NUM_PLAYERS" in src_p, "prod_share_reward must use constants.NUM_PLAYERS"
    assert "constants.NUM_PLAYERS" in src_pl, "planet_share_reward must use constants.NUM_PLAYERS"
    print("[OK ] prod_share / planet_share use constants.NUM_PLAYERS (no hard-coded 2.0)")

    # --- capture_roi_hist_reward: nearby factory > weak mop -----------------
    rewards.SHAPING_CAPTURE_ROI = 0.025
    rewards.SHAPING_CAPTURE = 0.0
    try:
        s0 = state.empty_state(constants.MAX_PLANETS, constants.MAX_FLEETS)
        s0 = s0.replace(
            planet_mask=s0.planet_mask.at[0].set(True).at[1].set(True).at[2].set(True),
            planet_owner=s0.planet_owner.at[0].set(0).at[1].set(constants.NEUTRAL_OWNER).at[2].set(
                constants.NEUTRAL_OWNER
            ),
            planet_x=s0.planet_x.at[0].set(0.0).at[1].set(5.0).at[2].set(6.0),
            planet_y=s0.planet_y.at[0].set(0.0).at[1].set(0.0).at[2].set(0.0),
            planet_ships=s0.planet_ships.at[0].set(100).at[1].set(2).at[2].set(18),
            planet_prod=s0.planet_prod.at[0].set(5).at[1].set(1).at[2].set(4),
            step=jnp.int32(50),
        )
        prev = s0

        next_weak = s0.replace(
            planet_owner=s0.planet_owner.at[1].set(0),
            planet_ships=s0.planet_ships.at[1].set(5),
        )
        next_factory = s0.replace(
            planet_owner=s0.planet_owner.at[2].set(0),
            planet_ships=s0.planet_ships.at[2].set(22),
        )
        r_weak = float(rewards.capture_roi_hist_reward(prev, next_weak, 0, 500))
        r_factory = float(rewards.capture_roi_hist_reward(prev, next_factory, 0, 500))
        print(f"[check] ROI reward weak={r_weak:.5f} factory={r_factory:.5f}")
        if r_factory <= r_weak:
            raise AssertionError(
                f"factory capture ROI reward should exceed weak: {r_factory} <= {r_weak}"
            )
        print("[OK ] capture_roi_hist_reward: nearby factory > weak mop")
    finally:
        rewards.SHAPING_CAPTURE_ROI = float(
            __import__("os").environ.get("ORBITWARS_SHAPING_CAPTURE_ROI", "0.0")
        )
        rewards.SHAPING_CAPTURE = float(
            __import__("os").environ.get("ORBITWARS_SHAPING_CAPTURE", "0.0")
        )

    print("\n[ALL PASS] reward function matches kaggle rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
