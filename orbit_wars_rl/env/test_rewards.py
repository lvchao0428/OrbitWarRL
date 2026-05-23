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

    print("\n[ALL PASS] reward function matches kaggle rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
