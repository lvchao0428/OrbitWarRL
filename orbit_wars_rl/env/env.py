"""High-level env API: reset / step / vectorized variants.

All methods are jit-compatible. ``reset`` is keyed by an rng; ``step`` takes a
PyTree of player actions and returns a fresh state + scalar/array outputs.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from orbit_wars_rl.env import constants, dynamics, init, rewards
from orbit_wars_rl.env.actions import MultiPlayerAction, PlayerAction, single_to_multi
from orbit_wars_rl.env.state import EnvState


@chex.dataclass(frozen=True)
class EnvOutput:
    """Per-step env outputs aside from the next state."""

    reward: chex.Array
    done: chex.Array
    info_my_ships: chex.Array
    info_opp_ships: chex.Array


class OrbitWarsEnv:
    """Single-env wrapper. Use ``jax.vmap`` for batches.

    The class is just a thin namespace -- nothing is stored on it; methods are
    pure functions of (state, rng, actions). This keeps everything jit-friendly.
    """

    def __init__(
        self,
        num_groups: int = 5,
        episode_steps: int = constants.DEFAULT_EPISODE_STEPS,
    ) -> None:
        self.num_groups = int(num_groups)
        self.episode_steps = int(episode_steps)

    def reset(self, rng: jnp.ndarray) -> EnvState:
        return init.reset(rng, num_groups=self.num_groups)

    def step(
        self,
        state: EnvState,
        actions: tuple[MultiPlayerAction, MultiPlayerAction],
    ) -> tuple[EnvState, EnvOutput]:
        """Take a turn given each player's multi-fleet action.

        ``actions`` must be a tuple of two ``MultiPlayerAction``. Legacy code
        that only has single ``PlayerAction``s can wrap them via
        ``actions.single_to_multi`` (kept jit-pure).
        """
        s1, valids, ships_launched = dynamics.launch_fleets_with_info(state, actions)
        valid_p0 = valids[0]
        ships_p0 = ships_launched[0]
        s2 = dynamics.produce(s1)
        # Fleet movement uses swept-pair collision against each planet's
        # *future* (post-rotation) segment. This is how the Kaggle env
        # detects continuous collision: any fleet crossing the swept area of
        # a moving planet is sent into combat with it. So there's no
        # separate sweep phase -- the move + planet rotation happen jointly.
        s3, hit_pidx, hit_mask = dynamics.move_and_collide(s2)
        s_combat = dynamics.resolve_combat(s3, hit_pidx, hit_mask)
        # Apply the planet rotation we already accounted for in collision
        # detection. After this, planet_x/y reflect the end-of-tick position.
        s4 = dynamics.rotate_planets(s_combat)

        next_step = s4.step + 1
        done_now = rewards.is_terminal(s4.replace(step=next_step), self.episode_steps)
        terminal_r = rewards.terminal_reward(s4, 0)
        # Potential-based dense shaping computed on the *board state before
        # the step counter advances*. Suppressed on the terminal step so the
        # +/-1 isn't double-counted by the value head.
        shaping = rewards.shaping_delta(state, s4, 0)
        # Day 4 Track 3 shaping family. Each term defaults off via env var
        # so v8 ckpt resume is reward-bit-exact. All terms operate on player 0.
        #
        #   keep_home / fleet_size : v1 family (DAY4 §11.1-§11.2, kept for
        #     parity / ablation; default off since v9, replaced below).
        #   prod_share / planet_share / fleet_size_log : v2 family motivated
        #     by 5-episode top10 expert analysis (DAY4 §12). prod_share has
        #     PERFECT 5-of-5 separator between winners and losers in expert
        #     replays.
        keep_r = rewards.keep_home_reward(s4, 0)
        fleet_r = rewards.fleet_size_reward(valid_p0, ships_p0)
        prod_r = rewards.prod_share_reward(s4, 0)
        planet_r = rewards.planet_share_reward(s4, 0)
        fleet_log_r = rewards.fleet_size_log_reward(valid_p0, ships_p0)
        # Day 5 (post-top10 deep-analysis) shaping family. All default 0 so
        # adding the call sites is reward-bit-exact for existing configs.
        #
        #   prod_share_DELTA : credit-assigning version of prod_share. Sum
        #     over an episode telescopes to ``coef * (share_end - share_start)``.
        #   emit_log         : log1p of valid launches this turn -- no
        #     "multi-emit >=K" threshold, just monotone reward for emitting.
        #   release_bonus    : log_size * tanh(src_garr/src_prod/K - 1).
        #     Normalizes stockpile by each planet's own production; no
        #     ship-count threshold and no stored EMA state.
        prod_d_r = rewards.prod_share_delta_reward(state, s4, 0)
        capture_r = rewards.capture_flip_reward(state, s4, 0)
        release_r = rewards.release_bonus_reward(
            state, actions[0].src_idx, valid_p0, ships_p0
        )
        emit_log_r = rewards.emit_log_reward_gated(
            valid_p0, state, actions[0].src_idx
        )
        # Day 8 / f35: sharp anti-1-ship + high-prod capture. Default 0.
        one_ship_r = rewards.one_ship_penalty_reward(valid_p0, ships_p0)
        high_prod_r = rewards.high_prod_capture_reward(state, s4, 0)
        # Day 11 / f42: fleet-scaled capture bonus. Default 0.
        capture_fs_r = rewards.capture_fleet_scale_reward(state, s4, 0)
        # Day 11 / f43: gated multi-emit bonus. Default 0.
        multi_emit_r = rewards.multi_emit_gated_bonus_reward(valid_p0, ships_p0)
        non_terminal_r = (
            shaping
            + keep_r + fleet_r
            + prod_r + planet_r + fleet_log_r
            + prod_d_r + capture_r + emit_log_r + release_r
            + one_ship_r + high_prod_r + capture_fs_r + multi_emit_r
        )
        reward_p0 = jnp.where(done_now, terminal_r, non_terminal_r)

        out = EnvOutput(
            reward=reward_p0,
            done=done_now,
            info_my_ships=rewards.player_total_ships(s4, 0).astype(jnp.float32),
            info_opp_ships=rewards._strongest_opp_ships(s4, 0).astype(jnp.float32),
        )
        return s4.replace(step=next_step, done=done_now), out

    def step_and_autoreset(
        self,
        state: EnvState,
        actions: tuple[MultiPlayerAction, MultiPlayerAction],
        reset_rng: jnp.ndarray,
    ) -> tuple[EnvState, EnvOutput]:
        """Like ``step`` but resets when done. Used inside rollouts."""
        next_state, out = self.step(state, actions)
        fresh = init.reset(reset_rng, num_groups=self.num_groups)
        chosen = jax.tree_util.tree_map(
            lambda a, b: jnp.where(out.done, a, b),
            fresh, next_state,
        )
        return chosen, out

    def step_single(
        self,
        state: EnvState,
        actions: tuple[PlayerAction, PlayerAction],
    ) -> tuple[EnvState, EnvOutput]:
        """Legacy single-fleet step. Wraps the inputs as K=1 MultiPlayerAction."""
        multi_actions = (single_to_multi(actions[0]), single_to_multi(actions[1]))
        return self.step(state, multi_actions)

    def step_and_autoreset_single(
        self,
        state: EnvState,
        actions: tuple[PlayerAction, PlayerAction],
        reset_rng: jnp.ndarray,
    ) -> tuple[EnvState, EnvOutput]:
        """Legacy single-fleet step_and_autoreset. Wraps inputs as K=1."""
        multi_actions = (single_to_multi(actions[0]), single_to_multi(actions[1]))
        return self.step_and_autoreset(state, multi_actions, reset_rng)
