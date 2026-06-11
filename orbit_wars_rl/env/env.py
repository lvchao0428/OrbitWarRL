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
from orbit_wars_rl.features.history import update_global_hist


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

    ``wins_needed`` controls multi-match series (v15). Default 1 = legacy
    single-match behaviour where every match end starts a fresh map. Set to 2
    for Best-of-3 (same map rematches until one player reaches 2 wins).
    """

    def __init__(
        self,
        num_groups: int = 5,
        episode_steps: int = constants.DEFAULT_EPISODE_STEPS,
        wins_needed: int = 1,
    ) -> None:
        self.num_groups = int(num_groups)
        self.episode_steps = int(episode_steps)
        self.wins_needed = int(wins_needed)

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
        shaping = rewards.shaping_delta(state, s4, 0)
        keep_r = rewards.keep_home_reward(s4, 0)
        fleet_r = rewards.fleet_size_reward(valid_p0, ships_p0)
        prod_r = rewards.prod_share_reward(s4, 0)
        planet_r = rewards.planet_share_reward(s4, 0)
        fleet_log_r = rewards.fleet_size_log_reward(valid_p0, ships_p0)
        prod_d_r = rewards.prod_share_delta_reward(state, s4, 0)
        capture_r = rewards.capture_hist_balance_reward(state, s4, 0)
        defense_empty_r = rewards.defense_empty_penalty_reward(state, s4, 0)
        release_r = rewards.release_bonus_reward(
            state, actions[0].src_idx, valid_p0, ships_p0
        )
        emit_log_r = rewards.emit_log_reward_gated(
            valid_p0, state, actions[0].src_idx
        )
        one_ship_r = rewards.one_ship_penalty_reward(valid_p0, ships_p0)
        high_prod_r = rewards.high_prod_capture_reward(state, s4, 0)
        capture_fs_r = rewards.capture_fleet_scale_reward(state, s4, 0)
        multi_emit_r = rewards.multi_emit_gated_bonus_reward(valid_p0, ships_p0)
        hold_r = rewards.hold_bonus_reward(valid_p0, state, 0)
        anti_hoard_r = rewards.anti_hoard_penalty_reward(valid_p0, state, 0)
        non_terminal_r = (
            shaping
            + keep_r + fleet_r
            + prod_r + planet_r + fleet_log_r
            + prod_d_r + capture_r + defense_empty_r + emit_log_r + release_r
            + one_ship_r + high_prod_r + capture_fs_r + multi_emit_r + hold_r
            + anti_hoard_r
        )
        reward_p0 = jnp.where(done_now, terminal_r, non_terminal_r)

        s5 = s4.replace(step=next_step, done=done_now)
        s5 = update_global_hist(s5, self.episode_steps, self.wins_needed)

        out = EnvOutput(
            reward=reward_p0,
            done=done_now,
            info_my_ships=rewards.player_total_ships(s5, 0).astype(jnp.float32),
            info_opp_ships=rewards._strongest_opp_ships(s5, 0).astype(jnp.float32),
        )
        return s5, out

    def step_and_autoreset(
        self,
        state: EnvState,
        actions: tuple[MultiPlayerAction, MultiPlayerAction],
        reset_rng: jnp.ndarray,
    ) -> tuple[EnvState, EnvOutput]:
        """Like ``step`` but resets when done. Used inside rollouts.

        With multi-match series (wins_needed > 1):
        - On match end: determine match winner, update match_score.
        - If a player reaches ``wins_needed``, series is over → full reset
          with a new map.
        - Otherwise, reset dynamics only (same map) for the next match.

        With wins_needed=1 (default), every match end is a series end,
        preserving legacy single-match behaviour.
        """
        next_state, out = self.step(state, actions)

        # Determine per-player match result for score tracking.
        # terminal_reward is from player 0's perspective: +1 win, -1 loss, 0 tie.
        # We need a bool winner mask for both players.
        tr = rewards.terminal_reward(next_state.replace(step=next_state.step), 0)
        p0_won = tr > 0
        p0_lost = tr < 0
        # Ties: both get a point (prevents infinite series).
        is_tie = jnp.equal(tr, 0.0) & out.done
        winner_mask = jnp.stack([
            (p0_won | is_tie).astype(jnp.int32),
            (p0_lost | is_tie).astype(jnp.int32),
        ])

        new_score = next_state.match_score + jnp.where(
            out.done, winner_mask, jnp.zeros_like(winner_mask)
        )
        series_done = out.done & (new_score.max() >= self.wins_needed)
        match_only = out.done & ~series_done

        # Branch 1: series done → full reset with new map
        fresh = init.reset(reset_rng, num_groups=self.num_groups)
        # Branch 2: match done but series continues → same-map reset
        match_reset = init.reset_match_same_map(next_state, winner_mask)

        # Pick: series_done → fresh, match_only → match_reset, else → next_state
        chosen = jax.tree_util.tree_map(
            lambda f, m, n: jnp.where(series_done, f, jnp.where(match_only, m, n)),
            fresh, match_reset, next_state,
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
