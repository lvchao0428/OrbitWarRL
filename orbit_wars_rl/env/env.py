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
        s1 = dynamics.launch_fleets(state, actions)
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
        reward_p0 = jnp.where(done_now, terminal_r, shaping)

        out = EnvOutput(
            reward=reward_p0,
            done=done_now,
            info_my_ships=rewards.player_total_ships(s4, 0).astype(jnp.float32),
            info_opp_ships=rewards.player_total_ships(s4, 1).astype(jnp.float32),
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
