"""CompetitiveEnvironment — concrete implementation of the Competitive archetype.

Implements the BaseEnvironment contract:
  reset(seed) -> initial observations
  step(actions) -> dict[AgentID, StepResult]

Terminal bonus is issued at episode end to ALL agents (active + eliminated)
based on final score rankings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simulation.config.competitive_schema import CompetitiveEnvironmentConfig
from simulation.core.base_env import BaseEnvironment
from simulation.core.seeding import make_rng
from simulation.core.types import AgentID, StepResult, TerminationReason
from simulation.envs.competitive.actions import Action, ActionType
from simulation.envs.competitive.rewards import compute_rewards
from simulation.envs.competitive.state import AgentState, GlobalState, OpponentHistoryState
from simulation.envs.competitive.termination import check_termination
from simulation.envs.competitive.transition import resolve_actions


class CompetitiveEnvironment(BaseEnvironment):
    """Competitive-archetype environment with elimination and score-based rankings."""

    def __init__(self, config: CompetitiveEnvironmentConfig) -> None:
        self._config = config
        self._state: GlobalState | None = None
        self._rng: np.random.Generator | None = None
        self._done = False
        self._termination: TerminationReason | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> dict[AgentID, dict[str, Any]]:
        effective_seed = seed if seed is not None else self._config.identity.seed
        self._rng = make_rng(effective_seed)
        self._done = False
        self._termination = None

        pop = self._config.population
        agent_ids = [f"agent_{i}" for i in range(pop.num_agents)]

        # Build empty opponent histories for all pairs
        opponent_histories = {}
        for i, a in enumerate(agent_ids):
            for b in agent_ids[i + 1 :]:
                opponent_histories[(a, b)] = OpponentHistoryState()

        self._state = GlobalState(
            step=0,
            agents={
                aid: AgentState(
                    agent_id=aid,
                    score=pop.initial_score,
                    resources=pop.initial_resources,
                    active=True,
                )
                for aid in agent_ids
            },
            opponent_histories=opponent_histories,
        )

        obs_window = self._config.layers.opponent_obs_window
        mem = self._config.agents.observation_memory_steps
        return {
            aid: self._state.to_observation(aid, obs_window, mem)
            for aid in agent_ids
        }

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: dict[AgentID, Any]
    ) -> dict[AgentID, StepResult]:
        if self._state is None:
            raise RuntimeError("Must call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        state = self._state
        config = self._config

        # Coerce raw actions — accept Action objects or dicts
        parsed: dict[AgentID, Action] = {}
        for aid in state.active_agent_ids():
            raw = actions.get(aid)
            if raw is None:
                parsed[aid] = Action(type=ActionType.DEFEND)
            elif isinstance(raw, Action):
                parsed[aid] = raw
            elif isinstance(raw, dict):
                parsed[aid] = Action(
                    type=ActionType(raw["type"]),
                    amount=raw.get("amount", 0.0),
                )
            else:
                raise TypeError(f"Invalid action for {aid}: {raw!r}")

        # Snapshot pre-transition state for reward computation
        prev_scores = {aid: state.agents[aid].score for aid in parsed}
        prev_resources = {aid: state.agents[aid].resources for aid in parsed}

        # Resolve actions (mutates state)
        action_meta = resolve_actions(state, parsed, config, self._rng)

        # Compute rewards
        reward_results = compute_rewards(
            state, prev_scores, prev_resources, action_meta, config
        )

        # Record history
        mem_depth = config.layers.opponent_history_depth
        for aid, (reward, _) in reward_results.items():
            state.agents[aid].record(parsed[aid], reward, mem_depth)

        # Advance step counter
        state.step += 1

        # Check termination
        self._termination = check_termination(state, config)
        self._done = self._termination is not None

        # Terminal bonus: issued to ALL agents at episode end
        terminal_bonuses: dict[AgentID, float] = {}
        if self._done and self._termination in (
            TerminationReason.ELIMINATION,
            TerminationReason.MAX_STEPS,
            TerminationReason.NO_ACTIVE_AGENTS,
        ):
            terminal_bonuses = self._compute_terminal_bonuses()
            # Add terminal bonus to each agent's reward
            for aid in reward_results:
                scalar, components = reward_results[aid]
                bonus = terminal_bonuses.get(aid, 0.0)
                reward_results[aid] = (scalar + bonus, components)

        # Build StepResults for agents that acted this step
        obs_window = config.layers.opponent_obs_window
        mem = config.agents.observation_memory_steps
        results: dict[AgentID, StepResult] = {}
        for aid in action_meta:
            reward, components = reward_results[aid]
            info: dict[str, Any] = {"reward_components": components}
            if aid in terminal_bonuses:
                info["terminal_bonus"] = terminal_bonuses[aid]
            results[aid] = StepResult(
                observation=state.to_observation(aid, obs_window, mem),
                reward=reward,
                done=self._done,
                info=info,
            )

        return results

    # ------------------------------------------------------------------
    # Terminal bonus
    # ------------------------------------------------------------------

    def _compute_terminal_bonuses(self) -> dict[AgentID, float]:
        """Rank-based terminal bonus for ALL agents (active + eliminated)."""
        scale = self._config.rewards.terminal_bonus_scale
        rankings = self._state.rankings()  # sorted by score descending
        n = len(rankings)

        bonuses: dict[AgentID, float] = {}
        if n <= 1:
            for aid, _ in rankings:
                bonuses[aid] = scale
        else:
            for rank_idx, (aid, _score) in enumerate(rankings):
                # rank_idx 0 = 1st place (highest score)
                bonuses[aid] = scale * (1.0 - rank_idx / (n - 1))

        return bonuses

    # ------------------------------------------------------------------
    # Specs
    # ------------------------------------------------------------------

    def obs_dim(self) -> int:
        """Return the flattened observation dimension this env produces.

        Layout (matches competitive_ppo_agent._flatten_obs + _OBS_KEYS):
          5 scalars (step, own_score, own_resources, own_rank, num_active_agents)
          + (num_agents - 1) opponents_scores
          + observation_memory_steps * 5  (4-hot action type + amount per step)
          + (num_agents - 1) * opponent_obs_window * 4  (opponents_recent_actions)
        """
        n = self._config.population.num_agents
        n_opp = n - 1
        mem = self._config.agents.observation_memory_steps
        window = self._config.layers.opponent_obs_window
        n_action_types = 4  # BUILD, ATTACK, DEFEND, GAMBLE
        return 5 + n_opp + mem * (n_action_types + 1) + n_opp * window * n_action_types

    def observation_spec(self) -> dict[str, Any]:
        return {
            "step": {"type": "int", "min": 0},
            "own_score": {"type": "float", "min": 0.0},
            "own_resources": {"type": "float", "min": 0.0},
            "num_active_agents": {"type": "int", "min": 0},
            "rankings": {
                "type": "list",
                "item": {"agent_id": {"type": "str"}, "score": {"type": "float"}},
            },
            "action_history": {
                "type": "list",
                "item": {"type": {"type": "str"}, "amount": {"type": "float"}},
            },
            "opponents": {
                "type": "dict",
                "value": {
                    "score": {"type": "float"},
                    "recent_actions": {"type": "list", "item": {"type": "str"}},
                },
            },
        }

    def action_spec(self) -> dict[str, Any]:
        return {
            "type": {
                "type": "enum",
                "values": [t.value for t in ActionType],
            },
            "amount": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Required for build/attack, ignored otherwise.",
            },
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def active_agents(self) -> list[AgentID]:
        if self._state is None:
            return []
        return self._state.active_agent_ids()

    def is_done(self) -> bool:
        return self._done

    def termination_reason(self) -> TerminationReason | None:
        return self._termination

    @property
    def current_step(self) -> int:
        if self._state is None:
            return 0
        return self._state.step
