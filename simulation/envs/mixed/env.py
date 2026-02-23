"""MixedEnvironment — concrete implementation of the Mixed archetype.

Implements the BaseEnvironment contract:
  reset(seed) -> initial observations
  step(actions) -> dict[AgentID, StepResult]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simulation.config.schema import MixedEnvironmentConfig
from simulation.core.base_env import BaseEnvironment
from simulation.core.seeding import make_rng
from simulation.core.types import AgentID, StepResult, TerminationReason
from simulation.envs.mixed.actions import Action, ActionType
from simulation.envs.mixed.rewards import compute_rewards
from simulation.envs.mixed.state import AgentState, GlobalState
from simulation.envs.mixed.termination import check_termination
from simulation.envs.mixed.transition import resolve_actions


class MixedEnvironment(BaseEnvironment):
    """Mixed-archetype environment with cooperative/competitive dynamics."""

    def __init__(self, config: MixedEnvironmentConfig) -> None:
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

        self._state = GlobalState(
            shared_pool=pop.initial_shared_pool,
            step=0,
            agents={
                aid: AgentState(agent_id=aid, resources=pop.initial_agent_resources)
                for aid in agent_ids
            },
        )

        mem = self._config.agents.observation_memory_steps
        return {aid: self._state.to_observation(aid, mem) for aid in agent_ids}

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
        prev_resources = {aid: state.agents[aid].resources for aid in parsed}
        prev_pool = state.shared_pool

        # Resolve actions (mutates state)
        action_meta = resolve_actions(state, parsed, config, self._rng)

        # Compute rewards
        reward_results = compute_rewards(
            state, prev_resources, prev_pool, action_meta, config
        )

        # Record history
        mem_depth = config.layers.temporal_memory_depth
        for aid, (reward, _) in reward_results.items():
            state.agents[aid].record(parsed[aid], reward, mem_depth)

        # Advance step counter
        state.step += 1

        # Check termination
        self._termination = check_termination(state, config)
        self._done = self._termination is not None

        # Build StepResults for active agents (at time of action)
        mem = config.agents.observation_memory_steps
        results: dict[AgentID, StepResult] = {}
        for aid in action_meta:
            reward, components = reward_results[aid]
            results[aid] = StepResult(
                observation=state.to_observation(aid, mem),
                reward=reward,
                done=self._done,
                info={"reward_components": components},
            )

        return results

    # ------------------------------------------------------------------
    # Specs
    # ------------------------------------------------------------------

    def observation_spec(self) -> dict[str, Any]:
        return {
            "step": {"type": "int", "min": 0},
            "shared_pool": {"type": "float", "min": 0.0},
            "own_resources": {"type": "float", "min": 0.0},
            "num_active_agents": {"type": "int", "min": 0},
            "cooperation_scores": {"type": "dict", "value_type": "float"},
            "action_history": {
                "type": "list",
                "item": {"type": {"type": "str"}, "amount": {"type": "float"}},
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
                "description": "Required for cooperate/extract, ignored otherwise.",
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
