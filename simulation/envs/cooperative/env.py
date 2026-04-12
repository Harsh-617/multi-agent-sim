"""CooperativeEnvironment — concrete implementation of the Cooperative archetype.

Implements the BaseEnvironment contract:
  reset(seed) -> initial observations for all agents
  step(actions) -> dict[AgentID, StepResult]

Observation vector (flat np.float32 array, fixed length per config):
  [backlog_norm, queue_norm×T, system_stress, group_completion_rate,
   group_contrib_norm, episode_progress, effort_capacity_norm,
   own_contrib×T, own_spec×T, own_reward_norm,
   action_history×(K*(T+1)),
   group_contrib_by_type×T, free_rider_pressure]

Total length = 8 + 4*T + K*(T+1)
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
from simulation.core.base_env import BaseEnvironment
from simulation.core.seeding import make_rng
from simulation.core.types import AgentID, StepResult, TerminationReason
from simulation.envs.cooperative.actions import Action
from simulation.envs.cooperative.rewards import compute_rewards
from simulation.envs.cooperative.state import AgentState, GlobalState, RelationalState
from simulation.envs.cooperative.termination import check_termination
from simulation.envs.cooperative.transition import _broadcast, resolve_step


class CooperativeEnvironment(BaseEnvironment):
    """Cooperative-archetype environment: shared task queue, collective reward."""

    def __init__(self, config: CooperativeEnvironmentConfig) -> None:
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
        task = self._config.task
        layers = self._config.layers
        T = pop.num_task_types
        N = pop.num_agents

        # Draw per-type task difficulty with episode-level variance
        base_difficulties = _broadcast(task.task_difficulty, T)
        variance = layers.task_difficulty_variance
        difficulties: list[float] = []
        for d in base_difficulties:
            perturbed = d * (1.0 + self._rng.uniform(-variance, variance))
            difficulties.append(max(0.01, perturbed))

        # Build initial task queue from initial_backlog, distributed evenly
        per_type = task.initial_backlog // T
        remainder = task.initial_backlog % T
        initial_queue = [per_type + (1 if i < remainder else 0) for i in range(T)]

        agent_ids = [f"agent_{i}" for i in range(N)]

        rel = RelationalState(
            group_contribution_by_type=[0.0] * T,
            completion_rate_history=deque(maxlen=layers.history_window),
        )

        agents = {
            aid: AgentState(
                agent_id=aid,
                specialization_score=[0.0] * T,
                contribution_history=deque(maxlen=layers.history_window),
                cumulative_reward=0.0,
                steps_active=0,
                last_effort=[0.0] * T,
            )
            for aid in agent_ids
        }

        self._state = GlobalState(
            task_queue=initial_queue,
            task_difficulty=difficulties,
            tasks_completed_this_step=[0] * T,
            tasks_completed_total=[0] * T,
            tasks_arrived_this_step=[0] * T,
            backlog_level=sum(initial_queue),
            system_stress=min(sum(initial_queue) / task.collapse_threshold, 1.0),
            step=0,
            agents=agents,
            relational=rel,
        )

        return {aid: self._build_observation(aid) for aid in agent_ids}

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
        agent_ids = state.agent_ids()

        # Coerce raw actions — accept Action objects or dicts
        parsed: dict[AgentID, Action] = {}
        for aid in agent_ids:
            raw = actions.get(aid)
            if raw is None:
                parsed[aid] = Action(task_type=None)  # default IDLE
            elif isinstance(raw, Action):
                parsed[aid] = raw
            elif isinstance(raw, dict):
                parsed[aid] = Action(
                    task_type=raw.get("task_type"),
                    effort_amount=raw.get("effort_amount", 0.0),
                )
            else:
                raise TypeError(f"Invalid action for {aid}: {raw!r}")

        # Resolve transition (mutates state)
        action_meta = resolve_step(state, parsed, config, self._rng)

        # Compute rewards
        reward_results = compute_rewards(state, action_meta, config)

        # Update per-agent cumulative reward
        for aid, (reward, _) in reward_results.items():
            state.agents[aid].cumulative_reward += reward

        # Advance step counter
        state.step += 1

        # Check termination (also updates sustain window counters)
        self._termination = check_termination(state, config)
        self._done = self._termination is not None

        # Build StepResults
        results: dict[AgentID, StepResult] = {}
        for aid in agent_ids:
            reward, components = reward_results[aid]
            results[aid] = StepResult(
                observation=self._build_observation(aid),
                reward=reward,
                done=self._done,
                info={"reward_components": components},
            )

        return results

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self, agent_id: AgentID) -> dict[str, Any]:
        """Build the fixed-length observation vector for one agent."""
        state = self._state
        config = self._config
        pop = config.population
        layers = config.layers
        task = config.task
        T = pop.num_task_types
        K = layers.history_window
        capacity = pop.agent_effort_capacity
        collapse_thresh = task.collapse_threshold
        agent = state.agents[agent_id]
        rel = state.relational

        # --- Public observation (same for all agents, optionally noisy) ---
        backlog_norm = min(state.backlog_level / collapse_thresh, 1.0)
        queue_norm = [
            min(state.task_queue[t] / collapse_thresh, 1.0) for t in range(T)
        ]
        system_stress = state.system_stress
        group_completion = rel.group_completion_rate
        # Normalise by N * capacity (max possible contribution per step)
        max_contrib = pop.num_agents * capacity
        group_contrib_norm = (
            min(rel.group_contribution_last_step / max_contrib, 1.0)
            if max_contrib > 0 else 0.0
        )
        episode_progress = state.step / pop.max_steps

        # Inject observation noise into public components
        noise_scale = layers.observation_noise
        if noise_scale > 0.0:
            noise = self._rng.normal(0, noise_scale, size=T + 5).tolist()
            backlog_norm = float(np.clip(backlog_norm + noise[0], 0.0, 1.0))
            queue_norm = [
                float(np.clip(queue_norm[t] + noise[1 + t], 0.0, 1.0))
                for t in range(T)
            ]
            system_stress = float(np.clip(system_stress + noise[T + 1], 0.0, 1.0))
            group_completion = float(np.clip(group_completion + noise[T + 2], 0.0, 1.0))
            group_contrib_norm = float(np.clip(group_contrib_norm + noise[T + 3], 0.0, 1.0))

        # --- Private self observation ---
        effort_capacity_norm = 1.0  # always full; capacity resets each step
        own_contrib = [
            min(agent.last_effort[t] / capacity, 1.0) if capacity > 0 else 0.0
            for t in range(T)
        ]
        own_spec = list(agent.specialization_score)  # already in [0, 1]
        max_reward = pop.max_steps  # loose upper bound for normalisation
        own_reward_norm = min(agent.cumulative_reward / max(max_reward, 1.0), 1.0)
        own_reward_norm = max(0.0, own_reward_norm)

        # Action history: last K steps, each encoded as (T-dim one-hot + effort)
        history_arr = np.zeros((K, T + 1), dtype=np.float32)
        history = list(agent.contribution_history)  # oldest first
        for i, effort_record in enumerate(history[-K:]):
            row_idx = K - len(history[-K:]) + i  # right-align into array
            # Find which type had effort (at most one, since single-type actions)
            max_t = int(np.argmax(effort_record)) if any(e > 0 for e in effort_record) else -1
            if max_t >= 0:
                history_arr[row_idx, max_t] = 1.0
                history_arr[row_idx, T] = float(effort_record[max_t])
            # else: all zeros (IDLE step)

        # --- Partial social observation ---
        max_type_contrib = pop.num_agents * capacity
        group_by_type_norm = [
            min(rel.group_contribution_by_type[t] / max_type_contrib, 1.0)
            if max_type_contrib > 0 else 0.0
            for t in range(T)
        ]
        free_rider_pressure = (
            rel.free_rider_pressure * layers.free_rider_pressure_scale
        )
        free_rider_pressure = float(np.clip(free_rider_pressure, 0.0, 1.0))

        # --- Assemble flat vector ---
        parts: list[float] = (
            [backlog_norm]
            + queue_norm
            + [system_stress, group_completion, group_contrib_norm, episode_progress,
               effort_capacity_norm]
            + own_contrib
            + own_spec
            + [own_reward_norm]
            + history_arr.flatten().tolist()
            + group_by_type_norm
            + [free_rider_pressure]
        )
        obs_vector = np.array(parts, dtype=np.float32)

        return {
            "obs_vector": obs_vector,
            "step": state.step,
        }

    def obs_dim(self) -> int:
        """Return the fixed observation vector length for this config."""
        T = self._config.population.num_task_types
        K = self._config.layers.history_window
        return 8 + 4 * T + K * (T + 1)

    # ------------------------------------------------------------------
    # Specs
    # ------------------------------------------------------------------

    def observation_spec(self) -> dict[str, Any]:
        T = self._config.population.num_task_types
        K = self._config.layers.history_window
        dim = 8 + 4 * T + K * (T + 1)
        return {
            "obs_vector": {"type": "float32", "shape": (dim,), "min": 0.0, "max": 1.0},
            "step": {"type": "int", "min": 0},
        }

    def action_spec(self) -> dict[str, Any]:
        T = self._config.population.num_task_types
        return {
            "task_type": {
                "type": "int | None",
                "values": list(range(T)) + [None],
                "description": "None = IDLE; int = task type index",
            },
            "effort_amount": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
            },
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def active_agents(self) -> list[AgentID]:
        if self._state is None:
            return []
        return self._state.agent_ids()

    def is_done(self) -> bool:
        return self._done

    def termination_reason(self) -> TerminationReason | None:
        return self._termination

    @property
    def current_step(self) -> int:
        if self._state is None:
            return 0
        return self._state.step
