"""PettingZoo ParallelEnv adapter for CooperativeEnvironment.

Thin wrapper that translates between CooperativeEnvironment and the PettingZoo
ParallelEnv API (gymnasium spaces).

The adapter only translates — it never modifies reward semantics or injects
hidden state (ADR-002, ADR-011 pattern).

Observation space: Box(low=0, high=1, shape=(obs_dim,), dtype=float32)
Action space:      Dict(task_type: Discrete(T+1), effort_amount: Box([0,1]))
  where index T in Discrete means IDLE (all indices 0..T-1 map to task types).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
from simulation.core.types import TerminationReason
from simulation.envs.cooperative.actions import Action
from simulation.envs.cooperative.env import CooperativeEnvironment


class CooperativePettingZooParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv wrapper around CooperativeEnvironment."""

    metadata = {"render_modes": [], "name": "cooperative_v0"}

    def __init__(self, config: CooperativeEnvironmentConfig) -> None:
        super().__init__()
        self._config = config
        self._env = CooperativeEnvironment(config)
        self._num_agents = config.population.num_agents
        self._T = config.population.num_task_types  # number of task types

        # obs_dim = 6 + 4*T + K*(T+1)
        K = config.layers.history_window
        self._obs_dim = 8 + 4 * self._T + K * (self._T + 1)

        self.possible_agents: list[str] = [
            f"agent_{i}" for i in range(self._num_agents)
        ]
        self.agents: list[str] = []

        self._action_spaces = {
            agent: self._build_action_space() for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: self._build_observation_space() for agent in self.possible_agents
        }

    # ------------------------------------------------------------------
    # Space definitions
    # ------------------------------------------------------------------

    def _build_action_space(self) -> spaces.Dict:
        # task_type index T means IDLE; 0..T-1 mean task types
        return spaces.Dict(
            {
                "task_type": spaces.Discrete(self._T + 1),
                "effort_amount": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )

    def _build_observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    def observation_space(self, agent: str) -> spaces.Box:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Dict:
        return self._action_spaces[agent]

    # ------------------------------------------------------------------
    # Observation conversion
    # ------------------------------------------------------------------

    def _convert_obs(self, raw_obs: dict[str, Any]) -> np.ndarray:
        """Extract the flat obs_vector from a raw environment observation."""
        vec = raw_obs.get("obs_vector")
        if vec is None:
            return np.zeros(self._obs_dim, dtype=np.float32)
        return np.asarray(vec, dtype=np.float32)

    # ------------------------------------------------------------------
    # Action conversion
    # ------------------------------------------------------------------

    def _convert_action(self, gym_action: dict[str, Any]) -> Action:
        """Convert a gymnasium Dict action to our Action dataclass."""
        idx = int(gym_action["task_type"])
        effort = float(gym_action["effort_amount"].flat[0])
        if idx == self._T:  # IDLE sentinel
            return Action(task_type=None)
        return Action(task_type=idx, effort_amount=effort)

    # ------------------------------------------------------------------
    # ParallelEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        raw_obs = self._env.reset(seed=seed)
        self.agents = list(raw_obs.keys())

        observations = {a: self._convert_obs(raw_obs[a]) for a in self.agents}
        infos: dict[str, dict] = {a: {} for a in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        env_actions = {
            aid: self._convert_action(act) for aid, act in actions.items()
        }

        step_results = self._env.step(env_actions)
        is_done = self._env.is_done()
        reason = self._env.termination_reason()

        observations: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict] = {}

        for aid in self.agents:
            if aid in step_results:
                sr = step_results[aid]
                observations[aid] = self._convert_obs(sr.observation)
                rewards[aid] = sr.reward
                infos[aid] = sr.info
                terminations[aid] = is_done
                # Truncation = episode ended by time limit (max_steps or clearance)
                truncations[aid] = reason in (
                    TerminationReason.MAX_STEPS,
                    TerminationReason.PERFECT_CLEARANCE,
                )
            else:
                observations[aid] = np.zeros(self._obs_dim, dtype=np.float32)
                rewards[aid] = 0.0
                terminations[aid] = True
                truncations[aid] = False
                infos[aid] = {}

        if is_done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
