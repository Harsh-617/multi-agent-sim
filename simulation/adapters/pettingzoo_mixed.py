"""PettingZoo ParallelEnv adapter for MixedEnvironment.

Thin wrapper that translates between our MixedEnvironment interface
and the PettingZoo ParallelEnv API (gymnasium spaces).
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from simulation.config.schema import MixedEnvironmentConfig
from simulation.core.types import TerminationReason
from simulation.envs.mixed.actions import Action, ActionType
from simulation.envs.mixed.env import MixedEnvironment

# Canonical ordering of ActionType values for Discrete mapping.
ACTION_TYPE_ORDER: list[ActionType] = [
    ActionType.COOPERATE,
    ActionType.EXTRACT,
    ActionType.DEFEND,
    ActionType.CONDITIONAL,
]
_IDX_TO_ACTION = {i: at for i, at in enumerate(ACTION_TYPE_ORDER)}
_AMOUNT_TYPES = {ActionType.COOPERATE, ActionType.EXTRACT}


class MixedPettingZooParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv wrapper around MixedEnvironment."""

    metadata = {"render_modes": [], "name": "mixed_v0"}

    def __init__(self, config: MixedEnvironmentConfig) -> None:
        super().__init__()
        self._config = config
        self._env = MixedEnvironment(config)
        self._num_agents = config.population.num_agents
        self._mem_steps = config.agents.observation_memory_steps

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
        return spaces.Dict(
            {
                "action_type": spaces.Discrete(len(ACTION_TYPE_ORDER)),
                "amount": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )

    def _build_observation_space(self) -> spaces.Dict:
        n = self._num_agents
        mem = self._mem_steps
        return spaces.Dict(
            {
                "step": spaces.Box(
                    low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
                ),
                "shared_pool": spaces.Box(
                    low=0.0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32
                ),
                "own_resources": spaces.Box(
                    low=0.0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32
                ),
                "num_active_agents": spaces.Box(
                    low=0, high=n, shape=(1,), dtype=np.int32
                ),
                # Cooperation scores flattened to fixed-length array (max n-1 peers)
                "cooperation_scores": spaces.Box(
                    low=0.0, high=1.0, shape=(n - 1,), dtype=np.float32
                ),
                # Action history: each step has (action_type_onehot[4] + amount[1]) = 5
                "action_history": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(mem, 5),
                    dtype=np.float32,
                ),
            }
        )

    def observation_space(self, agent: str) -> spaces.Dict:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Dict:
        return self._action_spaces[agent]

    # ------------------------------------------------------------------
    # Observation conversion
    # ------------------------------------------------------------------

    def _convert_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw env observation dict into gymnasium-compatible arrays."""
        n = self._num_agents
        mem = self._mem_steps

        # Cooperation scores: pad to fixed length (n-1)
        coop_dict: dict[str, float] = raw_obs["cooperation_scores"]
        coop_arr = np.zeros(n - 1, dtype=np.float32)
        for i, v in enumerate(coop_dict.values()):
            if i >= n - 1:
                break
            coop_arr[i] = v

        # Action history: encode each entry as [onehot(4), amount]
        hist_arr = np.zeros((mem, 5), dtype=np.float32)
        for i, entry in enumerate(raw_obs["action_history"]):
            if i >= mem:
                break
            type_str = entry["type"]
            for j, at in enumerate(ACTION_TYPE_ORDER):
                if at.value == type_str:
                    hist_arr[i, j] = 1.0
                    break
            hist_arr[i, 4] = entry["amount"]

        return {
            "step": np.array([raw_obs["step"]], dtype=np.int32),
            "shared_pool": np.array([raw_obs["shared_pool"]], dtype=np.float32),
            "own_resources": np.array([raw_obs["own_resources"]], dtype=np.float32),
            "num_active_agents": np.array(
                [raw_obs["num_active_agents"]], dtype=np.int32
            ),
            "cooperation_scores": coop_arr,
            "action_history": hist_arr,
        }

    # ------------------------------------------------------------------
    # Action conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_action(gym_action: dict[str, Any]) -> Action:
        """Convert a gymnasium Dict action into our Action dataclass."""
        idx = int(gym_action["action_type"])
        action_type = _IDX_TO_ACTION[idx]
        amount = float(gym_action["amount"].flat[0]) if action_type in _AMOUNT_TYPES else 0.0
        return Action(type=action_type, amount=amount)

    # ------------------------------------------------------------------
    # ParallelEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        raw_obs = self._env.reset(seed=seed)
        self.agents = list(raw_obs.keys())

        observations = {a: self._convert_obs(raw_obs[a]) for a in self.agents}
        infos: dict[str, dict] = {a: {} for a in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[
        dict[str, dict],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        # Convert gym actions to our Action dataclass
        env_actions = {
            aid: self._convert_action(act) for aid, act in actions.items()
        }

        step_results = self._env.step(env_actions)
        is_done = self._env.is_done()
        reason = self._env.termination_reason()
        is_truncated = reason == TerminationReason.MAX_STEPS

        observations: dict[str, dict] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict] = {}

        for aid in self.agents:
            if aid in step_results:
                sr = step_results[aid]
                observations[aid] = self._convert_obs(sr.observation)
                rewards[aid] = sr.reward
                terminations[aid] = sr.done
                truncations[aid] = is_truncated if sr.done else False
                infos[aid] = sr.info
            else:
                # Agent was eliminated before this step
                observations[aid] = self._convert_obs(
                    self._make_zero_obs()
                )
                rewards[aid] = 0.0
                terminations[aid] = True
                truncations[aid] = False
                infos[aid] = {}

        # Update active agents list
        if is_done:
            self.agents = []
        else:
            self.agents = [a for a in self.agents if not terminations[a]]

        return observations, rewards, terminations, truncations, infos

    def _make_zero_obs(self) -> dict[str, Any]:
        """Return a zeroed-out raw observation for eliminated agents."""
        return {
            "step": 0,
            "shared_pool": 0.0,
            "own_resources": 0.0,
            "num_active_agents": 0,
            "cooperation_scores": {},
            "action_history": [],
        }

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
