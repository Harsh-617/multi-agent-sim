"""PettingZoo ParallelEnv adapter for CompetitiveEnvironment.

Thin wrapper that translates between our CompetitiveEnvironment interface
and the PettingZoo ParallelEnv API (gymnasium spaces).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from simulation.config.competitive_schema import CompetitiveEnvironmentConfig
from simulation.core.types import TerminationReason
from simulation.envs.competitive.actions import Action, ActionType
from simulation.envs.competitive.env import CompetitiveEnvironment

# Canonical ordering of ActionType values for Discrete mapping.
ACTION_TYPE_ORDER: list[ActionType] = [
    ActionType.BUILD,
    ActionType.ATTACK,
    ActionType.DEFEND,
    ActionType.GAMBLE,
]
_IDX_TO_ACTION = {i: at for i, at in enumerate(ACTION_TYPE_ORDER)}
_AMOUNT_TYPES = {ActionType.BUILD, ActionType.ATTACK}


class CompetitivePettingZooParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv wrapper around CompetitiveEnvironment."""

    metadata = {"render_modes": [], "name": "competitive_v0"}

    def __init__(self, config: CompetitiveEnvironmentConfig) -> None:
        super().__init__()
        self._config = config
        self._env = CompetitiveEnvironment(config)
        self._num_agents = config.population.num_agents
        self._mem_steps = config.agents.observation_memory_steps
        self._obs_window = config.layers.opponent_obs_window

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
        obs_window = self._obs_window
        return spaces.Dict(
            {
                "step": spaces.Box(
                    low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
                ),
                "own_score": spaces.Box(
                    low=0.0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32
                ),
                "own_resources": spaces.Box(
                    low=0.0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32
                ),
                "own_rank": spaces.Box(
                    low=1, high=n, shape=(1,), dtype=np.int32
                ),
                "num_active_agents": spaces.Box(
                    low=0, high=n, shape=(1,), dtype=np.int32
                ),
                "opponents_scores": spaces.Box(
                    low=0.0,
                    high=np.finfo(np.float32).max,
                    shape=(n - 1,),
                    dtype=np.float32,
                ),
                "own_action_history": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(mem, 5),
                    dtype=np.float32,
                ),
                "opponents_recent_actions": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(n - 1, obs_window, 4),
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

    def _convert_obs(self, raw_obs: dict[str, Any], agent_id: str) -> dict[str, Any]:
        """Convert a raw env observation dict into gymnasium-compatible arrays."""
        n = self._num_agents
        mem = self._mem_steps
        obs_window = self._obs_window

        # Own rank from rankings (1-indexed)
        rankings = raw_obs.get("rankings", [])
        own_rank = n  # default to last
        for rank_idx, (aid, _score) in enumerate(rankings):
            if aid == agent_id:
                own_rank = rank_idx + 1
                break

        # Opponents scores: pad to fixed length (n-1)
        opponents = raw_obs.get("opponents", {})
        opp_scores = np.zeros(n - 1, dtype=np.float32)
        for i, (opp_id, opp_data) in enumerate(opponents.items()):
            if i >= n - 1:
                break
            opp_scores[i] = opp_data["score"]

        # Own action history: encode each entry as [onehot(4), amount]
        hist_arr = np.zeros((mem, 5), dtype=np.float32)
        for i, entry in enumerate(raw_obs.get("action_history", [])):
            if i >= mem:
                break
            type_str = entry["type"]
            for j, at in enumerate(ACTION_TYPE_ORDER):
                if at.value == type_str:
                    hist_arr[i, j] = 1.0
                    break
            hist_arr[i, 4] = entry["amount"]

        # Opponents recent actions: (n-1, obs_window, 4) one-hot
        opp_actions = np.zeros((n - 1, obs_window, 4), dtype=np.float32)
        for i, (opp_id, opp_data) in enumerate(opponents.items()):
            if i >= n - 1:
                break
            for j, act_str in enumerate(opp_data.get("recent_actions", [])):
                if j >= obs_window:
                    break
                for k, at in enumerate(ACTION_TYPE_ORDER):
                    if at.value == act_str:
                        opp_actions[i, j, k] = 1.0
                        break

        return {
            "step": np.array([raw_obs.get("step", 0)], dtype=np.int32),
            "own_score": np.array(
                [raw_obs.get("own_score", 0.0)], dtype=np.float32
            ),
            "own_resources": np.array(
                [raw_obs.get("own_resources", 0.0)], dtype=np.float32
            ),
            "own_rank": np.array([own_rank], dtype=np.int32),
            "num_active_agents": np.array(
                [raw_obs.get("num_active_agents", 0)], dtype=np.int32
            ),
            "opponents_scores": opp_scores,
            "own_action_history": hist_arr,
            "opponents_recent_actions": opp_actions,
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

        observations = {a: self._convert_obs(raw_obs[a], a) for a in self.agents}
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
        active_set = set(self._env.active_agents())

        observations: dict[str, dict] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict] = {}

        for aid in self.agents:
            if aid in step_results:
                sr = step_results[aid]
                observations[aid] = self._convert_obs(sr.observation, aid)
                rewards[aid] = sr.reward
                infos[aid] = sr.info

                if is_done:
                    terminations[aid] = True
                    truncations[aid] = reason == TerminationReason.MAX_STEPS
                elif aid not in active_set:
                    # Agent eliminated this step, episode continues
                    terminations[aid] = True
                    truncations[aid] = False
                else:
                    terminations[aid] = False
                    truncations[aid] = False
            else:
                # Agent was eliminated before this step
                observations[aid] = self._convert_obs(
                    self._make_zero_obs(), aid
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
            "own_score": 0.0,
            "own_resources": 0.0,
            "num_active_agents": 0,
            "rankings": [],
            "action_history": [],
            "opponents": {},
        }

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
