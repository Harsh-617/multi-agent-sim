"""Unit tests for CooperativePettingZooParallelEnv adapter.

Verifies the PettingZoo ParallelEnv contract and correct obs/action translation
for the Cooperative archetype adapter (simulation/adapters/cooperative_pettingzoo.py).
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from simulation.adapters.cooperative_pettingzoo import CooperativePettingZooParallelEnv
from simulation.config.cooperative_defaults import default_cooperative_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def env() -> CooperativePettingZooParallelEnv:
    cfg = default_cooperative_config(seed=42)
    return CooperativePettingZooParallelEnv(cfg)


@pytest.fixture()
def reset_env(env: CooperativePettingZooParallelEnv):
    """Return an env that has already been reset (agents list is populated)."""
    env.reset(seed=42)
    return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCooperativePettingZooAdapter:

    def test_reset_returns_correct_obs_shape(self, env: CooperativePettingZooParallelEnv):
        """reset() must return one flat float32 array of shape (obs_dim,) per agent."""
        observations, infos = env.reset(seed=42)
        expected_dim = env._obs_dim

        assert len(observations) == env._num_agents, (
            f"Expected {env._num_agents} observations, got {len(observations)}"
        )
        for agent_id in env.possible_agents:
            assert agent_id in observations, f"{agent_id} missing from observations"
            obs = observations[agent_id]
            assert obs.shape == (expected_dim,), (
                f"{agent_id}: expected shape ({expected_dim},), got {obs.shape}"
            )
            assert obs.dtype == np.float32, (
                f"{agent_id}: expected float32, got {obs.dtype}"
            )

    def test_step_returns_obs_rewards_terminations_truncations_infos(
        self, env: CooperativePettingZooParallelEnv
    ):
        """step() must return the full 5-tuple with correct types and agent keys."""
        observations, _ = env.reset(seed=42)
        T = env._T
        actions = {
            aid: {
                "task_type": 0,
                "effort_amount": np.array([0.5], dtype=np.float32),
            }
            for aid in env.agents
        }
        result = env.step(actions)

        assert len(result) == 5, "step() must return a 5-tuple"
        next_obs, rewards, terminations, truncations, infos = result

        assert isinstance(next_obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)
        assert isinstance(infos, dict)

        for aid in env.possible_agents:
            assert aid in next_obs
            assert aid in rewards
            assert aid in terminations
            assert aid in truncations
            assert isinstance(rewards[aid], (int, float))
            assert isinstance(terminations[aid], bool)
            assert isinstance(truncations[aid], bool)

    def test_obs_space_matches_env_observation_vector_dimension(
        self, env: CooperativePettingZooParallelEnv
    ):
        """observation_space shape must equal 8 + 4*T + K*(T+1)."""
        cfg = env._config
        T = cfg.population.num_task_types
        K = cfg.layers.history_window
        expected_dim = 8 + 4 * T + K * (T + 1)

        assert env._obs_dim == expected_dim, (
            f"Adapter _obs_dim {env._obs_dim} != formula {expected_dim}"
        )
        for agent_id in env.possible_agents:
            space = env.observation_space(agent_id)
            assert isinstance(space, spaces.Box)
            assert space.shape == (expected_dim,), (
                f"{agent_id}: obs space shape {space.shape} != ({expected_dim},)"
            )

    def test_action_space_matches_env_action_spec(
        self, env: CooperativePettingZooParallelEnv
    ):
        """Action space must have task_type: Discrete(T+1) and effort_amount: Box([0,1])."""
        T = env._config.population.num_task_types

        for agent_id in env.possible_agents:
            action_space = env.action_space(agent_id)
            assert isinstance(action_space, spaces.Dict)
            assert "task_type" in action_space.spaces
            assert "effort_amount" in action_space.spaces

            task_type_space = action_space["task_type"]
            assert isinstance(task_type_space, spaces.Discrete)
            assert task_type_space.n == T + 1, (
                f"Expected Discrete({T + 1}), got Discrete({task_type_space.n})"
            )

            effort_space = action_space["effort_amount"]
            assert isinstance(effort_space, spaces.Box)
            assert effort_space.shape == (1,)
            assert float(effort_space.low[0]) == pytest.approx(0.0)
            assert float(effort_space.high[0]) == pytest.approx(1.0)

    def test_handles_all_idle_actions_without_error(
        self, env: CooperativePettingZooParallelEnv
    ):
        """IDLE actions (task_type = T) must not raise errors."""
        env.reset(seed=42)
        T = env._T  # IDLE sentinel index
        actions = {
            aid: {
                "task_type": T,
                "effort_amount": np.array([0.0], dtype=np.float32),
            }
            for aid in env.agents
        }
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # All agents should still receive observations
        for aid in env.possible_agents:
            assert aid in next_obs
            assert aid in rewards
            assert isinstance(rewards[aid], (int, float))

    def test_handles_all_work_actions_without_error(
        self, env: CooperativePettingZooParallelEnv
    ):
        """Full-effort work actions on task type 0 must not raise errors."""
        env.reset(seed=42)
        actions = {
            aid: {
                "task_type": 0,
                "effort_amount": np.array([1.0], dtype=np.float32),
            }
            for aid in env.agents
        }
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        for aid in env.possible_agents:
            assert aid in rewards
            # Rewards for work actions should be >= 0 (cooperative env guarantee)
            assert rewards[aid] >= 0.0

    def test_episode_runs_to_completion_with_random_actions(
        self, env: CooperativePettingZooParallelEnv
    ):
        """A full episode with random actions must terminate without error."""
        rng = np.random.default_rng(seed=7)
        observations, _ = env.reset(seed=7)
        T = env._T
        max_steps = env._config.population.max_steps + 20  # buffer past max

        steps = 0
        while env.agents and steps < max_steps:
            actions = {
                aid: {
                    "task_type": int(rng.integers(0, T + 1)),
                    "effort_amount": np.array(
                        [float(rng.uniform(0.0, 1.0))], dtype=np.float32
                    ),
                }
                for aid in env.agents
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            steps += 1

        assert steps > 0, "Episode produced zero steps"
        assert not env.agents, (
            "env.agents should be empty after episode ends "
            f"(took {steps} steps, max_steps={env._config.population.max_steps})"
        )

    def test_parallel_env_api_contract(self, env: CooperativePettingZooParallelEnv):
        """All attributes required by the PettingZoo ParallelEnv API must be present."""
        required_attrs = [
            "possible_agents",
            "agents",
            "observation_space",
            "action_space",
            "reset",
            "step",
            "render",
            "close",
            "metadata",
        ]
        for attr in required_attrs:
            assert hasattr(env, attr), f"Missing required PettingZoo attribute: {attr}"

        assert isinstance(env.possible_agents, list)
        assert len(env.possible_agents) >= 2, "Need at least 2 agents"
        assert callable(env.observation_space)
        assert callable(env.action_space)
        assert callable(env.reset)
        assert callable(env.step)
        assert isinstance(env.metadata, dict)
        assert "name" in env.metadata
