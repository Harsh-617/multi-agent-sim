"""Tests for the PettingZoo ParallelEnv adapter."""

from __future__ import annotations

import numpy as np
import pytest

from simulation.adapters.pettingzoo_mixed import MixedPettingZooParallelEnv
from simulation.config.defaults import default_config


def _small_config(seed: int = 42, max_steps: int = 200):
    """Config with defaults, overridable max_steps."""
    cfg = default_config(seed=seed)
    cfg = cfg.model_copy(
        update={"population": cfg.population.model_copy(update={"max_steps": max_steps})}
    )
    return cfg


def _random_action(env: MixedPettingZooParallelEnv, agent: str, rng: np.random.Generator):
    """Sample a valid random action from the action space."""
    space = env.action_space(agent)
    return {
        "action_type": rng.integers(0, space["action_type"].n),
        "amount": rng.uniform(0.0, 1.0, size=(1,)).astype(np.float32),
    }


class TestReset:
    def test_reset_returns_all_agents(self):
        cfg = _small_config()
        env = MixedPettingZooParallelEnv(cfg)
        obs, infos = env.reset(seed=1)

        assert len(obs) == cfg.population.num_agents
        assert set(obs.keys()) == set(env.possible_agents)
        assert set(infos.keys()) == set(env.possible_agents)
        assert env.agents == env.possible_agents

    def test_observation_spaces_defined(self):
        cfg = _small_config()
        env = MixedPettingZooParallelEnv(cfg)
        env.reset(seed=1)

        for agent in env.agents:
            space = env.observation_space(agent)
            assert "step" in space.spaces
            assert "shared_pool" in space.spaces
            assert "own_resources" in space.spaces

    def test_action_spaces_defined(self):
        cfg = _small_config()
        env = MixedPettingZooParallelEnv(cfg)
        for agent in env.possible_agents:
            space = env.action_space(agent)
            assert "action_type" in space.spaces
            assert "amount" in space.spaces
            assert space["action_type"].n == 4


class TestStep:
    def test_step_runs_without_error(self):
        cfg = _small_config()
        env = MixedPettingZooParallelEnv(cfg)
        env.reset(seed=1)
        rng = np.random.default_rng(99)

        for _ in range(5):
            if not env.agents:
                break
            actions = {a: _random_action(env, a, rng) for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)

            assert set(obs.keys()) == set(actions.keys())
            assert set(rewards.keys()) == set(actions.keys())
            for a in actions:
                assert isinstance(rewards[a], float)
                assert isinstance(terms[a], bool)
                assert isinstance(truncs[a], bool)


class TestDeterminism:
    def test_same_seed_same_actions_same_rewards(self):
        cfg = _small_config(seed=7)

        def run_episode():
            env = MixedPettingZooParallelEnv(cfg)
            env.reset(seed=7)
            rng = np.random.default_rng(123)
            all_rewards = []
            for _ in range(3):
                actions = {a: _random_action(env, a, rng) for a in env.agents}
                _, rewards, _, _, _ = env.step(actions)
                all_rewards.append(rewards)
            return all_rewards

        r1 = run_episode()
        r2 = run_episode()

        for step_r1, step_r2 in zip(r1, r2):
            for agent in step_r1:
                assert step_r1[agent] == pytest.approx(step_r2[agent])


class TestTruncation:
    def test_truncation_true_at_max_steps(self):
        cfg = _small_config(max_steps=10)
        # Need max_steps >= 10 per schema; use 10
        env = MixedPettingZooParallelEnv(cfg)
        env.reset(seed=1)
        rng = np.random.default_rng(42)

        truncated_seen = False
        for _ in range(15):
            if not env.agents:
                break
            actions = {a: _random_action(env, a, rng) for a in env.agents}
            _, _, terms, truncs, _ = env.step(actions)
            if any(truncs.values()):
                truncated_seen = True
                # When truncated, all agents should be terminated too
                assert all(terms.values())
                break

        assert truncated_seen, "Expected truncation at max_steps but it never occurred"
