"""Unit tests for the Mixed archetype environment core loop."""

import pytest

from simulation.config.defaults import default_config
from simulation.config.schema import (
    MixedEnvironmentConfig,
    EnvironmentIdentity,
    PopulationConfig,
    LayerConfig,
    RewardWeights,
    AgentConfig,
)
from simulation.core.types import TerminationReason
from simulation.envs.mixed.actions import Action, ActionType
from simulation.envs.mixed.env import MixedEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quick_config(seed: int = 42, **pop_overrides) -> MixedEnvironmentConfig:
    """Small fast config for testing."""
    pop_kwargs = dict(
        num_agents=3,
        max_steps=10,
        initial_shared_pool=100.0,
        initial_agent_resources=20.0,
        collapse_threshold=5.0,
    )
    pop_kwargs.update(pop_overrides)
    return MixedEnvironmentConfig(
        identity=EnvironmentIdentity(seed=seed),
        population=PopulationConfig(**pop_kwargs),
        layers=LayerConfig(
            information_asymmetry=0.0,
            temporal_memory_depth=5,
            reputation_sensitivity=0.5,
            incentive_softness=0.8,
            uncertainty_intensity=0.0,  # no noise for determinism tests
        ),
        rewards=RewardWeights(
            individual_weight=1.0,
            group_weight=0.5,
            relational_weight=0.3,
            penalty_scaling=1.0,
        ),
        agents=AgentConfig(observation_memory_steps=3),
    )


def _all_cooperate(env: MixedEnvironment, amount: float = 0.5) -> dict:
    return {
        aid: Action(type=ActionType.COOPERATE, amount=amount)
        for aid in env.active_agents()
    }


def _all_extract(env: MixedEnvironment, amount: float = 1.0) -> dict:
    return {
        aid: Action(type=ActionType.EXTRACT, amount=amount)
        for aid in env.active_agents()
    }


# ---------------------------------------------------------------------------
# Reset / step contract
# ---------------------------------------------------------------------------

class TestResetStepContract:
    def test_reset_returns_observations_for_all_agents(self):
        cfg = _quick_config()
        env = MixedEnvironment(cfg)
        obs = env.reset()

        assert len(obs) == 3
        for aid, ob in obs.items():
            assert aid.startswith("agent_")
            assert ob["step"] == 0
            assert ob["shared_pool"] == 100.0
            assert ob["own_resources"] == 20.0
            assert ob["num_active_agents"] == 3

    def test_step_returns_step_results(self):
        cfg = _quick_config()
        env = MixedEnvironment(cfg)
        env.reset()

        actions = _all_cooperate(env)
        results = env.step(actions)

        assert len(results) == 3
        for aid, sr in results.items():
            assert isinstance(sr.reward, float)
            assert isinstance(sr.observation, dict)
            assert isinstance(sr.done, bool)

    def test_step_without_reset_raises(self):
        env = MixedEnvironment(_quick_config())
        with pytest.raises(RuntimeError, match="reset"):
            env.step({})

    def test_step_after_done_raises(self):
        cfg = _quick_config(max_steps=10)
        env = MixedEnvironment(cfg)
        env.reset()
        for _ in range(10):
            if env.is_done():
                break
            env.step(_all_cooperate(env))
        assert env.is_done()
        with pytest.raises(RuntimeError, match="done"):
            env.step({})

    def test_current_step_increments(self):
        env = MixedEnvironment(_quick_config())
        env.reset()
        assert env.current_step == 0
        env.step(_all_cooperate(env))
        assert env.current_step == 1
        env.step(_all_cooperate(env))
        assert env.current_step == 2

    def test_missing_action_defaults_to_defend(self):
        env = MixedEnvironment(_quick_config())
        env.reset()
        # Submit empty dict — all agents default to DEFEND
        results = env.step({})
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_trajectory(self):
        cfg = _quick_config(seed=99)
        # Add small noise to make RNG usage visible
        cfg = _quick_config(seed=99)

        env1 = MixedEnvironment(cfg)
        env2 = MixedEnvironment(cfg)

        obs1 = env1.reset(seed=99)
        obs2 = env2.reset(seed=99)
        assert obs1 == obs2

        for _ in range(5):
            actions1 = _all_cooperate(env1, amount=0.3)
            actions2 = _all_cooperate(env2, amount=0.3)
            r1 = env1.step(actions1)
            r2 = env2.step(actions2)
            for aid in r1:
                assert r1[aid].reward == r2[aid].reward
                assert r1[aid].observation == r2[aid].observation

    def test_different_seed_different_trajectory(self):
        """With noise enabled, different seeds produce different pools."""
        cfg_a = _quick_config(seed=1)
        cfg_b = _quick_config(seed=2)
        # Enable noise
        cfg_a.layers.uncertainty_intensity = 0.1
        cfg_b.layers.uncertainty_intensity = 0.1

        env_a = MixedEnvironment(cfg_a)
        env_b = MixedEnvironment(cfg_b)
        env_a.reset(seed=1)
        env_b.reset(seed=2)

        for _ in range(5):
            env_a.step(_all_cooperate(env_a, 0.3))
            env_b.step(_all_cooperate(env_b, 0.3))

        # After 5 steps with noise, pools should differ
        pool_a = env_a._state.shared_pool
        pool_b = env_b._state.shared_pool
        assert pool_a != pool_b


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------

class TestTermination:
    def test_max_steps_termination(self):
        cfg = _quick_config(max_steps=10)
        env = MixedEnvironment(cfg)
        env.reset()

        for _ in range(10):
            env.step(_all_cooperate(env))

        assert env.is_done()
        assert env.termination_reason() == TerminationReason.MAX_STEPS

    def test_system_collapse_via_extraction(self):
        """Heavy extraction should drain the pool below collapse threshold."""
        cfg = _quick_config(
            initial_shared_pool=20.0,
            collapse_threshold=5.0,
            max_steps=100,
        )
        env = MixedEnvironment(cfg)
        env.reset()

        # Keep extracting until collapse
        for _ in range(100):
            if env.is_done():
                break
            env.step(_all_extract(env, amount=1.0))

        assert env.is_done()
        assert env.termination_reason() == TerminationReason.SYSTEM_COLLAPSE

    def test_not_done_initially(self):
        env = MixedEnvironment(_quick_config())
        env.reset()
        assert not env.is_done()
        assert env.termination_reason() is None


# ---------------------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------------------

class TestActions:
    def test_action_amount_validation(self):
        with pytest.raises(ValueError):
            Action(type=ActionType.COOPERATE, amount=1.5)
        with pytest.raises(ValueError):
            Action(type=ActionType.EXTRACT, amount=-0.1)

    def test_defend_ignores_amount(self):
        a = Action(type=ActionType.DEFEND, amount=0.9)
        assert a.amount == 0.0

    def test_conditional_ignores_amount(self):
        a = Action(type=ActionType.CONDITIONAL, amount=0.7)
        assert a.amount == 0.0

    def test_cooperate_preserves_amount(self):
        a = Action(type=ActionType.COOPERATE, amount=0.5)
        assert a.amount == 0.5


# ---------------------------------------------------------------------------
# Cooperation and extraction mechanics
# ---------------------------------------------------------------------------

class TestMechanics:
    def test_cooperation_increases_pool(self):
        env = MixedEnvironment(_quick_config())
        env.reset()
        initial_pool = env._state.shared_pool
        env.step(_all_cooperate(env, amount=0.5))
        # Each agent contributes 0.5 * 20 = 10, total +30
        assert env._state.shared_pool > initial_pool

    def test_extraction_decreases_pool(self):
        env = MixedEnvironment(_quick_config())
        env.reset()
        initial_pool = env._state.shared_pool
        env.step(_all_extract(env, amount=0.5))
        assert env._state.shared_pool < initial_pool

    def test_defend_gives_small_bonus(self):
        env = MixedEnvironment(_quick_config())
        env.reset()
        initial_resources = env._state.agents["agent_0"].resources
        env.step({aid: Action(type=ActionType.DEFEND) for aid in env.active_agents()})
        assert env._state.agents["agent_0"].resources > initial_resources
