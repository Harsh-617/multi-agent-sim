"""Tests for the Mixed archetype configuration schema.

Covers:
  - valid config construction
  - field-level validation (ranges, types)
  - cross-field validators (collapse vs pool, memory depth consistency)
  - reward weight constraints
  - default config validity
"""

import pytest
from pydantic import ValidationError

from simulation.config.defaults import default_config
from simulation.config.schema import (
    AgentConfig,
    EnvironmentIdentity,
    LayerConfig,
    MixedEnvironmentConfig,
    PopulationConfig,
    RewardWeights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_kwargs(seed: int = 42) -> dict:
    """Return kwargs that produce a valid MixedEnvironmentConfig."""
    return dict(
        identity=EnvironmentIdentity(seed=seed),
        population=PopulationConfig(
            num_agents=4,
            max_steps=100,
            initial_shared_pool=50.0,
            initial_agent_resources=10.0,
            collapse_threshold=2.0,
        ),
        layers=LayerConfig(
            information_asymmetry=0.3,
            temporal_memory_depth=10,
            reputation_sensitivity=0.5,
            incentive_softness=0.8,
            uncertainty_intensity=0.1,
        ),
        rewards=RewardWeights(
            individual_weight=1.0,
            group_weight=0.5,
            relational_weight=0.3,
            penalty_scaling=1.0,
        ),
        agents=AgentConfig(observation_memory_steps=5),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestValidConfig:
    def test_default_config_is_valid(self):
        cfg = default_config()
        assert cfg.identity.environment_type == "mixed"
        assert cfg.population.num_agents >= 2

    def test_explicit_valid_config(self):
        cfg = MixedEnvironmentConfig(**_base_kwargs())
        assert cfg.population.num_agents == 4
        assert cfg.identity.seed == 42

    def test_instrumentation_defaults(self):
        cfg = MixedEnvironmentConfig(**_base_kwargs())
        assert cfg.instrumentation.enable_step_metrics is True
        assert cfg.instrumentation.step_log_frequency == 1


# ---------------------------------------------------------------------------
# Environment identity
# ---------------------------------------------------------------------------

class TestEnvironmentIdentity:
    def test_rejects_non_mixed_type(self):
        with pytest.raises(ValidationError, match="environment_type"):
            EnvironmentIdentity(environment_type="competitive", seed=1)

    def test_rejects_negative_seed(self):
        with pytest.raises(ValidationError):
            EnvironmentIdentity(seed=-1)


# ---------------------------------------------------------------------------
# Population constraints
# ---------------------------------------------------------------------------

class TestPopulation:
    def test_min_agents(self):
        with pytest.raises(ValidationError):
            PopulationConfig(
                num_agents=1, max_steps=100,
                initial_shared_pool=50.0, initial_agent_resources=10.0,
                collapse_threshold=0.0,
            )

    def test_max_agents(self):
        with pytest.raises(ValidationError):
            PopulationConfig(
                num_agents=21, max_steps=100,
                initial_shared_pool=50.0, initial_agent_resources=10.0,
                collapse_threshold=0.0,
            )

    def test_min_steps(self):
        with pytest.raises(ValidationError):
            PopulationConfig(
                num_agents=2, max_steps=5,
                initial_shared_pool=50.0, initial_agent_resources=10.0,
                collapse_threshold=0.0,
            )

    def test_pool_must_be_positive(self):
        with pytest.raises(ValidationError):
            PopulationConfig(
                num_agents=2, max_steps=100,
                initial_shared_pool=0.0, initial_agent_resources=10.0,
                collapse_threshold=0.0,
            )


# ---------------------------------------------------------------------------
# Layer constraints
# ---------------------------------------------------------------------------

class TestLayers:
    def test_uncertainty_capped(self):
        with pytest.raises(ValidationError):
            LayerConfig(
                information_asymmetry=0.3,
                temporal_memory_depth=10,
                reputation_sensitivity=0.5,
                incentive_softness=0.8,
                uncertainty_intensity=0.9,  # exceeds 0.5 cap
            )

    def test_memory_depth_minimum(self):
        with pytest.raises(ValidationError):
            LayerConfig(
                information_asymmetry=0.3,
                temporal_memory_depth=0,  # must be >= 1
                reputation_sensitivity=0.5,
                incentive_softness=0.8,
                uncertainty_intensity=0.1,
            )


# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------

class TestRewardWeights:
    def test_all_zero_weights_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            RewardWeights(
                individual_weight=0.0,
                group_weight=0.0,
                relational_weight=0.0,
                penalty_scaling=1.0,
            )

    def test_single_positive_weight_ok(self):
        rw = RewardWeights(
            individual_weight=1.0,
            group_weight=0.0,
            relational_weight=0.0,
            penalty_scaling=0.0,
        )
        assert rw.individual_weight == 1.0

    def test_negative_weight_rejected(self):
        with pytest.raises(ValidationError):
            RewardWeights(
                individual_weight=-0.5,
                group_weight=1.0,
                relational_weight=0.0,
                penalty_scaling=1.0,
            )


# ---------------------------------------------------------------------------
# Cross-field validators
# ---------------------------------------------------------------------------

class TestCrossFieldValidation:
    def test_collapse_above_pool_rejected(self):
        kw = _base_kwargs()
        kw["population"] = PopulationConfig(
            num_agents=4, max_steps=100,
            initial_shared_pool=50.0,
            initial_agent_resources=10.0,
            collapse_threshold=60.0,  # > pool
        )
        with pytest.raises(ValidationError, match="collapse_threshold"):
            MixedEnvironmentConfig(**kw)

    def test_collapse_equal_to_pool_ok(self):
        kw = _base_kwargs()
        kw["population"] = PopulationConfig(
            num_agents=4, max_steps=100,
            initial_shared_pool=50.0,
            initial_agent_resources=10.0,
            collapse_threshold=50.0,
        )
        cfg = MixedEnvironmentConfig(**kw)
        assert cfg.population.collapse_threshold == 50.0

    def test_agent_memory_exceeds_env_memory_rejected(self):
        kw = _base_kwargs()
        kw["layers"] = LayerConfig(
            information_asymmetry=0.3,
            temporal_memory_depth=5,
            reputation_sensitivity=0.5,
            incentive_softness=0.8,
            uncertainty_intensity=0.1,
        )
        kw["agents"] = AgentConfig(observation_memory_steps=10)  # > 5
        with pytest.raises(ValidationError, match="observation_memory_steps"):
            MixedEnvironmentConfig(**kw)

    def test_agent_memory_equal_to_env_memory_ok(self):
        kw = _base_kwargs()
        kw["layers"] = LayerConfig(
            information_asymmetry=0.3,
            temporal_memory_depth=5,
            reputation_sensitivity=0.5,
            incentive_softness=0.8,
            uncertainty_intensity=0.1,
        )
        kw["agents"] = AgentConfig(observation_memory_steps=5)
        cfg = MixedEnvironmentConfig(**kw)
        assert cfg.agents.observation_memory_steps == 5


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_json_round_trip(self):
        cfg = default_config()
        json_str = cfg.model_dump_json()
        restored = MixedEnvironmentConfig.model_validate_json(json_str)
        assert restored == cfg

    def test_dict_round_trip(self):
        cfg = default_config()
        d = cfg.model_dump()
        restored = MixedEnvironmentConfig.model_validate(d)
        assert restored == cfg
