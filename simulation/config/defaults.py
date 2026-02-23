"""Default Mixed-archetype configuration.

Provides a sensible baseline for quick experiments.
All values are explicit — no hidden magic.
"""

from simulation.config.schema import (
    AgentConfig,
    EnvironmentIdentity,
    InstrumentationConfig,
    LayerConfig,
    MixedEnvironmentConfig,
    PopulationConfig,
    RewardWeights,
)


def default_config(seed: int = 42) -> MixedEnvironmentConfig:
    """Return a complete, valid default config for the Mixed archetype."""
    return MixedEnvironmentConfig(
        identity=EnvironmentIdentity(seed=seed),
        population=PopulationConfig(
            num_agents=5,
            max_steps=200,
            initial_shared_pool=100.0,
            initial_agent_resources=20.0,
            collapse_threshold=5.0,
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
        agents=AgentConfig(
            observation_memory_steps=5,
        ),
        instrumentation=InstrumentationConfig(),
    )
