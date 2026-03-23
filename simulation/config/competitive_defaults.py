"""Default Competitive-archetype configuration.

Provides a sensible baseline for quick experiments.
All values are explicit — no hidden magic.
"""

from simulation.config.competitive_schema import (
    AgentConfig,
    CompetitiveEnvironmentConfig,
    EnvironmentIdentity,
    InstrumentationConfig,
    LayerConfig,
    PopulationConfig,
    RewardWeights,
)


def default_competitive_config(seed: int = 42) -> CompetitiveEnvironmentConfig:
    """Return a complete, valid default config for the Competitive archetype."""
    return CompetitiveEnvironmentConfig(
        identity=EnvironmentIdentity(seed=seed),
        population=PopulationConfig(
            num_agents=4,
            max_steps=200,
            initial_score=0.0,
            initial_resources=20.0,
            resource_regeneration_rate=1.0,
            elimination_threshold=0.0,
            dominance_margin=0.0,
        ),
        layers=LayerConfig(
            information_asymmetry=0.3,
            opponent_history_depth=10,
            opponent_obs_window=5,
            history_sensitivity=0.5,
            incentive_softness=0.8,
            uncertainty_intensity=0.1,
            gamble_variance=0.5,
        ),
        rewards=RewardWeights(
            absolute_gain_weight=1.0,
            relative_gain_weight=0.5,
            efficiency_weight=0.3,
            terminal_bonus_scale=2.0,
            penalty_scaling=1.0,
        ),
        agents=AgentConfig(
            observation_memory_steps=5,
        ),
        instrumentation=InstrumentationConfig(),
    )
