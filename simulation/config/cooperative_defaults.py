"""Default Cooperative-archetype configuration.

Sensible baseline for quick experiments.
All values are explicit — no hidden magic.
"""

from simulation.config.cooperative_schema import (
    CooperativeEnvironmentConfig,
    EnvironmentIdentity,
    InstrumentationConfig,
    LayerConfig,
    PopulationConfig,
    RewardWeights,
    TaskConfig,
)


def default_cooperative_config(seed: int = 42) -> CooperativeEnvironmentConfig:
    """Return a complete, valid default config for the Cooperative archetype."""
    return CooperativeEnvironmentConfig(
        identity=EnvironmentIdentity(seed=seed),
        population=PopulationConfig(
            num_agents=4,
            max_steps=200,
            num_task_types=3,
            agent_effort_capacity=1.0,
            collapse_sustain_window=10,
            enable_early_success=False,
            clearance_sustain_window=15,
        ),
        layers=LayerConfig(
            observation_noise=0.0,
            history_window=5,
            specialization_scale=0.3,
            specialization_decay=0.1,
            task_arrival_noise=0.1,
            task_difficulty_variance=0.0,
            free_rider_pressure_scale=1.0,
        ),
        task=TaskConfig(
            task_arrival_rate=[1.0, 1.0, 1.0],
            task_difficulty=[1.0, 1.0, 1.0],
            collapse_threshold=50,
            initial_backlog=0,
        ),
        rewards=RewardWeights(
            w_group=0.7,
            w_individual=0.2,
            w_efficiency=0.1,
        ),
        instrumentation=InstrumentationConfig(),
    )
