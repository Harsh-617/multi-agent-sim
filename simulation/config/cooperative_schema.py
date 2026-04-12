"""Configuration schema for the Cooperative archetype — single source of truth.

Six sections (spec Part 9):
  identity       — what this environment instance is
  population     — agents, episode, termination parameters
  layers         — knob settings for each framework layer
  task           — task world parameters
  rewards        — reward component weights
  instrumentation — logging flags
"""

from __future__ import annotations

from typing import Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Section 1: Environment Identity
# ---------------------------------------------------------------------------

class EnvironmentIdentity(BaseModel):
    environment_type: str = Field(
        default="cooperative",
        pattern=r"^cooperative$",
        description="Must be 'cooperative'.",
    )
    environment_version: str = Field(
        default="1.0.0",
        description="Schema version for compatibility checks.",
    )
    archetype: str = Field(
        default="shared_goal_collective",
        description="Archetype identifier.",
    )
    seed: int = Field(
        ge=0,
        description="Global random seed — controls all stochasticity.",
    )


# ---------------------------------------------------------------------------
# Section 2: Population & Episode Parameters
# ---------------------------------------------------------------------------

class PopulationConfig(BaseModel):
    num_agents: int = Field(ge=2, description="Number of agents. Must be >= 2.")
    max_steps: int = Field(ge=10, description="Maximum timesteps per episode.")
    num_task_types: int = Field(ge=1, description="Number of distinct task types.")
    agent_effort_capacity: float = Field(
        gt=0.0, description="Maximum effort an agent can apply per step."
    )
    collapse_sustain_window: int = Field(
        ge=1,
        default=10,
        description="Consecutive steps at full stress before system collapse triggers.",
    )
    enable_early_success: bool = Field(
        default=False,
        description="Whether perfect clearance termination is active.",
    )
    clearance_sustain_window: int = Field(
        ge=1,
        default=15,
        description="Consecutive steps at zero backlog before early success triggers.",
    )


# ---------------------------------------------------------------------------
# Section 3: Layer Configuration
# ---------------------------------------------------------------------------

class LayerConfig(BaseModel):
    observation_noise: float = Field(
        ge=0.0, le=0.2,
        description="Noise added to public observation signals.",
    )
    history_window: int = Field(
        ge=1,
        description="Steps of history included in agent observations (K).",
    )
    specialization_scale: float = Field(
        ge=0.0, le=0.5,
        description="Maximum specialization efficiency bonus (0 = no effect).",
    )
    specialization_decay: float = Field(
        gt=0.0, lt=1.0,
        description="EMA decay rate for specialization score updates.",
    )
    task_arrival_noise: float = Field(
        ge=0.0, le=0.3,
        description="Fractional noise on task arrival rates per step.",
    )
    task_difficulty_variance: float = Field(
        ge=0.0, le=0.3,
        description="Episode-level variance in task difficulty.",
    )
    free_rider_pressure_scale: float = Field(
        ge=0.0, le=1.0,
        description="Scales the free_rider_pressure signal in observations.",
    )


# ---------------------------------------------------------------------------
# Section 4: Task & World Parameters
# ---------------------------------------------------------------------------

class TaskConfig(BaseModel):
    task_arrival_rate: Union[float, list[float]] = Field(
        description="Base task arrival rate per step. Scalar or per-type list.",
    )
    task_difficulty: Union[float, list[float]] = Field(
        description="Effort required per task. Scalar or per-type list.",
    )
    collapse_threshold: int = Field(
        gt=0, description="Backlog level at which system_stress reaches 1.0."
    )
    initial_backlog: int = Field(
        ge=0, default=0, description="Tasks pre-loaded at episode start."
    )

    @model_validator(mode="after")
    def positive_arrival_rate(self) -> "TaskConfig":
        rates = (
            [self.task_arrival_rate]
            if isinstance(self.task_arrival_rate, (int, float))
            else self.task_arrival_rate
        )
        if any(r <= 0 for r in rates):
            raise ValueError("All task_arrival_rate values must be > 0.")
        return self

    @model_validator(mode="after")
    def positive_difficulty(self) -> "TaskConfig":
        diffs = (
            [self.task_difficulty]
            if isinstance(self.task_difficulty, (int, float))
            else self.task_difficulty
        )
        if any(d <= 0 for d in diffs):
            raise ValueError("All task_difficulty values must be > 0.")
        return self


# ---------------------------------------------------------------------------
# Section 5: Reward Weights
# ---------------------------------------------------------------------------

class RewardWeights(BaseModel):
    w_group: float = Field(
        gt=0.0, lt=1.0, description="Weight on group completion/stress component."
    )
    w_individual: float = Field(
        gt=0.0, lt=1.0, description="Weight on individual effort component."
    )
    w_efficiency: float = Field(
        gt=0.0, lt=1.0, description="Weight on specialization efficiency component."
    )

    @model_validator(mode="after")
    def weights_sum_to_one_and_group_dominates(self) -> "RewardWeights":
        total = self.w_group + self.w_individual + self.w_efficiency
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"w_group + w_individual + w_efficiency must equal 1.0 (got {total:.6f})."
            )
        if self.w_group < 0.5:
            raise ValueError(
                f"w_group must be >= 0.5 to ensure group signal dominates (got {self.w_group})."
            )
        return self


# ---------------------------------------------------------------------------
# Section 6: Instrumentation
# ---------------------------------------------------------------------------

class InstrumentationConfig(BaseModel):
    log_per_agent_contributions: bool = Field(default=True)
    log_specialization_scores: bool = Field(default=True)
    log_group_signals: bool = Field(default=True)
    log_task_queue: bool = Field(default=True)
    log_reward_components: bool = Field(default=True)
    log_termination_detail: bool = Field(default=True)
    # Step/episode metric collection (mirrors competitive pattern for runner)
    enable_step_metrics: bool = Field(default=True)
    enable_episode_metrics: bool = Field(default=True)
    step_log_frequency: int = Field(default=1, ge=1)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class CooperativeEnvironmentConfig(BaseModel):
    """Complete configuration for one Cooperative-archetype environment instance."""

    identity: EnvironmentIdentity
    population: PopulationConfig
    layers: LayerConfig
    task: TaskConfig
    rewards: RewardWeights
    instrumentation: InstrumentationConfig = InstrumentationConfig()
