"""Configuration schema for the Mixed archetype — single source of truth.

This module defines the Pydantic models that fully describe a Mixed
environment instance.  The backend imports these directly; no duplication.

Design references:
  - Mixed archetype Part 9 (Configuration Schema)
  - V1 decisions: discrete+parameter actions, 3 termination conditions,
    no agent-to-agent targeting
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Section 1: Environment Identity
# ---------------------------------------------------------------------------

class EnvironmentIdentity(BaseModel):
    """What this environment instance is."""

    environment_type: str = Field(
        default="mixed",
        pattern=r"^mixed$",
        description="Archetype type. V1 only supports 'mixed'.",
    )
    environment_version: str = Field(
        default="0.1.0",
        description="Schema version for compatibility checks.",
    )
    seed: int = Field(
        ge=0,
        description="Root seed for full reproducibility.",
    )


# ---------------------------------------------------------------------------
# Section 2: Population & Episode
# ---------------------------------------------------------------------------

class PopulationConfig(BaseModel):
    """Who exists and for how long."""

    num_agents: int = Field(
        ge=2, le=20,
        description="Number of agents. Must be >= 2.",
    )
    max_steps: int = Field(
        ge=10, le=10_000,
        description="Maximum timesteps per episode.",
    )
    initial_shared_pool: float = Field(
        gt=0.0,
        description="Starting size of the shared resource pool.",
    )
    initial_agent_resources: float = Field(
        gt=0.0,
        description="Starting private resources per agent.",
    )
    collapse_threshold: float = Field(
        ge=0.0,
        description=(
            "Shared pool level at or below which system collapse occurs. "
            "Set to 0 for no collapse termination."
        ),
    )


# ---------------------------------------------------------------------------
# Section 3: Layer Configuration (knob settings)
# ---------------------------------------------------------------------------

class LayerConfig(BaseModel):
    """Controls intensity of each framework layer.

    Values are normalized floats in [0, 1] representing intensity.
    Layers that are LOW for V1 are constrained but still present.
    """

    information_asymmetry: float = Field(
        ge=0.0, le=1.0,
        description="How much observation is masked. 0 = full visibility, 1 = heavy masking.",
    )
    temporal_memory_depth: int = Field(
        ge=1, le=50,
        description="How many past steps of history are tracked for reputation/trust.",
    )
    reputation_sensitivity: float = Field(
        ge=0.0, le=1.0,
        description="How strongly past behavior affects current dynamics.",
    )
    incentive_softness: float = Field(
        ge=0.0, le=1.0,
        description="0 = hard bans on bad actions, 1 = only penalties (preferred).",
    )
    uncertainty_intensity: float = Field(
        ge=0.0, le=0.5,
        description="Noise magnitude. Capped low for V1.",
    )


# ---------------------------------------------------------------------------
# Section 4: Reward Weights
# ---------------------------------------------------------------------------

class RewardWeights(BaseModel):
    """How the three reward components are combined.

    Weights do not need to sum to 1 — they are relative scaling factors.
    """

    individual_weight: float = Field(
        ge=0.0,
        description="Weight for personal resource gain / survival.",
    )
    group_weight: float = Field(
        ge=0.0,
        description="Weight for shared pool health / group outcome.",
    )
    relational_weight: float = Field(
        ge=0.0,
        description="Weight for reputation / history component.",
    )
    penalty_scaling: float = Field(
        ge=0.0, le=5.0,
        description="Multiplier applied to penalty terms (over-extraction, betrayal).",
    )

    @model_validator(mode="after")
    def at_least_one_positive_weight(self) -> RewardWeights:
        total = self.individual_weight + self.group_weight + self.relational_weight
        if total <= 0:
            raise ValueError("At least one reward weight must be positive.")
        return self


# ---------------------------------------------------------------------------
# Section 5: Agent Configuration
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """Per-agent settings.  V1 uses homogeneous agents (same config for all)."""

    observation_memory_steps: int = Field(
        ge=0, le=50,
        description="How many past steps the agent observes in its history window.",
    )


# ---------------------------------------------------------------------------
# Section 6: Instrumentation
# ---------------------------------------------------------------------------

class InstrumentationConfig(BaseModel):
    """What metrics to collect and how often."""

    enable_step_metrics: bool = Field(
        default=True,
        description="Collect per-step metrics (rewards, actions, state deltas).",
    )
    enable_episode_metrics: bool = Field(
        default=True,
        description="Collect episode-level summary metrics.",
    )
    enable_event_log: bool = Field(
        default=True,
        description="Log semantic events (collapse, elimination, etc.).",
    )
    step_log_frequency: int = Field(
        default=1, ge=1,
        description="Log step metrics every N steps. 1 = every step.",
    )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class MixedEnvironmentConfig(BaseModel):
    """Complete configuration for one Mixed-archetype environment instance.

    A single instance of this model fully defines a reproducible experiment.
    """

    identity: EnvironmentIdentity
    population: PopulationConfig
    layers: LayerConfig
    rewards: RewardWeights
    agents: AgentConfig
    instrumentation: InstrumentationConfig = InstrumentationConfig()

    @model_validator(mode="after")
    def collapse_threshold_below_pool(self) -> MixedEnvironmentConfig:
        if self.population.collapse_threshold > self.population.initial_shared_pool:
            raise ValueError(
                "collapse_threshold must be <= initial_shared_pool "
                f"(got {self.population.collapse_threshold} > {self.population.initial_shared_pool})."
            )
        return self

    @model_validator(mode="after")
    def memory_depth_consistency(self) -> MixedEnvironmentConfig:
        if self.agents.observation_memory_steps > self.layers.temporal_memory_depth:
            raise ValueError(
                "Agent observation_memory_steps cannot exceed the environment's "
                f"temporal_memory_depth ({self.agents.observation_memory_steps} > "
                f"{self.layers.temporal_memory_depth})."
            )
        return self
