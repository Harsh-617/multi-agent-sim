"""Configuration schema for the Competitive archetype — single source of truth.

This module defines the Pydantic models that fully describe a Competitive
environment instance.  The backend imports these directly; no duplication.

Design references:
  - Competitive archetype Part 9 (Configuration Schema)
  - V1 decisions: discrete+parameter actions, 3 termination conditions,
    untargeted attacks, dominance_margin disabled
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Section 1: Environment Identity
# ---------------------------------------------------------------------------

class EnvironmentIdentity(BaseModel):
    """What this environment instance is."""

    environment_type: str = Field(
        default="competitive",
        pattern=r"^competitive$",
        description="Archetype type. Must be 'competitive'.",
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
    initial_score: float = Field(
        ge=0.0,
        description="Starting score for all agents (typically 0).",
    )
    initial_resources: float = Field(
        gt=0.0,
        description="Starting resource budget per agent.",
    )
    resource_regeneration_rate: float = Field(
        ge=0.0,
        description="Resources recovered per step.",
    )
    elimination_threshold: float = Field(
        ge=0.0,
        description=(
            "Resource level at or below which an agent is eliminated. "
            "Set to 0 for elimination only at exactly zero resources."
        ),
    )
    dominance_margin: float = Field(
        ge=0.0,
        description=(
            "Score lead required for early dominance termination. "
            "V1: must be 0 (disabled). Reserved for V2."
        ),
    )

    @model_validator(mode="after")
    def elimination_threshold_below_initial_resources(self) -> PopulationConfig:
        if self.elimination_threshold > self.initial_resources:
            raise ValueError(
                "elimination_threshold must be <= initial_resources "
                f"(got {self.elimination_threshold} > {self.initial_resources})."
            )
        return self

    @model_validator(mode="after")
    def dominance_margin_disabled_v1(self) -> PopulationConfig:
        if self.dominance_margin != 0:
            raise ValueError(
                "dominance_margin must be 0 in V1 "
                f"(got {self.dominance_margin}). Reserved for V2."
            )
        return self


# ---------------------------------------------------------------------------
# Section 3: Layer Configuration (knob settings)
# ---------------------------------------------------------------------------

class LayerConfig(BaseModel):
    """Controls intensity of each framework layer."""

    information_asymmetry: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How much opponent scores/resources are masked. "
            "0 = full visibility, 1 = heavy masking."
        ),
    )
    opponent_history_depth: int = Field(
        ge=1, le=50,
        description="How many past steps of opponent action history are tracked.",
    )
    opponent_obs_window: int = Field(
        ge=1, le=50,
        description=(
            "How many recent opponent actions are visible in observations. "
            "Must be <= opponent_history_depth."
        ),
    )
    history_sensitivity: float = Field(
        ge=0.0, le=1.0,
        description="How strongly past opponent patterns modulate transition outcomes.",
    )
    incentive_softness: float = Field(
        ge=0.0, le=1.0,
        description="0 = hard bans on bad actions, 1 = only penalties (preferred).",
    )
    uncertainty_intensity: float = Field(
        ge=0.0, le=0.3,
        description="Noise magnitude on action outcomes. Capped at 0.3 for V1.",
    )
    gamble_variance: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Spread of GAMBLE action outcomes. "
            "0.0 = deterministic multiplier, 1.0 = maximum spread within [0.0, 2.5]."
        ),
    )


# ---------------------------------------------------------------------------
# Section 4: Reward Weights
# ---------------------------------------------------------------------------

class RewardWeights(BaseModel):
    """How the three reward components are combined.

    Weights do not need to sum to 1 — they are relative scaling factors.
    """

    absolute_gain_weight: float = Field(
        ge=0.0,
        description="Weight for raw score/resource gain this step.",
    )
    relative_gain_weight: float = Field(
        ge=0.0,
        description="Weight for rank/score-gap improvement vs opponents.",
    )
    efficiency_weight: float = Field(
        ge=0.0,
        description="Weight for score-gained per resource-spent.",
    )
    terminal_bonus_scale: float = Field(
        ge=0.0,
        description="Multiplier on the rank-based terminal bonus at episode end.",
    )
    penalty_scaling: float = Field(
        ge=0.0, le=5.0,
        description="Multiplier applied to failed-attack and over-commit penalties.",
    )

    @model_validator(mode="after")
    def at_least_one_positive_weight(self) -> RewardWeights:
        total = (
            self.absolute_gain_weight
            + self.relative_gain_weight
            + self.efficiency_weight
        )
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
        description="Log semantic events (elimination, etc.).",
    )
    step_log_frequency: int = Field(
        default=1, ge=1,
        description="Log step metrics every N steps. 1 = every step.",
    )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class CompetitiveEnvironmentConfig(BaseModel):
    """Complete configuration for one Competitive-archetype environment instance.

    A single instance of this model fully defines a reproducible experiment.
    """

    identity: EnvironmentIdentity
    population: PopulationConfig
    layers: LayerConfig
    rewards: RewardWeights
    agents: AgentConfig
    instrumentation: InstrumentationConfig = InstrumentationConfig()

    @model_validator(mode="after")
    def opponent_obs_window_within_depth(self) -> CompetitiveEnvironmentConfig:
        if self.layers.opponent_obs_window > self.layers.opponent_history_depth:
            raise ValueError(
                "opponent_obs_window must be <= opponent_history_depth "
                f"(got {self.layers.opponent_obs_window} > "
                f"{self.layers.opponent_history_depth})."
            )
        return self

    @model_validator(mode="after")
    def memory_depth_consistency(self) -> CompetitiveEnvironmentConfig:
        if self.agents.observation_memory_steps > self.layers.opponent_history_depth:
            raise ValueError(
                "Agent observation_memory_steps cannot exceed the environment's "
                f"opponent_history_depth ({self.agents.observation_memory_steps} > "
                f"{self.layers.opponent_history_depth})."
            )
        return self
