"""Request/response models for the API layer.

These are thin API-surface models only.  The actual config schema lives in
simulation.config.schema and is imported directly — no duplication.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------

class ConfigListItem(BaseModel):
    """Summary returned when listing saved configs."""

    config_id: str
    seed: int
    num_agents: int
    max_steps: int


class ConfigCreatedResponse(BaseModel):
    config_id: str


# ---------------------------------------------------------------------------
# Run control
# ---------------------------------------------------------------------------

class StartRunRequest(BaseModel):
    config_id: str = Field(description="ID of a previously saved config.")
    agent_policy: Literal[
        "random", "always_cooperate", "always_extract", "tit_for_tat",
        "ppo_shared", "league_snapshot",
    ] = Field(
        default="random",
        description="Agent policy to use for all agents in the run.",
    )
    league_member_id: str | None = Field(
        default=None,
        description="Required when agent_policy is 'league_snapshot'.",
    )


class StartRunResponse(BaseModel):
    run_id: str


class RunStatus(BaseModel):
    running: bool
    run_id: str | None = None
    step: int | None = None
    max_steps: int | None = None
    termination_reason: str | None = None


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class BenchmarkRequest(BaseModel):
    config_id: str = Field(description="ID of a previously saved config.")
    agent_policies: list[
        Literal["random", "always_cooperate", "always_extract", "tit_for_tat"]
    ] = Field(
        description="List of agent policies to compare.",
        min_length=1,
    )
