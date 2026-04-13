"""20-variant robustness sweep definitions for the Cooperative archetype.

Varies: task_arrival_rate, task_difficulty, num_agents, specialization_scale,
        collapse_threshold, observation_noise.

Mirrors simulation/evaluation/sweeps.py — cooperative-specific parameter ranges.
Does NOT modify sweeps.py (ADR-013).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from simulation.config.cooperative_schema import CooperativeEnvironmentConfig


@dataclass(frozen=True, slots=True)
class CoopSweepSpec:
    """One environment variant for cooperative robustness testing."""

    name: str
    description: str
    config_patch: dict[str, Any]
    tags: tuple[str, ...] = ()


def apply_coop_sweep(
    config: CooperativeEnvironmentConfig,
    sweep: CoopSweepSpec,
) -> CooperativeEnvironmentConfig:
    """Apply a CoopSweepSpec to a config, returning a new config.

    The config_patch maps section names to field updates.
    The base config is never mutated.
    """
    nested_updates: dict[str, Any] = {}
    for section, updates in sweep.config_patch.items():
        current_section = getattr(config, section)
        nested_updates[section] = current_section.model_copy(update=updates)

    return config.model_copy(update=nested_updates)


def build_cooperative_sweeps() -> list[CoopSweepSpec]:
    """Return 20 sweep variants covering cooperative stress dimensions."""
    sweeps: list[CoopSweepSpec] = []

    # A) Task arrival rate stress (4 variants)
    for rate_scale, label in (
        (0.5, "low_arrival"),
        (1.5, "high_arrival"),
        (2.0, "very_high_arrival"),
        (0.25, "very_low_arrival"),
    ):
        sweeps.append(CoopSweepSpec(
            name=f"arrival_{label}",
            description=f"Task arrival rate scaled to {rate_scale}x default",
            config_patch={
                "task": {
                    "task_arrival_rate": [rate_scale, rate_scale, rate_scale]
                }
            },
            tags=("arrival",),
        ))

    # B) Task difficulty stress (4 variants)
    for diff, label in (
        (0.5, "easy"),
        (1.5, "hard"),
        (2.0, "very_hard"),
        (0.25, "very_easy"),
    ):
        sweeps.append(CoopSweepSpec(
            name=f"difficulty_{label}",
            description=f"Task difficulty = {diff}",
            config_patch={
                "task": {
                    "task_difficulty": [diff, diff, diff]
                }
            },
            tags=("difficulty",),
        ))

    # C) Agent count (3 variants)
    for n_agents in (2, 6, 8):
        sweeps.append(CoopSweepSpec(
            name=f"agents_{n_agents}",
            description=f"{n_agents} agents instead of default 4",
            config_patch={"population": {"num_agents": n_agents}},
            tags=("population",),
        ))

    # D) Specialization scale (2 variants)
    for spec_scale, label in ((0.1, "low_spec"), (0.6, "high_spec")):
        sweeps.append(CoopSweepSpec(
            name=f"spec_{label}",
            description=f"Specialization scale = {spec_scale}",
            config_patch={"layers": {"specialization_scale": spec_scale}},
            tags=("specialization",),
        ))

    # E) Collapse threshold (2 variants)
    for threshold, label in ((20, "tight_collapse"), (100, "loose_collapse")):
        sweeps.append(CoopSweepSpec(
            name=f"collapse_{label}",
            description=f"Collapse threshold = {threshold}",
            config_patch={"task": {"collapse_threshold": threshold}},
            tags=("collapse",),
        ))

    # F) Observation noise (2 variants)
    for noise, label in ((0.1, "low_noise"), (0.3, "high_noise")):
        sweeps.append(CoopSweepSpec(
            name=f"noise_{label}",
            description=f"Observation noise = {noise}",
            config_patch={"layers": {"observation_noise": noise}},
            tags=("observation",),
        ))

    # G) Combined stress variants (3 variants)
    sweeps.append(CoopSweepSpec(
        name="combined_hard",
        description="High arrival + hard tasks + small team",
        config_patch={
            "task": {
                "task_arrival_rate": [1.8, 1.8, 1.8],
                "task_difficulty": [1.5, 1.5, 1.5],
                "collapse_threshold": 30,
            },
            "population": {"num_agents": 2},
        },
        tags=("combined", "hard"),
    ))

    sweeps.append(CoopSweepSpec(
        name="combined_easy",
        description="Low arrival + easy tasks + large team",
        config_patch={
            "task": {
                "task_arrival_rate": [0.6, 0.6, 0.6],
                "task_difficulty": [0.6, 0.6, 0.6],
                "collapse_threshold": 80,
            },
            "population": {"num_agents": 6},
        },
        tags=("combined", "easy"),
    ))

    sweeps.append(CoopSweepSpec(
        name="combined_noisy_hard",
        description="High noise + hard tasks + high specialization pressure",
        config_patch={
            "layers": {
                "observation_noise": 0.3,
                "specialization_scale": 0.5,
            },
            "task": {"task_difficulty": [1.8, 1.8, 1.8]},
        },
        tags=("combined", "observation", "specialization"),
    ))

    # Ensure exactly 20 sweeps
    assert len(sweeps) == 20, f"Expected 20 sweeps, got {len(sweeps)}"
    return sweeps
