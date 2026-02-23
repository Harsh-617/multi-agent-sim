"""Sweep specifications for robustness evaluation.

Each SweepSpec describes a config perturbation to test policy robustness
under different environment conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from simulation.config.schema import MixedEnvironmentConfig


@dataclass(frozen=True, slots=True)
class SweepSpec:
    """One environment variant for robustness testing."""

    name: str
    description: str
    config_patch: dict[str, Any]
    tags: tuple[str, ...] = ()


def apply_sweep(
    config: MixedEnvironmentConfig,
    sweep: SweepSpec,
) -> MixedEnvironmentConfig:
    """Apply a SweepSpec to a config, returning a new config.

    The config_patch maps dot-separated section names to field updates.
    Example: {"layers": {"information_asymmetry": 0.6}}
    The base config is never mutated.
    """
    nested_updates: dict[str, Any] = {}
    for section, updates in sweep.config_patch.items():
        current_section = getattr(config, section)
        nested_updates[section] = current_section.model_copy(update=updates)

    return config.model_copy(update=nested_updates)


def build_default_sweeps() -> list[SweepSpec]:
    """Return ~10 sweep variants covering key stress dimensions."""
    sweeps: list[SweepSpec] = []

    # A) Observation stress: information_asymmetry
    for ia in (0.0, 0.6):
        sweeps.append(SweepSpec(
            name=f"obs_ia_{ia}",
            description=f"Information asymmetry = {ia}",
            config_patch={"layers": {"information_asymmetry": ia}},
            tags=("observation",),
        ))

    # B) Uncertainty stress: uncertainty_intensity
    for ui in (0.0, 0.3):
        sweeps.append(SweepSpec(
            name=f"uncertainty_{ui}",
            description=f"Uncertainty intensity = {ui}",
            config_patch={"layers": {"uncertainty_intensity": ui}},
            tags=("uncertainty",),
        ))

    # C) Incentive regime: incentive_softness
    for soft in (0.2, 0.8):
        sweeps.append(SweepSpec(
            name=f"incentive_soft_{soft}",
            description=f"Incentive softness = {soft}",
            config_patch={"layers": {"incentive_softness": soft}},
            tags=("incentive",),
        ))

    # D) Scarcity: initial_shared_pool scale factors
    for scale, label in ((0.5, "scarce"), (1.5, "abundant")):
        sweeps.append(SweepSpec(
            name=f"pool_{label}",
            description=f"Shared pool scaled to {scale}x default",
            config_patch={"population": {"initial_shared_pool": 100.0 * scale}},
            tags=("scarcity",),
        ))

    # E) Larger population
    sweeps.append(SweepSpec(
        name="pop_10",
        description="10 agents instead of default 5",
        config_patch={"population": {"num_agents": 10}},
        tags=("population",),
    ))

    # F) Combined stress: high asymmetry + scarcity
    sweeps.append(SweepSpec(
        name="combined_hard",
        description="High info asymmetry + scarce pool + high uncertainty",
        config_patch={
            "layers": {"information_asymmetry": 0.6, "uncertainty_intensity": 0.3},
            "population": {"initial_shared_pool": 50.0},
        },
        tags=("combined", "observation", "uncertainty", "scarcity"),
    ))

    return sweeps
