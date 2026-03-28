"""Sweep specifications for competitive robustness evaluation.

Each sweep variant describes a config perturbation to test policy robustness
under different competitive environment conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from simulation.config.competitive_schema import CompetitiveEnvironmentConfig


@dataclass(frozen=True, slots=True)
class CompetitiveSweepSpec:
    """One environment variant for competitive robustness testing."""

    name: str
    description: str
    config_patch: dict[str, Any]
    tags: tuple[str, ...] = ()


def apply_competitive_sweep(
    config: CompetitiveEnvironmentConfig,
    sweep: CompetitiveSweepSpec,
) -> CompetitiveEnvironmentConfig:
    """Apply a CompetitiveSweepSpec to a config, returning a new config.

    The config_patch maps dot-separated section names to field updates.
    Example: {"layers": {"information_asymmetry": 0.6}}
    The base config is never mutated.
    """
    nested_updates: dict[str, Any] = {}
    for section, updates in sweep.config_patch.items():
        current_section = getattr(config, section)
        nested_updates[section] = current_section.model_copy(update=updates)

    return config.model_copy(update=nested_updates)


def build_competitive_default_sweeps(
    config: CompetitiveEnvironmentConfig | None = None,
) -> list[tuple[str, CompetitiveSweepSpec]]:
    """Return ~8 sweep variants covering key competitive stress dimensions.

    Parameters
    ----------
    config:
        Base config (unused — kept for API symmetry). Sweeps are config-
        independent specifications.

    Returns a list of (sweep_name, sweep_spec) tuples.
    """
    sweeps: list[tuple[str, CompetitiveSweepSpec]] = []

    # A) Information asymmetry
    for ia in (0.0, 0.5, 0.9):
        spec = CompetitiveSweepSpec(
            name=f"info_asym_{ia}",
            description=f"Information asymmetry = {ia}",
            config_patch={"layers": {"information_asymmetry": ia}},
            tags=("observation",),
        )
        sweeps.append((spec.name, spec))

    # B) Uncertainty intensity
    for ui in (0.0, 0.1, 0.3):
        spec = CompetitiveSweepSpec(
            name=f"uncertainty_{ui}",
            description=f"Uncertainty intensity = {ui}",
            config_patch={"layers": {"uncertainty_intensity": ui}},
            tags=("uncertainty",),
        )
        sweeps.append((spec.name, spec))

    # C) Population size
    for n in (2, 4, 6):
        spec = CompetitiveSweepSpec(
            name=f"agents_{n}",
            description=f"Number of agents = {n}",
            config_patch={"population": {"num_agents": n}},
            tags=("population",),
        )
        sweeps.append((spec.name, spec))

    # D) Initial resources
    for res in (10.0, 20.0, 40.0):
        spec = CompetitiveSweepSpec(
            name=f"resources_{res}",
            description=f"Initial resources = {res}",
            config_patch={"population": {"initial_resources": res}},
            tags=("resources",),
        )
        sweeps.append((spec.name, spec))

    # E) Resource regeneration rate
    for regen in (0.0, 0.1, 0.5):
        spec = CompetitiveSweepSpec(
            name=f"regen_{regen}",
            description=f"Resource regeneration rate = {regen}",
            config_patch={"population": {"resource_regeneration_rate": regen}},
            tags=("regeneration",),
        )
        sweeps.append((spec.name, spec))

    # F) Gamble variance
    for gv in (0.0, 0.5, 1.0):
        spec = CompetitiveSweepSpec(
            name=f"gamble_var_{gv}",
            description=f"Gamble variance = {gv}",
            config_patch={"layers": {"gamble_variance": gv}},
            tags=("gamble",),
        )
        sweeps.append((spec.name, spec))

    # G) Combined stress: high asymmetry + scarce resources + high uncertainty
    spec = CompetitiveSweepSpec(
        name="combined_hard",
        description="High info asymmetry + scarce resources + high uncertainty",
        config_patch={
            "layers": {"information_asymmetry": 0.9, "uncertainty_intensity": 0.3},
            "population": {"initial_resources": 10.0},
        },
        tags=("combined", "observation", "uncertainty", "resources"),
    )
    sweeps.append((spec.name, spec))

    # H) Combined easy: full visibility + abundant resources + no uncertainty
    spec = CompetitiveSweepSpec(
        name="combined_easy",
        description="Full visibility + abundant resources + no uncertainty",
        config_patch={
            "layers": {"information_asymmetry": 0.0, "uncertainty_intensity": 0.0},
            "population": {"initial_resources": 40.0},
        },
        tags=("combined", "observation", "uncertainty", "resources"),
    )
    sweeps.append((spec.name, spec))

    return sweeps
