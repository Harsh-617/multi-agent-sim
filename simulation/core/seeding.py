"""Deterministic seeding utilities.

All randomness in the simulation must flow through seeded RNGs
so that experiments are fully reproducible given the same config + seed.
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Create a NumPy Generator from an explicit seed.

    If seed is None a fresh (non-reproducible) generator is returned.
    """
    return np.random.default_rng(seed)


def derive_seed(parent_seed: int, index: int) -> int:
    """Derive a child seed deterministically from a parent seed + index.

    Useful for giving each agent or subsystem its own RNG while
    keeping the whole experiment reproducible from one root seed.
    """
    ss = np.random.SeedSequence(parent_seed).spawn(index + 1)
    return int(ss[-1].generate_state(1)[0])
