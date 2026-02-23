"""Random agent — picks actions uniformly at random (deterministic given seed)."""

from __future__ import annotations

import numpy as np

from simulation.core.seeding import make_rng
from simulation.envs.mixed.actions import Action, ActionType

from simulation.agents.base import BaseAgent

_ACTION_TYPES = [ActionType.COOPERATE, ActionType.EXTRACT, ActionType.DEFEND, ActionType.CONDITIONAL]
_AMOUNT_TYPES = {ActionType.COOPERATE, ActionType.EXTRACT}


class RandomAgent(BaseAgent):
    """Uniformly random policy, fully deterministic given its seed."""

    def __init__(self) -> None:
        self._rng: np.random.Generator | None = None

    def reset(self, agent_id: str, seed: int) -> None:
        self._rng = make_rng(seed)

    def act(self, observation: dict) -> Action:
        assert self._rng is not None, "Must call reset() before act()"
        idx = int(self._rng.integers(len(_ACTION_TYPES)))
        atype = _ACTION_TYPES[idx]
        amount = float(self._rng.uniform(0.0, 1.0)) if atype in _AMOUNT_TYPES else 0.0
        return Action(type=atype, amount=amount)
