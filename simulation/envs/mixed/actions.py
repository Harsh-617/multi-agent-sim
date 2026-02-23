"""Action types for the Mixed archetype.

V1 action model:
  - ActionType enum: COOPERATE, EXTRACT, DEFEND, CONDITIONAL
  - amount ∈ [0, 1] required for COOPERATE and EXTRACT, ignored for others
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Discrete action types available to agents."""

    COOPERATE = "cooperate"
    EXTRACT = "extract"
    DEFEND = "defend"
    CONDITIONAL = "conditional"


# Action types that require an amount parameter
_AMOUNT_TYPES = {ActionType.COOPERATE, ActionType.EXTRACT}


@dataclass(frozen=True, slots=True)
class Action:
    """A single agent's action for one timestep.

    Parameters
    ----------
    type : ActionType
        The discrete action choice.
    amount : float
        Intensity in [0, 1].  Only meaningful for COOPERATE and EXTRACT.
        Silently clamped to 0 for DEFEND and CONDITIONAL.
    """

    type: ActionType
    amount: float = 0.0

    def __post_init__(self) -> None:
        if self.type in _AMOUNT_TYPES:
            if not 0.0 <= self.amount <= 1.0:
                raise ValueError(
                    f"amount must be in [0, 1] for {self.type.value}, got {self.amount}"
                )
        else:
            # Force amount to 0 for types that don't use it
            object.__setattr__(self, "amount", 0.0)
