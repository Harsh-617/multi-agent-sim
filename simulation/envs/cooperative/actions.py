"""Action types for the Cooperative archetype.

V1 action model:
  - task_type: int | None — None means IDLE; 0..N-1 means work on task type index
  - effort_amount: float [0, 1] — fraction of effort capacity; ignored (forced 0) for IDLE
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Action:
    """A single agent's action for one cooperative timestep.

    Parameters
    ----------
    task_type : int | None
        None = IDLE (no contribution).
        Non-negative integer = index of the task type to work on.
    effort_amount : float
        Fraction of effort capacity in [0, 1].
        Clamped silently; forced to 0.0 for IDLE.
    """

    task_type: int | None
    effort_amount: float = 0.0

    def __post_init__(self) -> None:
        if self.task_type is None:
            # IDLE: effort is meaningless — force to 0
            object.__setattr__(self, "effort_amount", 0.0)
        else:
            if self.task_type < 0:
                raise ValueError(
                    f"task_type must be None (IDLE) or >= 0, got {self.task_type}"
                )
            # Clamp effort to [0, 1] — spec says values outside are normalised, not rejected
            clamped = max(0.0, min(1.0, self.effort_amount))
            object.__setattr__(self, "effort_amount", clamped)

    @property
    def is_idle(self) -> bool:
        return self.task_type is None
