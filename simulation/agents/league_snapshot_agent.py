"""LeaguePolicyAgent — loads a league snapshot for deterministic inference.

Reuses the same network architecture and observation flattening as
:class:`PPOSharedAgent`, but loads weights from a league member directory
instead of the default ``storage/agents/ppo_shared/`` path.

Torch is imported lazily.
"""

from __future__ import annotations

from pathlib import Path

from simulation.agents.ppo_shared_agent import PPOSharedAgent


class LeaguePolicyAgent(PPOSharedAgent):
    """Inference agent backed by a league member snapshot.

    Parameters
    ----------
    member_dir : Path | str
        Path to the league member folder containing ``policy.pt`` and
        ``metadata.json`` (e.g. ``storage/agents/league/league_000003``).
    deterministic : bool
        If True (default), use argmax / Beta-mean for action selection.
    """

    def __init__(
        self,
        member_dir: Path | str,
        *,
        deterministic: bool = True,
    ) -> None:
        super().__init__(agent_dir=member_dir, deterministic=deterministic)
