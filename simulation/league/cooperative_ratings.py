"""Elo rating system for cooperative league members.

Rating metric: mean_completion_ratio from evaluation episodes (primary outcome).
Higher completion_ratio = better agent.

Mirrors simulation/league/ratings.py — adapted metric source only.
Does NOT modify ratings.py (ADR-013).
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from simulation.config.cooperative_defaults import default_cooperative_config
from simulation.core.seeding import derive_seed
from simulation.league.cooperative_registry import CooperativeLeagueRegistry
from simulation.training.ppo_shared import SharedPolicyNetwork

START_RATING = 1000.0
K_FACTOR = 32.0


# ------------------------------------------------------------------
# Elo math (identical to ratings.py)
# ------------------------------------------------------------------

def elo_expected(rating_a: float, rating_b: float) -> float:
    """Expected score for player A given both ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update(
    rating_a: float, rating_b: float, score_a: float, k: float = K_FACTOR
) -> tuple[float, float]:
    """Return updated (rating_a, rating_b) after a match.

    *score_a* is 1.0 for a win, 0.0 for a loss, 0.5 for a draw.
    """
    ea = elo_expected(rating_a, rating_b)
    eb = 1.0 - ea
    new_a = rating_a + k * (score_a - ea)
    new_b = rating_b + k * ((1.0 - score_a) - eb)
    return new_a, new_b


# ------------------------------------------------------------------
# Single evaluation episode (cooperative metric: completion_ratio)
# ------------------------------------------------------------------

def _eval_completion_ratio(
    member_dir: Path,
    seed: int,
) -> float:
    """Run one cooperative episode with the member policy and return completion_ratio.

    Uses a small default config for speed.
    """
    from simulation.adapters.cooperative_pettingzoo import CooperativePettingZooParallelEnv

    config = default_cooperative_config(seed=seed)
    # Smaller episode for speed during rating computation
    config.population.max_steps = 50
    config.population.num_agents = 3

    # Load policy
    meta = json.loads((member_dir / "metadata.json").read_text(encoding="utf-8"))
    obs_dim = meta["obs_dim"]
    num_action_types = meta.get("num_action_types", 4)

    net = SharedPolicyNetwork(obs_dim, num_action_types)
    net.load_state_dict(torch.load(member_dir / "policy.pt", weights_only=True))
    net.eval()

    env = CooperativePettingZooParallelEnv(config)
    observations, _ = env.reset(seed=seed)

    while env.agents:
        actions: dict[str, Any] = {}
        for agent_id in env.agents:
            obs_arr = observations[agent_id]
            obs_t = torch.from_numpy(obs_arr).unsqueeze(0)
            with torch.no_grad():
                logits, _, _, _ = net(obs_t)
            at = int(logits.argmax(dim=-1).item())
            actions[agent_id] = {
                "task_type": at,
                "effort_amount": np.array([0.8], dtype=np.float32),
            }
        observations, _, _, _, _ = env.step(actions)

    # Extract completion_ratio from final environment state
    state = env._env._state
    if state is not None:
        total_completed = sum(state.tasks_completed_total)
        total_work = total_completed + state.backlog_level
        return float(total_completed) / max(float(total_work), 1.0)
    return 0.0


# ------------------------------------------------------------------
# Compute ratings
# ------------------------------------------------------------------

def compute_cooperative_ratings(
    registry: CooperativeLeagueRegistry,
    num_matches: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Elo ratings for all cooperative league members.

    Each pair of members is evaluated independently; the member with
    the higher mean_completion_ratio wins the match.

    Returns a dict mapping ``member_id`` -> ``rating``.
    """
    members = registry.list_members()
    if not members:
        return {}

    member_ids = [m["member_id"] for m in members]
    ratings: dict[str, float] = {mid: START_RATING for mid in member_ids}

    if len(member_ids) == 1:
        return ratings

    for match_idx in range(num_matches):
        for id_a, id_b in combinations(member_ids, 2):
            ep_seed = derive_seed(seed, match_idx * 1000 + hash(id_a + id_b) % 999)

            dir_a = registry.load_member(id_a)
            dir_b = registry.load_member(id_b)

            ratio_a = _eval_completion_ratio(dir_a, ep_seed)
            ratio_b = _eval_completion_ratio(dir_b, ep_seed)

            if ratio_a > ratio_b:
                score_a = 1.0
            elif ratio_b > ratio_a:
                score_a = 0.0
            else:
                score_a = 0.5

            ratings[id_a], ratings[id_b] = elo_update(
                ratings[id_a], ratings[id_b], score_a
            )

    return ratings


# ------------------------------------------------------------------
# Persistence (same format as ratings.py)
# ------------------------------------------------------------------

def save_cooperative_ratings(path: str | Path, ratings: dict[str, float]) -> None:
    """Write cooperative ratings dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"member_id": mid, "rating": round(r, 2)}
        for mid, r in sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    ]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_cooperative_ratings(path: str | Path) -> list[dict[str, Any]]:
    """Load cooperative ratings from JSON file. Returns empty list if not found."""
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))
