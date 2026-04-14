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

from simulation.config.cooperative_defaults import default_cooperative_config
from simulation.core.seeding import derive_seed
from simulation.league.cooperative_registry import CooperativeLeagueRegistry

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
    max_steps: int = 200,
) -> float:
    """Run one cooperative episode with the member policy and return completion_ratio.

    Uses CooperativeEnvironment directly (no PettingZoo adapter) for reliable
    episode termination — mirrors the ``while not env.is_done()`` pattern in
    simulation/league/ratings.py.  A hard ``step < max_steps`` guard guarantees
    finite runtime even if the environment's own termination signal is delayed.
    """
    from simulation.agents.cooperative_baselines import CooperativePPOAgent
    from simulation.envs.cooperative.env import CooperativeEnvironment

    config = default_cooperative_config(seed=seed)
    config.population.max_steps = max_steps
    config.population.num_agents = 3

    env = CooperativeEnvironment(config)
    obs = env.reset(seed=seed)

    agent_ids = list(obs.keys())
    agents: dict[str, CooperativePPOAgent] = {}
    for i, aid in enumerate(agent_ids):
        a = CooperativePPOAgent(agent_dir=member_dir)
        a.reset(aid, derive_seed(seed, i + 100))
        agents[aid] = a

    step = 0
    while not env.is_done() and step < max_steps:
        active = env.active_agents()
        actions: dict[str, Any] = {}
        for aid in active:
            actions[aid] = agents[aid].act(obs.get(aid, {}))
        results = env.step(actions)
        obs = {aid: sr.observation for aid, sr in results.items()}
        step += 1

    state = env._state
    if state is not None:
        total_completed = sum(state.tasks_completed_total)
        total_work = total_completed + state.backlog_level
        return float(total_completed) / max(float(total_work), 1.0)
    return 0.0


# ------------------------------------------------------------------
# Compute ratings
# ------------------------------------------------------------------

_MAX_MEMBERS = 10
_MAX_MATCHES = 3


def compute_cooperative_ratings(
    registry: CooperativeLeagueRegistry,
    num_matches: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Elo ratings for all cooperative league members.

    Each pair of members is evaluated independently; the member with
    the higher mean_completion_ratio wins the match.

    Hard limits: evaluates only the 10 most recent members, and runs at
    most 3 matches per pair, to keep runtime bounded.

    Returns a dict mapping ``member_id`` -> ``rating``.
    """
    members = registry.list_members()
    if not members:
        return {}

    # Cap to the _MAX_MEMBERS most recent members (by created_at, falling back to id).
    members_sorted = sorted(
        members,
        key=lambda m: (m.get("created_at") or "", m["member_id"]),
        reverse=True,
    )
    members = members_sorted[:_MAX_MEMBERS]

    # Cap matches per pair.
    num_matches = min(num_matches, _MAX_MATCHES)

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
