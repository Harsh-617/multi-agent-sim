"""Elo rating system for league members.

Compares league members pairwise by running each against the same baseline
opponent mix, then updating Elo ratings based on relative performance.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from simulation.agents import create_agent
from simulation.agents.base import BaseAgent
from simulation.config.defaults import default_config
from simulation.core.seeding import derive_seed
from simulation.envs.mixed.env import MixedEnvironment
from simulation.league.registry import LeagueRegistry

START_RATING = 1000.0
K_FACTOR = 32.0


# ------------------------------------------------------------------
# Elo math
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
# Single evaluation episode (lightweight)
# ------------------------------------------------------------------

def _run_eval_episode(
    agent: BaseAgent,
    seed: int,
) -> float:
    """Run one episode with *agent* as agent_0 vs baseline opponents.

    Returns the total return for agent_0.
    """
    config = default_config(seed=seed)
    # Use small config for speed
    config.population.num_agents = 3
    config.population.max_steps = 50
    env = MixedEnvironment(config)
    obs = env.reset()

    agent_ids = list(obs.keys())
    eval_id = agent_ids[0]
    opp_ids = agent_ids[1:]

    agent.reset(eval_id, seed)

    # Create baseline opponents (tit_for_tat for deterministic behaviour)
    opponents: dict[str, BaseAgent] = {}
    for i, oid in enumerate(opp_ids):
        opp = create_agent("tit_for_tat")
        opp.reset(oid, derive_seed(seed, i + 100))
        opponents[oid] = opp

    total_return = 0.0
    while not env.is_done():
        actions: dict[str, Any] = {}
        if eval_id in obs:
            actions[eval_id] = agent.act(obs[eval_id])
        for oid in opp_ids:
            if oid in obs:
                actions[oid] = opponents[oid].act(obs[oid])

        results = env.step(actions)
        obs = {aid: sr.observation for aid, sr in results.items()}
        if eval_id in results:
            total_return += results[eval_id].reward

    return total_return


# ------------------------------------------------------------------
# Compute ratings
# ------------------------------------------------------------------

def compute_ratings(
    registry: LeagueRegistry,
    num_matches: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Elo ratings for all league members.

    Each pair of members plays *num_matches* rounds.  In each round both
    members are evaluated against the same baseline opponents (same seed).
    The member with the higher total return wins the round.

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

            agent_a = _make_league_agent(registry, id_a)
            agent_b = _make_league_agent(registry, id_b)

            ret_a = _run_eval_episode(agent_a, ep_seed)
            ret_b = _run_eval_episode(agent_b, ep_seed)

            if ret_a > ret_b:
                score_a = 1.0
            elif ret_b > ret_a:
                score_a = 0.0
            else:
                score_a = 0.5

            ratings[id_a], ratings[id_b] = elo_update(
                ratings[id_a], ratings[id_b], score_a
            )

    return ratings


def _make_league_agent(registry: LeagueRegistry, member_id: str) -> BaseAgent:
    """Create an agent from a league snapshot."""
    member_dir = registry.load_member(member_id)
    return create_agent("league_snapshot", member_dir=member_dir)


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def save_ratings(path: str | Path, ratings: dict[str, float]) -> None:
    """Write ratings dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Store as sorted list for stable output
    data = [
        {"member_id": mid, "rating": round(r, 2)}
        for mid, r in sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    ]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_ratings(path: str | Path) -> list[dict[str, Any]]:
    """Load ratings from JSON file. Returns empty list if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))
