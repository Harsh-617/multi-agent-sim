"""End-to-end competitive pipeline orchestrator.

Stages (in order):
  1. loading_config   – load / validate CompetitiveEnvironmentConfig
  2. training         – competitive PPO with league self-play + snapshots
  3. rating           – recompute Elo ratings and persist ratings.json
  4. evaluating       – competitive population eval (cross-seed)
  5. reporting        – write storage/pipelines/competitive_<ts>_<hash>/summary.json

Usage::

    from simulation.pipeline.competitive_pipeline_run import run_competitive_pipeline
    summary_dir = run_competitive_pipeline(seeds=[42], total_timesteps=50_000)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from simulation.config.competitive_defaults import default_competitive_config
from simulation.config.competitive_schema import CompetitiveEnvironmentConfig
from simulation.league.competitive_eval import run_competitive_population_eval
from simulation.league.ratings import compute_ratings, save_ratings
from simulation.league.registry import LeagueRegistry
from simulation.training.competitive_ppo import CompetitivePPOConfig, train

# ---------------------------------------------------------------------------
# Default storage locations
# ---------------------------------------------------------------------------

_AGENTS_DIR = Path("storage/agents")
_COMPETITIVE_PPO_DIR = _AGENTS_DIR / "competitive_ppo"
_PIPELINES_DIR = Path("storage/pipelines")
_CONFIGS_DIR = Path("storage/configs")

_DEFAULT_RATING = 1000.0


# ---------------------------------------------------------------------------
# Champion selection (deterministic)
# ---------------------------------------------------------------------------

def _find_champion(
    members: list[dict],
    ratings: dict[str, float],
) -> dict | None:
    """Return the member with the highest Elo rating.

    Tie-breaker: newest ``created_at`` string wins (ISO-8601 lexicographic).
    """
    if not members:
        return None
    best: dict | None = None
    best_rating = -1.0
    best_created = ""
    for m in members:
        r = ratings.get(m["member_id"], _DEFAULT_RATING)
        created = m.get("created_at") or ""
        if r > best_rating or (r == best_rating and created > best_created):
            best = m
            best_rating = r
            best_created = created
    return best


# ---------------------------------------------------------------------------
# Pipeline ID helper
# ---------------------------------------------------------------------------

def _pipeline_id(config_id: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    h = hashlib.sha256(f"{config_id}:{seed}".encode()).hexdigest()[:8]
    return f"competitive_{ts}_{h}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_competitive_pipeline(
    config_id: str = "default",
    *,
    seed: int = 42,
    seeds: list[int] | None = None,
    episodes_per_seed: int = 2,
    max_steps: int | None = None,
    total_timesteps: int = 50_000,
    snapshot_every_timesteps: int = 10_000,
    max_league_members: int = 50,
    num_matches: int = 10,
    limit_sweeps: int | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
    # Overridable for test isolation
    ppo_agent_dir: Path = _COMPETITIVE_PPO_DIR,
    pipelines_dir: Path = _PIPELINES_DIR,
    configs_dir: Path = _CONFIGS_DIR,
) -> Path:
    """Run the full competitive automation pipeline and return the summary directory.

    Parameters
    ----------
    config_id:
        ``"default"`` or a config ID saved in ``storage/configs/{id}.json``.
    seed:
        Master seed for training and evaluation determinism.
    seeds:
        Explicit list of evaluation seeds.  Defaults to ``[seed]``.
    episodes_per_seed:
        Episodes per seed per policy during population evaluation.
    max_steps:
        Optional environment ``max_steps`` override for faster evaluation.
    total_timesteps:
        Total environment steps for competitive PPO training.
    snapshot_every_timesteps:
        How often to snapshot the current policy into the league.
    max_league_members:
        Maximum league members to retain (oldest dropped first).
    num_matches:
        Number of Elo rating matches per member pair.
    limit_sweeps:
        Cap the number of policies evaluated (for faster runs).
    progress_callback:
        Optional ``callback(stage, detail)`` called at each pipeline stage.
    ppo_agent_dir:
        Destination for competitive PPO artifacts.
    pipelines_dir:
        Root directory for pipeline summary artifacts.
    configs_dir:
        Directory containing saved ``{config_id}.json`` files.

    Returns
    -------
    Path
        Directory containing ``summary.json`` for this pipeline run.
    """

    if seeds is None:
        seeds = [seed]

    def _notify(stage: str, detail: str = "") -> None:
        if progress_callback is not None:
            progress_callback(stage, detail)

    # Derived paths
    agents_dir = ppo_agent_dir.parent
    league_root = agents_dir / "competitive_league"
    ratings_path = league_root / "ratings.json"

    # ------------------------------------------------------------------
    # Stage 1: Load config
    # ------------------------------------------------------------------
    _notify("loading_config", config_id)

    if config_id == "default":
        config = default_competitive_config(seed=seed)
        config_dict: dict[str, Any] = json.loads(config.model_dump_json())
    else:
        config_path = configs_dir / f"{config_id}.json"
        raw = config_path.read_text(encoding="utf-8")
        config = CompetitiveEnvironmentConfig.model_validate_json(raw)
        config_dict = json.loads(raw)

    # Apply max_steps override if provided
    if max_steps is not None:
        config.population.max_steps = max_steps

    # ------------------------------------------------------------------
    # Stage 2: Competitive PPO training with league self-play
    # ------------------------------------------------------------------
    _notify("training", f"total_timesteps={total_timesteps}")

    ppo_cfg = CompetitivePPOConfig(
        total_timesteps=total_timesteps,
        seed=seed,
        league_selfplay=True,
        snapshot_every_timesteps=snapshot_every_timesteps,
        max_league_members=max_league_members,
        save_dir=str(agents_dir),
        agent_id="competitive_ppo",
    )
    train(config, ppo_cfg)

    # ------------------------------------------------------------------
    # Stage 3: Recompute Elo ratings
    # ------------------------------------------------------------------
    _notify("rating", "computing Elo ratings")

    registry = LeagueRegistry(league_root=str(league_root))
    ratings: dict[str, float] = compute_ratings(
        registry,
        num_matches=num_matches,
        seed=seed,
    )
    save_ratings(ratings_path, ratings)

    members = registry.list_members()
    champion = _find_champion(members, ratings)
    champion_id = champion["member_id"] if champion else None
    champion_rating = ratings.get(champion_id, _DEFAULT_RATING) if champion_id else None

    # ------------------------------------------------------------------
    # Stage 4: Competitive population evaluation (cross-seed)
    # ------------------------------------------------------------------
    _notify("evaluating", f"cross-seed eval over {len(seeds)} seed(s)")

    all_eval_results: list[dict[str, Any]] = []
    for eval_seed in seeds:
        seed_results = run_competitive_population_eval(
            config,
            episodes_per_policy=episodes_per_seed,
            seed=eval_seed,
            league_root=str(league_root),
        )
        all_eval_results.append({
            "seed": eval_seed,
            "policies": seed_results,
        })

    # ------------------------------------------------------------------
    # Stage 5: Write pipeline summary
    # ------------------------------------------------------------------
    _notify("reporting", "writing summary")

    pid = _pipeline_id(config_id, seed)
    out_dir = pipelines_dir / pid
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "pipeline_id": pid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_id": config_id,
        "config_hash": hashlib.sha256(
            json.dumps(config_dict, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12],
        "seed": seed,
        "training": {
            "total_timesteps": total_timesteps,
            "ppo_agent_dir": str(ppo_agent_dir),
            "snapshot_every_timesteps": snapshot_every_timesteps,
            "max_league_members": max_league_members,
        },
        "rating": {
            "champion_id": champion_id,
            "champion_rating": champion_rating,
            "num_members_rated": len(ratings),
            "num_matches": num_matches,
        },
        "evaluating": {
            "seeds": seeds,
            "episodes_per_seed": episodes_per_seed,
            "results": all_eval_results,
        },
        "reporting": {
            "summary_path": str(out_dir / "summary.json"),
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _notify("done", pid)
    return out_dir
