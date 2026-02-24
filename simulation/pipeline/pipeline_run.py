"""End-to-end pipeline orchestrator.

Stages (in order):
  1. loading_config   – load / validate MixedEnvironmentConfig
  2. training         – league PPO with periodic snapshots
  3. ratings          – recompute Elo ratings and persist ratings.json
  4. robustness       – evaluate policy set across environment sweeps
  5. saving           – write storage/pipelines/<id>/summary.json
  6. done             – pipeline completed

Usage::

    from simulation.pipeline.pipeline_run import run_pipeline
    summary_dir = run_pipeline("default", seed=42, total_timesteps=50_000)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from simulation.config.defaults import default_config
from simulation.config.schema import MixedEnvironmentConfig
from simulation.evaluation.reporting import write_robustness_report
from simulation.evaluation.policy_set import resolve_policy_set
from simulation.evaluation.robustness import evaluate_robustness
from simulation.evaluation.sweeps import build_default_sweeps
from simulation.league.ratings import compute_ratings, save_ratings
from simulation.league.registry import LeagueRegistry
from simulation.training.ppo_shared import PPOConfig, train

# ---------------------------------------------------------------------------
# Default storage locations
# ---------------------------------------------------------------------------

_AGENTS_DIR = Path("storage/agents")
_PPO_AGENT_DIR = _AGENTS_DIR / "ppo_shared"
_PIPELINES_DIR = Path("storage/pipelines")
_CONFIGS_DIR = Path("storage/configs")
_REPORTS_DIR = Path("storage/reports")

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
# Pipeline summary helper
# ---------------------------------------------------------------------------

def _pipeline_id(config_id: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    h = hashlib.sha256(f"{config_id}:{seed}".encode()).hexdigest()[:8]
    return f"pipeline_{ts}_{h}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    config_id: str = "default",
    *,
    seed: int = 42,
    seeds: int = 3,
    episodes_per_seed: int = 2,
    max_steps: int | None = None,
    total_timesteps: int = 50_000,
    snapshot_every_timesteps: int = 10_000,
    max_league_members: int = 50,
    num_matches: int = 10,
    limit_sweeps: int | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
    # Overridable for test isolation
    ppo_agent_dir: Path = _PPO_AGENT_DIR,
    pipelines_dir: Path = _PIPELINES_DIR,
    configs_dir: Path = _CONFIGS_DIR,
    reports_dir: Path = _REPORTS_DIR,
) -> Path:
    """Run the full automation pipeline and return the summary directory.

    Parameters
    ----------
    config_id:
        ``"default"`` or a config ID saved in ``storage/configs/{id}.json``.
    seed:
        Master seed for training and evaluation determinism.
    seeds:
        Number of evaluation seeds (``base_seed``, ``base_seed+1``, …).
    episodes_per_seed:
        Episodes per seed per policy per robustness sweep.
    max_steps:
        Optional environment ``max_steps`` override for faster evaluation.
    total_timesteps:
        Total environment steps for league PPO training.
    snapshot_every_timesteps:
        How often to snapshot the current policy into the league.
    max_league_members:
        Maximum league members to retain (oldest dropped first).
    num_matches:
        Number of Elo rating matches per member pair.
    limit_sweeps:
        Cap the number of robustness sweeps for faster runs.
    progress_callback:
        Optional ``callback(stage, detail)`` called at each pipeline stage.
    ppo_agent_dir:
        Destination for PPO artifacts (``policy.pt`` + ``metadata.json``).
        League snapshots land in ``ppo_agent_dir.parent / "league"``.
    pipelines_dir:
        Root directory for pipeline summary artifacts.
    configs_dir:
        Directory containing saved ``{config_id}.json`` files.
    reports_dir:
        Root directory for robustness evaluation reports.

    Returns
    -------
    Path
        Directory containing ``summary.json`` for this pipeline run.
    """

    def _notify(stage: str, detail: str = "") -> None:
        if progress_callback is not None:
            progress_callback(stage, detail)

    # Derived paths (consistent whether real or test-isolated)
    agents_dir = ppo_agent_dir.parent
    league_root = agents_dir / "league"
    ratings_path = league_root / "ratings.json"

    # ------------------------------------------------------------------
    # Stage 1: Load config
    # ------------------------------------------------------------------
    _notify("loading_config", config_id)

    if config_id == "default":
        config = default_config(seed=seed)
        config_dict: dict[str, Any] = json.loads(config.model_dump_json())
    else:
        config_path = configs_dir / f"{config_id}.json"
        raw = config_path.read_text(encoding="utf-8")
        config = MixedEnvironmentConfig.model_validate_json(raw)
        config_dict = json.loads(raw)

    base_seed = config.identity.seed
    eval_seeds = [base_seed + i for i in range(seeds)]

    # ------------------------------------------------------------------
    # Stage 2: League PPO training with snapshotting
    # ------------------------------------------------------------------
    _notify("training", f"total_timesteps={total_timesteps}")

    ppo_cfg = PPOConfig(
        total_timesteps=total_timesteps,
        seed=seed,
        league_training=True,
        snapshot_every_timesteps=snapshot_every_timesteps,
        max_league_members=max_league_members,
        save_dir=str(agents_dir),
        agent_id="ppo_shared",
    )
    train(config, ppo_cfg)

    # ------------------------------------------------------------------
    # Stage 3: Recompute Elo ratings
    # ------------------------------------------------------------------
    _notify("ratings", "computing Elo ratings")

    registry = LeagueRegistry(league_root=str(league_root))
    ratings: dict[str, float] = compute_ratings(
        registry,
        num_matches=num_matches,
        seed=seed,
    )
    save_ratings(ratings_path, ratings)

    members = registry.list_members()
    ratings_map = {r["member_id"]: r for r in registry.list_members()}
    champion = _find_champion(members, ratings)
    champion_id = champion["member_id"] if champion else None
    champion_rating = ratings.get(champion_id, _DEFAULT_RATING) if champion_id else None

    # ------------------------------------------------------------------
    # Stage 4: Robustness evaluation
    # ------------------------------------------------------------------
    _notify("robustness", "building policy set")

    policy_specs = resolve_policy_set(
        league_root=league_root,
        ratings_path=ratings_path,
        ppo_dir=ppo_agent_dir,
        top_k=2,
    )

    sweeps = build_default_sweeps()
    if limit_sweeps is not None:
        sweeps = sweeps[:limit_sweeps]

    _notify("robustness", f"running {len(sweeps)} sweeps")

    robustness_result = evaluate_robustness(
        config,
        policy_specs,
        sweeps,
        seeds=eval_seeds,
        episodes_per_seed=episodes_per_seed,
        max_steps_override=max_steps,
    )

    # ------------------------------------------------------------------
    # Stage 5: Write robustness report
    # ------------------------------------------------------------------
    _notify("saving", "writing robustness report")

    report_dir = write_robustness_report(
        robustness_result,
        config_dict=config_dict,
        report_root=reports_dir,
    )
    report_id = report_dir.name

    # ------------------------------------------------------------------
    # Stage 6: Write pipeline summary
    # ------------------------------------------------------------------
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
        "ratings": {
            "champion_id": champion_id,
            "champion_rating": champion_rating,
            "num_members_rated": len(ratings),
            "num_matches": num_matches,
        },
        "robustness": {
            "report_id": report_id,
            "report_dir": str(report_dir),
            "n_sweeps": len(sweeps),
            "n_policies": len(policy_specs),
        },
        # Top-level shortcuts for easy consumption
        "report_id": report_id,
        "summary_path": str(out_dir / "summary.json"),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _notify("done", pid)
    return out_dir
