"""End-to-end cooperative pipeline orchestrator.

Stages (in order):
  1. loading_config   – load / validate CooperativeEnvironmentConfig
  2. training         – cooperative league PPO with periodic snapshots
  3. ratings          – recompute Elo ratings and persist ratings.json
  4. evaluating       – cross-seed evaluation
  5. robustness       – 20-variant robustness sweep
  6. saving           – write storage/pipelines/cooperative_<ts>_<hash>/summary.json
  7. done             – pipeline completed

Storage paths:
  storage/agents/cooperative/league/
  storage/reports/cooperative_eval_{hash}_{ts}/
  storage/reports/cooperative_robust_{hash}_{ts}/
  storage/pipelines/cooperative_pipeline_{ts}_{hash}/

Usage::

    from simulation.pipeline.cooperative_pipeline_run import run_cooperative_pipeline
    summary_dir = run_cooperative_pipeline(seed=42, total_timesteps=50_000)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from simulation.config.cooperative_defaults import default_cooperative_config
from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
from simulation.evaluation.cooperative_eval_runner import run_cooperative_eval
from simulation.evaluation.cooperative_robustness import run_cooperative_robustness
from simulation.evaluation.cooperative_sweeps import build_cooperative_sweeps
from simulation.league.cooperative_ratings import (
    compute_cooperative_ratings,
    save_cooperative_ratings,
)
from simulation.league.cooperative_registry import CooperativeLeagueRegistry
from simulation.training.cooperative_league_train import train_cooperative_league
from simulation.training.ppo_shared import PPOConfig

# ---------------------------------------------------------------------------
# Default storage locations
# ---------------------------------------------------------------------------

_STORAGE_ROOT = Path(__file__).resolve().parent.parent.parent / "storage"
_AGENTS_DIR = _STORAGE_ROOT / "agents/cooperative"
_COOPERATIVE_PPO_DIR = _AGENTS_DIR / "ppo_shared"
_PIPELINES_DIR = _STORAGE_ROOT / "pipelines"
_CONFIGS_DIR = _STORAGE_ROOT / "configs"
_REPORTS_DIR = _STORAGE_ROOT / "reports"

_DEFAULT_RATING = 1000.0


# ---------------------------------------------------------------------------
# Champion selection (deterministic)
# ---------------------------------------------------------------------------

def _find_champion(
    members: list[dict],
    ratings: dict[str, float],
) -> dict | None:
    """Return the member with the highest Elo rating."""
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
    return f"cooperative_pipeline_{ts}_{h}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cooperative_pipeline(
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
    ppo_agent_dir: Path = _COOPERATIVE_PPO_DIR,
    pipelines_dir: Path = _PIPELINES_DIR,
    configs_dir: Path = _CONFIGS_DIR,
    reports_dir: Path = _REPORTS_DIR,
) -> Path:
    """Run the full cooperative automation pipeline.

    Parameters
    ----------
    config_id:
        ``"default"`` or a config ID saved in ``storage/configs/{id}.json``.
    seed:
        Master seed for training and evaluation determinism.
    seeds:
        Number of cross-seed evaluation seeds.
    episodes_per_seed:
        Episodes per seed per robustness sweep.
    max_steps:
        Optional environment max_steps override for faster evaluation.
    total_timesteps:
        Total PPO training steps.
    snapshot_every_timesteps:
        League snapshot interval.
    max_league_members:
        Maximum league members to retain (oldest dropped first).
    num_matches:
        Number of Elo rating matches per member pair.
    limit_sweeps:
        Cap the number of robustness sweeps for faster runs.
    progress_callback:
        Optional ``callback(stage, detail)`` called at each stage.
    ppo_agent_dir:
        Destination for PPO artifacts.
    pipelines_dir:
        Root directory for pipeline summary artifacts.
    configs_dir:
        Directory containing saved ``{config_id}.json`` files.
    reports_dir:
        Root directory for evaluation reports.

    Returns
    -------
    Path
        Directory containing ``summary.json`` for this pipeline run.
    """

    def _notify(stage: str, detail: str = "") -> None:
        if progress_callback is not None:
            progress_callback(stage, detail)

    # Derived paths
    agents_dir = ppo_agent_dir.parent  # storage/agents/cooperative
    league_root = agents_dir / "league"
    ratings_path = league_root / "ratings.json"

    # ------------------------------------------------------------------
    # Stage 1: Load config
    # ------------------------------------------------------------------
    _notify("loading_config", config_id)

    if config_id == "default":
        config = default_cooperative_config(seed=seed)
        config_dict: dict[str, Any] = json.loads(config.model_dump_json())
    else:
        config_path = configs_dir / f"{config_id}.json"
        raw = config_path.read_text(encoding="utf-8")
        config_dict = json.loads(raw)
        config = CooperativeEnvironmentConfig.model_validate(config_dict)

    if max_steps is not None:
        config.population.max_steps = max_steps

    eval_seeds = [seed + i for i in range(seeds)]

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
    train_cooperative_league(config, ppo_cfg, league_root=league_root)

    # ------------------------------------------------------------------
    # Stage 3: Recompute Elo ratings
    # ------------------------------------------------------------------
    _notify("ratings", "computing Elo ratings")

    registry = CooperativeLeagueRegistry(league_root=str(league_root))
    ratings: dict[str, float] = compute_cooperative_ratings(
        registry,
        num_matches=num_matches,
        seed=seed,
    )
    save_cooperative_ratings(ratings_path, ratings)

    members = registry.list_members()
    champion = _find_champion(members, ratings)
    champion_id = champion["member_id"] if champion else None
    champion_rating = ratings.get(champion_id, _DEFAULT_RATING) if champion_id else None

    # ------------------------------------------------------------------
    # Stage 4: Cross-seed evaluation
    # ------------------------------------------------------------------
    _notify("evaluating", f"cross-seed eval over {seeds} seed(s)")

    eval_report_dir: Path | None = None
    if ppo_agent_dir.exists():
        eval_report_dir = run_cooperative_eval(
            config,
            agent_dir=ppo_agent_dir,
            num_seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            base_seed=seed + 1000,
            report_root=reports_dir,
            policy_name="cooperative_ppo",
        )

    # ------------------------------------------------------------------
    # Stage 5: Robustness sweep
    # ------------------------------------------------------------------
    _notify("robustness", "building sweep variants")

    sweeps = build_cooperative_sweeps()
    if limit_sweeps is not None:
        sweeps = sweeps[:limit_sweeps]

    _notify("robustness", f"running {len(sweeps)} sweeps")

    robust_report_dir: Path | None = None
    champion_dir = registry.load_member(champion_id) if champion_id else None
    if champion_dir is None and ppo_agent_dir.exists():
        champion_dir = ppo_agent_dir

    if champion_dir is not None and (champion_dir / "policy.pt").exists():
        robust_report_dir = run_cooperative_robustness(
            config,
            agent_dir=champion_dir,
            sweeps=sweeps,
            seeds=eval_seeds,
            episodes_per_seed=episodes_per_seed,
            policy_name="cooperative_champion",
            report_root=reports_dir,
        )

    report_id = robust_report_dir.name if robust_report_dir else (
        eval_report_dir.name if eval_report_dir else None
    )

    # ------------------------------------------------------------------
    # Stage 6: Write pipeline summary
    # ------------------------------------------------------------------
    _notify("saving", "writing pipeline summary")

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
        "archetype": "cooperative",
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
        "evaluating": {
            "seeds": eval_seeds,
            "episodes_per_seed": episodes_per_seed,
            "eval_report_id": eval_report_dir.name if eval_report_dir else None,
        },
        "robustness": {
            "report_id": robust_report_dir.name if robust_report_dir else None,
            "n_sweeps": len(sweeps),
        },
        "report_id": report_id,
        "summary_path": str(out_dir / "summary.json"),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _notify("done", pid)
    return out_dir
