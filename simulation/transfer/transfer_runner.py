"""Transfer experiment runner.

Loads a trained champion policy from one archetype's league and evaluates it
inside a different archetype's environment, applying truncation/padding to
bridge the observation dimension mismatch.

Entry point:
    run_transfer_experiment(source_archetype, source_member_id,
                             target_archetype, target_config_id,
                             episodes, seed) -> dict
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.storage_root import STORAGE_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry / storage paths (mirrors existing route conventions)
# ---------------------------------------------------------------------------

_MIXED_LEAGUE_ROOT = STORAGE_ROOT / "agents/league"
_COMPETITIVE_LEAGUE_ROOT = STORAGE_ROOT / "agents/competitive_league"
_COOPERATIVE_LEAGUE_ROOT = STORAGE_ROOT / "agents/cooperative/league"
_CONFIGS_DIR = STORAGE_ROOT / "configs"
_REPORTS_ROOT = STORAGE_ROOT / "reports"


# ---------------------------------------------------------------------------
# Observation flattening helpers
# ---------------------------------------------------------------------------

def _flatten_obs_for_archetype(obs: dict[str, Any], archetype: str) -> np.ndarray:
    """Flatten a native env observation dict to a 1-D float32 array.

    Each archetype has its own observation layout; this dispatches to the
    appropriate flattener so the source policy's network sees a consistent
    flat vector regardless of which target environment produced it.
    """
    if archetype == "cooperative":
        vec = obs.get("obs_vector")
        if vec is not None:
            return np.asarray(vec, dtype=np.float32).flatten()
        return np.zeros(0, dtype=np.float32)

    if archetype == "mixed":
        from simulation.agents.ppo_shared_agent import _flatten_obs
        return _flatten_obs(obs)

    if archetype == "competitive":
        from simulation.agents.competitive_ppo_agent import _flatten_obs
        return _flatten_obs(obs)

    return np.zeros(0, dtype=np.float32)


def _adapt_obs(flat_obs: np.ndarray, source_obs_dim: int) -> np.ndarray:
    """Truncate or zero-pad *flat_obs* to *source_obs_dim* floats."""
    n = len(flat_obs)
    if n == source_obs_dim:
        return flat_obs
    if n > source_obs_dim:
        return flat_obs[:source_obs_dim]
    # n < source_obs_dim — zero-pad
    padded = np.zeros(source_obs_dim, dtype=np.float32)
    padded[:n] = flat_obs
    return padded


def _mismatch_strategy(source_obs_dim: int, target_obs_dim: int) -> str:
    if target_obs_dim > source_obs_dim:
        return "truncate"
    if target_obs_dim < source_obs_dim:
        return "pad"
    return "none"


# ---------------------------------------------------------------------------
# Target obs-dim probe
# ---------------------------------------------------------------------------

def _probe_target_obs_dim(env: Any, target_archetype: str) -> int:
    """Return the flat observation dimension produced by *env*.

    All three archetypes expose an obs_dim() method that computes the
    dimension analytically from config — immune to empty histories at reset.
    """
    return env.obs_dim()


# ---------------------------------------------------------------------------
# Action inference — run source network on adapted obs, map to target format
# ---------------------------------------------------------------------------

def _infer_target_action(
    source_net: Any,
    adapted_obs: np.ndarray,
    target_archetype: str,
    target_env: Any,
) -> Any:
    """Run *source_net* on *adapted_obs* and return a target-archetype action.

    The source SharedPolicyNetwork always outputs:
        logits  — discrete action logits
        alpha   — Beta distribution alpha
        beta    — Beta distribution beta
    We deterministically pick the argmax action and the Beta mean for the
    continuous parameter, then map to the target env's action space.
    """
    import torch

    obs_t = torch.from_numpy(adapted_obs).unsqueeze(0)
    with torch.no_grad():
        logits, alpha, beta_param, _value = source_net(obs_t)

    action_idx = int(logits.argmax(dim=-1).item())
    amount = float((alpha / (alpha + beta_param)).squeeze().item())
    amount = float(np.clip(amount, 0.0, 1.0))

    if target_archetype == "mixed":
        from simulation.envs.mixed.actions import Action as MixedAction, ActionType
        action_types = [
            ActionType.COOPERATE,
            ActionType.EXTRACT,
            ActionType.DEFEND,
            ActionType.CONDITIONAL,
        ]
        chosen = action_types[action_idx % len(action_types)]
        return MixedAction(type=chosen, amount=amount)

    if target_archetype == "competitive":
        from simulation.envs.competitive.actions import Action as CompAction, ActionType
        action_types = [
            ActionType.BUILD,
            ActionType.ATTACK,
            ActionType.DEFEND,
            ActionType.GAMBLE,
        ]
        chosen = action_types[action_idx % len(action_types)]
        return CompAction(type=chosen, amount=amount)

    if target_archetype == "cooperative":
        from simulation.envs.cooperative.actions import Action as CoopAction
        T = target_env._config.population.num_task_types
        n_choices = T + 1  # 0..T-1 = task type, T = IDLE
        idx = action_idx % n_choices
        if idx >= T:
            return CoopAction(task_type=None)
        return CoopAction(task_type=idx, effort_amount=amount)

    raise ValueError(f"Unknown target archetype: {target_archetype!r}")


# ---------------------------------------------------------------------------
# Primary metric helpers
# ---------------------------------------------------------------------------

def _compute_cooperative_episode_metric(env: Any) -> dict[str, Any]:
    """Compute completion_ratio for a completed cooperative episode."""
    state = env._state
    if state is None:
        return {"completion_ratio": 0.0}
    total_completed = float(sum(state.tasks_completed_total))
    backlog = float(state.backlog_level)
    completion_ratio = total_completed / max(total_completed + backlog, 1.0)
    return {"completion_ratio": round(completion_ratio, 6)}


def _compute_mixed_episode_metric(episode_actions: dict[str, int]) -> dict[str, Any]:
    """Compute cooperation_rate from tallied COOPERATE vs total actions."""
    total = episode_actions.get("total", 0)
    coop = episode_actions.get("cooperate", 0)
    cooperation_rate = coop / max(total, 1)
    return {"cooperation_rate": round(cooperation_rate, 6)}


def _compute_competitive_episode_metric(env: Any, agent0_id: str) -> dict[str, Any]:
    """Compute normalized_rank for *agent0_id* in completed competitive episode.

    normalized_rank = (n - rank_0based_from_best) / n
      rank 0 (best score) → 1.0
      rank n-1 (worst score) → 1/n
    Higher is always better.
    """
    state = env._state
    if state is None:
        return {"normalized_rank": 0.0}
    rankings = state.rankings()  # sorted descending by score
    n = len(rankings)
    if n == 0:
        return {"normalized_rank": 0.0}
    rank_0based = next(
        (i for i, (aid, _) in enumerate(rankings) if aid == agent0_id),
        n - 1,
    )
    normalized_rank = (n - rank_0based) / n
    return {"normalized_rank": round(normalized_rank, 6)}


def _extract_primary_metric(result: dict[str, Any], target_archetype: str) -> float:
    """Extract the scalar primary metric from an episode result dict."""
    if target_archetype == "mixed":
        return result.get("cooperation_rate", 0.0)
    if target_archetype == "competitive":
        return result.get("normalized_rank", 0.0)
    if target_archetype == "cooperative":
        return result.get("completion_ratio", 0.0)
    return 0.0


# ---------------------------------------------------------------------------
# Source policy loader
# ---------------------------------------------------------------------------

def _load_source_registry_and_member(source_archetype: str, source_member_id: str):
    """Return (registry, member_dir, metadata) for the source champion."""
    if source_archetype == "mixed":
        from simulation.league.registry import LeagueRegistry
        registry = LeagueRegistry(_MIXED_LEAGUE_ROOT)
    elif source_archetype == "competitive":
        from simulation.league.registry import LeagueRegistry
        registry = LeagueRegistry(_COMPETITIVE_LEAGUE_ROOT)
    elif source_archetype == "cooperative":
        from simulation.league.cooperative_registry import CooperativeLeagueRegistry
        registry = CooperativeLeagueRegistry(_COOPERATIVE_LEAGUE_ROOT)
    else:
        raise ValueError(f"Unknown source_archetype: {source_archetype!r}")

    member_dir = registry.load_member(source_member_id)
    metadata = registry.get_member_metadata(source_member_id)
    return registry, member_dir, metadata


def _load_source_agent(source_archetype: str, member_dir: Path) -> Any:
    """Instantiate and load (torch weights) the appropriate PPO agent."""
    if source_archetype == "mixed":
        from simulation.agents.ppo_shared_agent import PPOSharedAgent
        agent = PPOSharedAgent(agent_dir=member_dir)
    elif source_archetype == "competitive":
        from simulation.agents.competitive_ppo_agent import CompetitivePPOAgent
        agent = CompetitivePPOAgent(agent_dir=member_dir)
    elif source_archetype == "cooperative":
        from simulation.agents.cooperative_baselines import CooperativePPOAgent
        agent = CooperativePPOAgent(agent_dir=member_dir)
    else:
        raise ValueError(f"Unknown source_archetype: {source_archetype!r}")

    agent._ensure_loaded()
    return agent


# ---------------------------------------------------------------------------
# Target environment factory
# ---------------------------------------------------------------------------

def _build_target_env(target_archetype: str, target_config_id: str) -> Any:
    """Load config and instantiate target environment.

    Returns the env instance (already validated against target_archetype).
    Raises ValueError if config environment_type mismatches target_archetype.
    """
    if target_config_id == "default":
        return _build_default_target_env(target_archetype)

    config_path = _CONFIGS_DIR / f"{target_config_id}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config {target_config_id!r} not found.")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    env_type = raw.get("identity", {}).get("environment_type", "")

    if env_type != target_archetype:
        raise ValueError(
            f"Config {target_config_id!r} has environment_type={env_type!r} "
            f"but target_archetype is {target_archetype!r}."
        )

    if target_archetype == "mixed":
        from simulation.config.schema import MixedEnvironmentConfig
        from simulation.envs.mixed.env import MixedEnvironment
        config = MixedEnvironmentConfig.model_validate(raw)
        return MixedEnvironment(config)

    if target_archetype == "competitive":
        from simulation.config.competitive_schema import CompetitiveEnvironmentConfig
        from simulation.envs.competitive.env import CompetitiveEnvironment
        config = CompetitiveEnvironmentConfig.model_validate(raw)
        return CompetitiveEnvironment(config)

    if target_archetype == "cooperative":
        from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
        from simulation.envs.cooperative.env import CooperativeEnvironment
        config = CooperativeEnvironmentConfig.model_validate(raw)
        return CooperativeEnvironment(config)

    raise ValueError(f"Unknown target_archetype: {target_archetype!r}")


def _build_default_target_env(target_archetype: str) -> Any:
    """Return a default-config environment for *target_archetype*."""
    if target_archetype == "mixed":
        from simulation.config.defaults import default_config
        from simulation.envs.mixed.env import MixedEnvironment
        return MixedEnvironment(default_config())

    if target_archetype == "competitive":
        from simulation.config.competitive_defaults import default_competitive_config
        from simulation.envs.competitive.env import CompetitiveEnvironment
        return CompetitiveEnvironment(default_competitive_config())

    if target_archetype == "cooperative":
        from simulation.config.cooperative_defaults import default_cooperative_config
        from simulation.envs.cooperative.env import CooperativeEnvironment
        return CooperativeEnvironment(default_cooperative_config())

    raise ValueError(f"Unknown target_archetype: {target_archetype!r}")


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def _run_transferred_episodes(
    source_net: Any,
    source_obs_dim: int,
    target_env: Any,
    target_archetype: str,
    episodes: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run *episodes* episodes using the transferred source policy."""
    results: list[dict[str, Any]] = []

    for ep in range(episodes):
        ep_seed = seed + ep
        obs_dict = target_env.reset(seed=ep_seed)
        agent_ids = target_env.active_agents()
        agent0_id = agent_ids[0] if agent_ids else "agent_0"

        # For mixed: track action tallies
        action_tally: dict[str, int] = {"total": 0, "cooperate": 0}

        while not target_env.is_done():
            active = target_env.active_agents()
            actions: dict[str, Any] = {}
            for aid in active:
                raw_obs = obs_dict.get(aid, {})
                flat = _flatten_obs_for_archetype(raw_obs, target_archetype)
                adapted = _adapt_obs(flat, source_obs_dim)
                action = _infer_target_action(source_net, adapted, target_archetype, target_env)
                actions[aid] = action

                # Track Mixed cooperation metric
                if target_archetype == "mixed":
                    action_tally["total"] += 1
                    from simulation.envs.mixed.actions import ActionType
                    if hasattr(action, "type") and action.type == ActionType.COOPERATE:
                        action_tally["cooperate"] += 1

            step_results = target_env.step(actions)
            obs_dict = {aid: sr.observation for aid, sr in step_results.items()}

        # Collect per-episode result
        if target_archetype == "mixed":
            ep_result = _compute_mixed_episode_metric(action_tally)
        elif target_archetype == "competitive":
            ep_result = _compute_competitive_episode_metric(target_env, agent0_id)
        else:  # cooperative
            ep_result = _compute_cooperative_episode_metric(target_env)

        results.append(ep_result)

    return results


def _run_baseline_episodes(
    target_env: Any,
    target_archetype: str,
    episodes: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run *episodes* episodes using the random baseline agent for target archetype."""
    from simulation.core.seeding import derive_seed

    results: list[dict[str, Any]] = []

    for ep in range(episodes):
        ep_seed = seed + ep
        obs_dict = target_env.reset(seed=ep_seed)
        agent_ids = target_env.active_agents()
        agent0_id = agent_ids[0] if agent_ids else "agent_0"

        # Instantiate per-episode baseline agents
        baseline_agents: dict[str, Any] = {}
        if target_archetype == "mixed":
            from simulation.agents import create_agent
            for i, aid in enumerate(agent_ids):
                a = create_agent("random")
                a.reset(aid, derive_seed(ep_seed, i))
                baseline_agents[aid] = a
        elif target_archetype == "competitive":
            from simulation.agents.competitive_baselines import create_competitive_agent
            for i, aid in enumerate(agent_ids):
                a = create_competitive_agent("random")
                a.reset(aid, derive_seed(ep_seed, i))
                baseline_agents[aid] = a
        elif target_archetype == "cooperative":
            from simulation.agents.cooperative_baselines import create_cooperative_agent
            T = target_env._config.population.num_task_types
            for i, aid in enumerate(agent_ids):
                a = create_cooperative_agent("random", num_task_types=T)
                a.reset(aid, derive_seed(ep_seed, i))
                baseline_agents[aid] = a

        action_tally: dict[str, int] = {"total": 0, "cooperate": 0}

        while not target_env.is_done():
            active = target_env.active_agents()
            actions: dict[str, Any] = {}
            for aid in active:
                raw_obs = obs_dict.get(aid, {})
                action = baseline_agents[aid].act(raw_obs)
                actions[aid] = action

                if target_archetype == "mixed":
                    action_tally["total"] += 1
                    from simulation.envs.mixed.actions import ActionType
                    if hasattr(action, "type") and action.type == ActionType.COOPERATE:
                        action_tally["cooperate"] += 1

            step_results = target_env.step(actions)
            obs_dict = {aid: sr.observation for aid, sr in step_results.items()}

        if target_archetype == "mixed":
            ep_result = _compute_mixed_episode_metric(action_tally)
        elif target_archetype == "competitive":
            ep_result = _compute_competitive_episode_metric(target_env, agent0_id)
        else:
            ep_result = _compute_cooperative_episode_metric(target_env)

        results.append(ep_result)

    return results


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------

def _config_hash(target_config_id: str) -> str:
    return hashlib.sha256(target_config_id.encode()).hexdigest()[:8]


def _save_report(
    report_dir: Path,
    summary: dict[str, Any],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_transfer_experiment(
    source_archetype: str,
    source_member_id: str,
    target_archetype: str,
    target_config_id: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Run a cross-archetype transfer experiment and return a results dict.

    Parameters
    ----------
    source_archetype:
        "mixed" | "competitive" | "cooperative"
    source_member_id:
        League registry member ID of the champion to transfer.
    target_archetype:
        Destination environment archetype (must differ from source).
    target_config_id:
        Saved config ID for the target environment, or "default".
    episodes:
        Number of episodes to run for each condition (transfer + baseline).
    seed:
        Base random seed; per-episode seeds are derived as seed + episode_idx.

    Returns
    -------
    dict matching the summary.json schema from the design doc.
    """
    # ------------------------------------------------------------------
    # 1. Load source champion policy
    # ------------------------------------------------------------------
    _registry, member_dir, metadata = _load_source_registry_and_member(
        source_archetype, source_member_id
    )
    source_obs_dim: int = metadata["obs_dim"]
    source_strategy_label: str | None = metadata.get("strategy_label")
    source_elo: float | None = metadata.get("elo") or metadata.get("rating")

    logger.info(
        "Transfer: loading %s/%s (obs_dim=%d)",
        source_archetype, source_member_id, source_obs_dim,
    )
    source_agent = _load_source_agent(source_archetype, member_dir)
    source_net = source_agent._net

    # ------------------------------------------------------------------
    # 2. Build target environment
    # ------------------------------------------------------------------
    target_env = _build_target_env(target_archetype, target_config_id)
    target_obs_dim = _probe_target_obs_dim(target_env, target_archetype)
    mismatch_strategy = _mismatch_strategy(source_obs_dim, target_obs_dim)

    logger.info(
        "Transfer: target=%s obs_dim=%d | source obs_dim=%d | strategy=%s",
        target_archetype, target_obs_dim, source_obs_dim, mismatch_strategy,
    )

    # ------------------------------------------------------------------
    # 3. Run transferred policy episodes
    # ------------------------------------------------------------------
    transferred_results = _run_transferred_episodes(
        source_net=source_net,
        source_obs_dim=source_obs_dim,
        target_env=target_env,
        target_archetype=target_archetype,
        episodes=episodes,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # 4. Run random baseline episodes
    # ------------------------------------------------------------------
    baseline_results = _run_baseline_episodes(
        target_env=target_env,
        target_archetype=target_archetype,
        episodes=episodes,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # 5. Aggregate primary metrics
    # ------------------------------------------------------------------
    t_metrics = [_extract_primary_metric(r, target_archetype) for r in transferred_results]
    b_metrics = [_extract_primary_metric(r, target_archetype) for r in baseline_results]

    transferred_mean = round(float(np.mean(t_metrics)) if t_metrics else 0.0, 6)
    baseline_mean = round(float(np.mean(b_metrics)) if b_metrics else 0.0, 6)
    vs_baseline_delta = round(transferred_mean - baseline_mean, 6)
    vs_baseline_pct = (
        round(vs_baseline_delta / baseline_mean * 100.0, 2)
        if baseline_mean != 0.0
        else None
    )

    # ------------------------------------------------------------------
    # 6. Build summary dict
    # ------------------------------------------------------------------
    config_hash = _config_hash(target_config_id)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_dir_name = (
        f"transfer_{source_archetype}_{target_archetype}_{config_hash}_{timestamp}"
    )
    report_dir = _REPORTS_ROOT / report_dir_name

    summary: dict[str, Any] = {
        "report_type": "transfer",
        "report_id": report_dir_name,
        "source_archetype": source_archetype,
        "source_member_id": source_member_id,
        "source_obs_dim": source_obs_dim,
        "source_strategy_label": source_strategy_label,
        "source_elo": source_elo,
        "target_archetype": target_archetype,
        "target_config_hash": config_hash,
        "target_obs_dim": target_obs_dim,
        "obs_mismatch_strategy": mismatch_strategy,
        "episodes": episodes,
        "seed": seed,
        "transferred_results": transferred_results,
        "baseline_results": baseline_results,
        "transferred_mean": transferred_mean,
        "baseline_mean": baseline_mean,
        "vs_baseline_delta": vs_baseline_delta,
        "vs_baseline_pct": vs_baseline_pct,
    }

    # ------------------------------------------------------------------
    # 7. Persist report
    # ------------------------------------------------------------------
    _save_report(report_dir, summary)
    logger.info("Transfer report saved to %s", report_dir)

    return summary
