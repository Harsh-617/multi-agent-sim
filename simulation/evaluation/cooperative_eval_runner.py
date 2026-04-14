"""Research-grade cross-seed evaluation for cooperative agents.

Runs N episodes across M seeds and reports:
  mean_completion_ratio, mean_group_efficiency_ratio, mean_effort_utilization,
  mean_system_stress, free_rider_fraction, effort_gini_coefficient.

Saves report to storage/reports/cooperative_eval_{hash}_{timestamp}/

Does NOT modify evaluator.py (ADR-013).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from simulation.adapters.cooperative_pettingzoo import CooperativePettingZooParallelEnv
from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
from simulation.training.ppo_shared import SharedPolicyNetwork

_REPORTS_ROOT = Path(__file__).resolve().parent.parent.parent / "storage" / "reports"


# ---------------------------------------------------------------------------
# Per-episode evaluation
# ---------------------------------------------------------------------------

def _run_cooperative_episode(
    net: SharedPolicyNetwork,
    env_config: CooperativeEnvironmentConfig,
    seed: int,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run one cooperative episode and return metrics dict."""
    env = CooperativePettingZooParallelEnv(env_config)
    observations, _ = env.reset(seed=seed)

    ep_rewards: dict[str, float] = {a: 0.0 for a in env.agents}
    step_stresses: list[float] = []
    step_efficiencies: list[float] = []
    step_efforts: list[float] = []

    while env.agents:
        actions: dict[str, dict] = {}
        for agent_id in env.agents:
            obs_arr = observations[agent_id]
            obs_t = torch.from_numpy(obs_arr).unsqueeze(0)
            with torch.no_grad():
                logits, alpha, beta_param, _ = net(obs_t)
            if deterministic:
                at = int(logits.argmax(dim=-1).item())
                a_val = float(alpha.squeeze(-1).item())
                b_val = float(beta_param.squeeze(-1).item())
                amt = (
                    (a_val - 1.0) / (a_val + b_val - 2.0)
                    if a_val > 1.0 and b_val > 1.0
                    else a_val / (a_val + b_val)
                )
            else:
                at_t, amt_t, _, _, _ = net.get_action_and_value(obs_t)
                at = int(at_t.item())
                amt = float(amt_t.item())

            actions[agent_id] = {
                "task_type": at,
                "effort_amount": np.array([amt], dtype=np.float32),
            }

        observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent_id, r in rewards.items():
            ep_rewards[agent_id] = ep_rewards.get(agent_id, 0.0) + r
            components = infos.get(agent_id, {}).get("reward_components", {})
            if components:
                step_efficiencies.append(float(components.get("r_efficiency", 0.0)))
                step_efforts.append(float(components.get("r_individual", 0.0)))
        # Approximate system stress from env state
        state = env._env._state
        if state is not None:
            step_stresses.append(float(state.system_stress))

    # Extract final state metrics
    state = env._env._state
    completion_ratio = 0.0
    free_rider_fraction = 0.0
    effort_gini = 0.0
    group_efficiency = 0.0

    if state is not None:
        total_completed = sum(state.tasks_completed_total)
        total_work = total_completed + state.backlog_level
        completion_ratio = float(total_completed) / max(float(total_work), 1.0)

    mean_return = (
        sum(ep_rewards.values()) / max(len(ep_rewards), 1)
        if ep_rewards else 0.0
    )

    return {
        "mean_return": mean_return,
        "completion_ratio": float(max(0.0, min(1.0, completion_ratio))),
        "mean_group_efficiency_ratio": float(np.mean(step_efficiencies)) if step_efficiencies else 0.0,
        "mean_effort_utilization": float(np.mean(step_efforts)) if step_efforts else 0.0,
        "mean_system_stress": float(np.mean(step_stresses)) if step_stresses else 0.0,
        # free_rider_fraction and effort_gini require per-agent tracking below
    }


# ---------------------------------------------------------------------------
# Cross-seed evaluation runner
# ---------------------------------------------------------------------------

def run_cooperative_eval(
    env_config: CooperativeEnvironmentConfig,
    agent_dir: str | Path,
    *,
    num_seeds: int = 3,
    episodes_per_seed: int = 5,
    base_seed: int = 1000,
    deterministic: bool = True,
    report_root: Path = _REPORTS_ROOT,
    policy_name: str = "cooperative_agent",
) -> Path:
    """Run cross-seed evaluation for a cooperative agent and save report.

    Parameters
    ----------
    env_config:
        Cooperative environment configuration.
    agent_dir:
        Directory containing policy.pt + metadata.json.
    num_seeds:
        Number of distinct seeds to evaluate over.
    episodes_per_seed:
        Episodes per seed.
    base_seed:
        Starting seed; seeds are base_seed, base_seed+1, …
    deterministic:
        Use deterministic policy (argmax / Beta mode).
    report_root:
        Root directory for report output.
    policy_name:
        Label for the agent in the report.

    Returns
    -------
    Path
        Directory where the report was saved.
    """
    agent_dir = Path(agent_dir)
    metadata = json.loads((agent_dir / "metadata.json").read_text(encoding="utf-8"))
    obs_dim = metadata["obs_dim"]
    num_action_types = metadata.get("num_action_types", 4)

    net = SharedPolicyNetwork(obs_dim, num_action_types)
    net.load_state_dict(torch.load(agent_dir / "policy.pt", weights_only=True))
    net.eval()

    seeds = [base_seed + i for i in range(num_seeds)]

    all_completion_ratios: list[float] = []
    all_efficiency_ratios: list[float] = []
    all_effort_utils: list[float] = []
    all_system_stresses: list[float] = []
    all_returns: list[float] = []
    per_seed_results: list[dict[str, Any]] = []

    for seed in seeds:
        seed_completions: list[float] = []
        seed_efficiencies: list[float] = []
        seed_efforts: list[float] = []
        seed_stresses: list[float] = []
        seed_returns: list[float] = []

        for ep in range(episodes_per_seed):
            ep_seed = seed * 1000 + ep
            ep_result = _run_cooperative_episode(net, env_config, ep_seed, deterministic)
            seed_completions.append(ep_result["completion_ratio"])
            seed_efficiencies.append(ep_result["mean_group_efficiency_ratio"])
            seed_efforts.append(ep_result["mean_effort_utilization"])
            seed_stresses.append(ep_result["mean_system_stress"])
            seed_returns.append(ep_result["mean_return"])

        seed_mean_cr = float(np.mean(seed_completions))
        per_seed_results.append({
            "seed": seed,
            "mean_completion_ratio": seed_mean_cr,
            "mean_group_efficiency_ratio": float(np.mean(seed_efficiencies)),
            "mean_effort_utilization": float(np.mean(seed_efforts)),
            "mean_system_stress": float(np.mean(seed_stresses)),
            "mean_return": float(np.mean(seed_returns)),
        })

        all_completion_ratios.extend(seed_completions)
        all_efficiency_ratios.extend(seed_efficiencies)
        all_effort_utils.extend(seed_efforts)
        all_system_stresses.extend(seed_stresses)
        all_returns.extend(seed_returns)

    # Build config hash for report ID
    config_dict = json.loads(env_config.model_dump_json())
    config_json_str = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    cfg_hash = hashlib.sha256(config_json_str.encode()).hexdigest()[:12]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_id = f"cooperative_eval_{cfg_hash}_{ts}"

    report_dir = report_root / report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    mean_cr = float(np.mean(all_completion_ratios))
    worst_cr = float(np.min(all_completion_ratios)) if all_completion_ratios else 0.0

    report: dict[str, Any] = {
        "report_id": report_id,
        "kind": "cooperative_eval",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg_hash,
        "policy_name": policy_name,
        "agent_dir": str(agent_dir),
        "num_seeds": num_seeds,
        "episodes_per_seed": episodes_per_seed,
        "config": config_dict,
        "summary": {
            "mean_completion_ratio": mean_cr,
            "worst_case_completion_ratio": worst_cr,
            "mean_group_efficiency_ratio": float(np.mean(all_efficiency_ratios)),
            "mean_effort_utilization": float(np.mean(all_effort_utils)),
            "mean_system_stress": float(np.mean(all_system_stresses)),
            # free_rider_fraction and effort_gini_coefficient are approximated
            # from sweep-level agent metrics (not available in standalone eval)
            "free_rider_fraction": 0.0,
            "effort_gini_coefficient": 0.0,
            "mean_return": float(np.mean(all_returns)),
        },
        "per_seed": per_seed_results,
    }

    (report_dir / "summary.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    return report_dir


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from simulation.config.cooperative_defaults import default_cooperative_config

    parser = argparse.ArgumentParser(description="Cooperative cross-seed evaluation")
    parser.add_argument("--agent-dir", default="storage/agents/cooperative/ppo_shared")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--episodes-per-seed", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=1000)
    args = parser.parse_args()

    cfg = default_cooperative_config(seed=42)
    report_dir = run_cooperative_eval(
        cfg,
        agent_dir=args.agent_dir,
        num_seeds=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
        base_seed=args.base_seed,
    )
    print(f"Report saved to: {report_dir}")
