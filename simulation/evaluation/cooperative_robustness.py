"""20-variant robustness sweep for cooperative champion agents.

Robustness score = 0.7 × mean_completion_ratio + 0.3 × worst_case_completion_ratio

Saves report to storage/reports/cooperative_robust_{hash}_{timestamp}/

Mirrors simulation/evaluation/robustness.py — adapted metric source only.
Does NOT modify robustness.py (ADR-013).
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from simulation.adapters.cooperative_pettingzoo import CooperativePettingZooParallelEnv
from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
from simulation.evaluation.cooperative_sweeps import CoopSweepSpec, apply_coop_sweep
from simulation.training.ppo_shared import SharedPolicyNetwork

_REPORTS_ROOT = Path("storage/reports")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CoopPolicyRobustness:
    """Aggregated robustness metrics for one cooperative policy across sweeps."""

    policy_name: str
    mean_completion_ratio: float = 0.0
    worst_case_completion_ratio: float = 0.0
    robustness_score: float = 0.0
    n_sweeps_evaluated: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "mean_completion_ratio": self.mean_completion_ratio,
            "worst_case_completion_ratio": self.worst_case_completion_ratio,
            "robustness_score": self.robustness_score,
            "n_sweeps_evaluated": self.n_sweeps_evaluated,
        }


@dataclass
class CoopRobustnessResult:
    """Complete cooperative robustness evaluation output."""

    metadata: dict[str, Any] = field(default_factory=dict)
    per_sweep_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_policy_robustness: dict[str, CoopPolicyRobustness] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "per_sweep_results": self.per_sweep_results,
            "per_policy_robustness": {
                k: v.to_dict() for k, v in self.per_policy_robustness.items()
            },
        }


# ---------------------------------------------------------------------------
# Single episode evaluation (cooperative completion_ratio)
# ---------------------------------------------------------------------------

def _eval_sweep_episode(
    net: SharedPolicyNetwork,
    env_config: CooperativeEnvironmentConfig,
    seed: int,
) -> float:
    """Run one episode under swept config and return completion_ratio."""
    env = CooperativePettingZooParallelEnv(env_config)
    observations, _ = env.reset(seed=seed)

    while env.agents:
        actions: dict[str, dict] = {}
        for agent_id in env.agents:
            obs_arr = observations[agent_id]
            obs_t = torch.from_numpy(obs_arr).unsqueeze(0)
            with torch.no_grad():
                logits, alpha, beta_param, _ = net(obs_t)
            at = int(logits.argmax(dim=-1).item())
            a_val = float(alpha.squeeze(-1).item())
            b_val = float(beta_param.squeeze(-1).item())
            amt = (
                (a_val - 1.0) / (a_val + b_val - 2.0)
                if a_val > 1.0 and b_val > 1.0
                else a_val / (a_val + b_val)
            )
            actions[agent_id] = {
                "task_type": at,
                "effort_amount": np.array([amt], dtype=np.float32),
            }
        observations, _, _, _, _ = env.step(actions)

    state = env._env._state
    if state is None:
        return 0.0
    total_completed = sum(state.tasks_completed_total)
    total_work = total_completed + state.backlog_level
    return float(total_completed) / max(float(total_work), 1.0)


# ---------------------------------------------------------------------------
# Main robustness evaluation
# ---------------------------------------------------------------------------

def run_cooperative_robustness(
    base_config: CooperativeEnvironmentConfig,
    agent_dir: Path,
    sweeps: list[CoopSweepSpec],
    *,
    seeds: list[int],
    episodes_per_seed: int = 2,
    policy_name: str = "cooperative_champion",
    report_root: Path = _REPORTS_ROOT,
) -> Path:
    """Evaluate a cooperative champion across all sweep variants.

    Robustness score = 0.7 × mean_completion_ratio + 0.3 × worst_case_completion_ratio

    Parameters
    ----------
    base_config:
        Base cooperative environment config (not mutated).
    agent_dir:
        Directory containing the champion's policy.pt + metadata.json.
    sweeps:
        List of 20 sweep variants.
    seeds:
        List of evaluation seeds.
    episodes_per_seed:
        Episodes per seed per sweep variant.
    policy_name:
        Label for the policy in the report.
    report_root:
        Root directory for report output.

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

    result = CoopRobustnessResult()

    config_dict = json.loads(base_config.model_dump_json())
    result.metadata = {
        "sweeps": [s.name for s in sweeps],
        "sweep_count": len(sweeps),
        "seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "policy_name": policy_name,
        "agent_dir": str(agent_dir),
    }

    all_completion_ratios: list[float] = []

    for sweep in sweeps:
        swept_config = apply_coop_sweep(base_config, sweep)
        sweep_ratios: list[float] = []

        for seed in seeds:
            for ep in range(episodes_per_seed):
                ep_seed = seed * 10000 + ep
                ratio = _eval_sweep_episode(net, swept_config, ep_seed)
                sweep_ratios.append(ratio)

        sweep_mean = float(statistics.mean(sweep_ratios)) if sweep_ratios else 0.0
        sweep_worst = float(min(sweep_ratios)) if sweep_ratios else 0.0

        result.per_sweep_results[sweep.name] = {
            "sweep_name": sweep.name,
            "description": sweep.description,
            "tags": list(sweep.tags),
            "mean_completion_ratio": round(sweep_mean, 4),
            "worst_case_completion_ratio": round(sweep_worst, 4),
            "n_episodes": len(sweep_ratios),
            "policy": policy_name,
        }

        all_completion_ratios.extend(sweep_ratios)

    # Aggregate per-policy robustness
    pr = CoopPolicyRobustness(policy_name=policy_name)
    if all_completion_ratios:
        overall_mean = float(statistics.mean(all_completion_ratios))
        worst_case = float(min(all_completion_ratios))
        pr.mean_completion_ratio = round(overall_mean, 4)
        pr.worst_case_completion_ratio = round(worst_case, 4)
        pr.robustness_score = round(0.7 * overall_mean + 0.3 * worst_case, 4)
        pr.n_sweeps_evaluated = len(sweeps)
    result.per_policy_robustness[policy_name] = pr

    # Write report
    cfg_hash = hashlib.sha256(
        json.dumps(config_dict, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_id = f"cooperative_robust_{cfg_hash}_{ts}"

    report_dir = report_root / report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    full_report = result.to_dict()
    full_report.update({
        "report_id": report_id,
        "kind": "cooperative_robust",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg_hash,
        "config": config_dict,
    })

    (report_dir / "summary.json").write_text(
        json.dumps(full_report, indent=2, default=str), encoding="utf-8"
    )

    return report_dir
