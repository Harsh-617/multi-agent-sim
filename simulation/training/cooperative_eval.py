"""Evaluate a trained cooperative PPO policy.

Loads a saved policy from storage/agents/cooperative/ppo_shared/ and runs N
episodes, reporting mean completion_ratio, mean group_efficiency_ratio, and
mean effort_utilization.

Mirrors eval_policy.py structure, adapted for CooperativeEnvironment.
Does NOT modify eval_policy.py (ADR-013).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from simulation.adapters.cooperative_pettingzoo import CooperativePettingZooParallelEnv
from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
from simulation.training.ppo_shared import SharedPolicyNetwork


_DEFAULT_AGENT_DIR = "storage/agents/cooperative/ppo_shared"


def evaluate(
    env_config: CooperativeEnvironmentConfig,
    agent_dir: str | Path,
    num_episodes: int = 10,
    seed: int = 1000,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run evaluation episodes and return summary statistics.

    Returns a dict with keys:
      mean_completion_ratio     — fraction of total work cleared per episode
      mean_group_efficiency_ratio — mean r_efficiency component per step
      mean_effort_utilization   — mean r_individual component per step (= mean effort)
      mean_return               — mean cumulative reward per agent per episode
      std_return                — std of per-episode mean returns
      episode_returns           — list of per-episode mean returns
    """
    agent_dir = Path(agent_dir)
    metadata = json.loads((agent_dir / "metadata.json").read_text(encoding="utf-8"))
    obs_dim = metadata["obs_dim"]
    num_action_types = metadata.get("num_action_types", 4)

    net = SharedPolicyNetwork(obs_dim, num_action_types)
    net.load_state_dict(torch.load(agent_dir / "policy.pt", weights_only=True))
    net.eval()

    env = CooperativePettingZooParallelEnv(env_config)

    episode_returns: list[float] = []
    episode_completion_ratios: list[float] = []
    episode_efficiency_ratios: list[float] = []
    episode_effort_utils: list[float] = []

    for ep in range(num_episodes):
        observations, _ = env.reset(seed=seed + ep)
        ep_rewards: dict[str, float] = {a: 0.0 for a in env.agents}
        step_efficiencies: list[float] = []
        step_efforts: list[float] = []

        while env.agents:
            gym_actions: dict[str, dict[str, Any]] = {}

            for agent_id in env.agents:
                obs_arr = observations[agent_id]
                obs_t = torch.from_numpy(obs_arr).unsqueeze(0)

                with torch.no_grad():
                    logits, alpha, beta_param, _ = net(obs_t)

                if deterministic:
                    at = int(logits.argmax(dim=-1).item())
                    a_val = float(alpha.squeeze(-1).item())
                    b_val = float(beta_param.squeeze(-1).item())
                    # Mode of Beta(a,b) when a,b > 1; mean otherwise
                    if a_val > 1.0 and b_val > 1.0:
                        amt = (a_val - 1.0) / (a_val + b_val - 2.0)
                    else:
                        amt = a_val / (a_val + b_val)
                else:
                    at_t, amt_t, _, _, _ = net.get_action_and_value(obs_t)
                    at = int(at_t.item())
                    amt = float(amt_t.item())

                gym_actions[agent_id] = {
                    "task_type": at,
                    "effort_amount": np.array([amt], dtype=np.float32),
                }

            observations, rewards, terminations, truncations, infos = env.step(gym_actions)

            for agent_id, r in rewards.items():
                ep_rewards[agent_id] = ep_rewards.get(agent_id, 0.0) + r
                components = infos.get(agent_id, {}).get("reward_components", {})
                if components:
                    step_efficiencies.append(float(components.get("r_efficiency", 0.0)))
                    step_efforts.append(float(components.get("r_individual", 0.0)))

        mean_return = sum(ep_rewards.values()) / max(len(ep_rewards), 1)
        episode_returns.append(mean_return)

        # Completion ratio: tasks_completed / (tasks_completed + remaining_backlog)
        state = env._env._state
        if state is not None:
            total_completed = sum(state.tasks_completed_total)
            total_work = total_completed + state.backlog_level
            completion_ratio = float(total_completed) / max(float(total_work), 1.0)
        else:
            completion_ratio = 0.0

        episode_completion_ratios.append(completion_ratio)
        episode_efficiency_ratios.append(
            float(np.mean(step_efficiencies)) if step_efficiencies else 0.0
        )
        episode_effort_utils.append(
            float(np.mean(step_efforts)) if step_efforts else 0.0
        )

    return {
        "mean_completion_ratio": float(np.mean(episode_completion_ratios)),
        "mean_group_efficiency_ratio": float(np.mean(episode_efficiency_ratios)),
        "mean_effort_utilization": float(np.mean(episode_effort_utils)),
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "episode_returns": episode_returns,
    }


if __name__ == "__main__":
    import argparse

    from simulation.config.cooperative_defaults import default_cooperative_config

    parser = argparse.ArgumentParser(
        description="Evaluate a trained cooperative PPO policy"
    )
    parser.add_argument("--agent-dir", default=_DEFAULT_AGENT_DIR)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    cfg = default_cooperative_config(seed=42)
    results = evaluate(
        cfg,
        agent_dir=args.agent_dir,
        num_episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
    )

    print(f"Episodes:                   {args.episodes}")
    print(f"Mean return:                {results['mean_return']:.4f} ± {results['std_return']:.4f}")
    print(f"Mean completion ratio:      {results['mean_completion_ratio']:.4f}")
    print(f"Mean group efficiency:      {results['mean_group_efficiency_ratio']:.4f}")
    print(f"Mean effort utilization:    {results['mean_effort_utilization']:.4f}")
