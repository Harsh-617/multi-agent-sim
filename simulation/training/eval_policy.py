"""Evaluate a trained PPO shared policy on MixedPettingZooParallelEnv.

Loads a saved policy from storage/agents/{agent_id}/ and runs N episodes,
printing average per-agent return and final shared pool level.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from simulation.adapters.pettingzoo_mixed import (
    ACTION_TYPE_ORDER,
    MixedPettingZooParallelEnv,
)
from simulation.config.schema import MixedEnvironmentConfig
from simulation.training.ppo_shared import (
    SharedPolicyNetwork,
    _flat_obs_dim,
    _flatten_obs,
)


def evaluate(
    env_config: MixedEnvironmentConfig,
    agent_dir: str | Path,
    num_episodes: int = 10,
    seed: int = 1000,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run evaluation episodes and return summary statistics.

    Returns dict with keys: mean_return, std_return, mean_final_pool,
    episode_returns, episode_final_pools.
    """
    agent_dir = Path(agent_dir)
    metadata = json.loads((agent_dir / "metadata.json").read_text(encoding="utf-8"))
    obs_dim = metadata["obs_dim"]

    net = SharedPolicyNetwork(obs_dim, len(ACTION_TYPE_ORDER))
    net.load_state_dict(torch.load(agent_dir / "policy.pt", weights_only=True))
    net.eval()

    env = MixedPettingZooParallelEnv(env_config)

    episode_returns: list[float] = []
    episode_final_pools: list[float] = []

    for ep in range(num_episodes):
        observations, _ = env.reset(seed=seed + ep)
        ep_rewards: dict[str, float] = {a: 0.0 for a in env.agents}

        while env.agents:
            gym_actions: dict[str, dict[str, Any]] = {}
            for agent_id in env.agents:
                obs_flat = _flatten_obs(observations[agent_id])
                obs_t = torch.from_numpy(obs_flat).unsqueeze(0)

                with torch.no_grad():
                    logits, alpha, beta_param, value = net(obs_t)

                if deterministic:
                    at = logits.argmax(dim=-1).item()
                    # Mode of Beta(a, b) = (a-1)/(a+b-2) when a,b > 1
                    a_val = alpha.squeeze(-1).item()
                    b_val = beta_param.squeeze(-1).item()
                    if a_val > 1 and b_val > 1:
                        amt = (a_val - 1) / (a_val + b_val - 2)
                    else:
                        amt = a_val / (a_val + b_val)
                else:
                    at_t, amt_t, _, _, _ = net.get_action_and_value(obs_t)
                    at = at_t.item()
                    amt = amt_t.item()

                gym_actions[agent_id] = {
                    "action_type": at,
                    "amount": np.array([amt], dtype=np.float32),
                }

            observations, rewards, terminations, truncations, infos = env.step(gym_actions)
            for agent_id, r in rewards.items():
                ep_rewards[agent_id] = ep_rewards.get(agent_id, 0.0) + r

        mean_return = sum(ep_rewards.values()) / max(len(ep_rewards), 1)
        episode_returns.append(mean_return)

        # Get final pool from the last observation
        # After env is done, we can read from the raw env state
        final_pool = env._env._state.shared_pool if env._env._state else 0.0
        episode_final_pools.append(final_pool)

    results = {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_final_pool": float(np.mean(episode_final_pools)),
        "episode_returns": episode_returns,
        "episode_final_pools": episode_final_pools,
    }
    return results


if __name__ == "__main__":
    import argparse

    from simulation.config.defaults import default_config

    parser = argparse.ArgumentParser(description="Evaluate trained PPO policy")
    parser.add_argument("--agent-dir", default="storage/agents/ppo_shared")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    cfg = default_config(seed=42)
    results = evaluate(
        cfg,
        agent_dir=args.agent_dir,
        num_episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
    )

    print(f"Episodes:         {args.episodes}")
    print(f"Mean return:      {results['mean_return']:.4f} ± {results['std_return']:.4f}")
    print(f"Mean final pool:  {results['mean_final_pool']:.2f}")
