"""Entry point to train a cooperative PPO agent.

Mirrors ppo_shared.py's non-league training loop structure, adapted for:
  - CooperativePettingZooParallelEnv  (flat Box obs, Dict action with task_type/effort_amount)
  - CooperativeEnvironmentConfig
  - Artifact storage under storage/agents/cooperative/ppo_shared/

Reuses from ppo_shared (unchanged): SharedPolicyNetwork, RolloutBuffer, PPOConfig.
Does NOT modify ppo_shared.py (ADR-003, ADR-013).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from simulation.adapters.cooperative_pettingzoo import CooperativePettingZooParallelEnv
from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
from simulation.training.ppo_shared import PPOConfig, RolloutBuffer, SharedPolicyNetwork


# ---------------------------------------------------------------------------
# Cooperative-specific obs / action helpers
# ---------------------------------------------------------------------------

def _flat_obs_dim_coop(env: CooperativePettingZooParallelEnv) -> int:
    """Observation dimension for cooperative env (Box space, already flat)."""
    agent = env.possible_agents[0]
    space = env.observation_space(agent)
    return int(np.prod(space.shape))


def _num_action_types_coop(env: CooperativePettingZooParallelEnv) -> int:
    """Discrete action count: T task types + 1 IDLE sentinel = T+1."""
    agent = env.possible_agents[0]
    return int(env.action_space(agent)["task_type"].n)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    env_config: CooperativeEnvironmentConfig,
    ppo_cfg: PPOConfig | None = None,
) -> Path:
    """Run PPO training for the cooperative archetype and return artifact path.

    Adapts only config loading and adapter instantiation relative to
    ppo_shared.train(); the PPO update math is identical (reused via imports).
    """
    if ppo_cfg is None:
        ppo_cfg = PPOConfig(
            save_dir="storage/agents/cooperative",
            agent_id="ppo_shared",
        )

    torch.manual_seed(ppo_cfg.seed)
    np.random.seed(ppo_cfg.seed)
    device = torch.device(ppo_cfg.device)

    # --- Environment & spaces ---
    env = CooperativePettingZooParallelEnv(env_config)
    obs_dim = _flat_obs_dim_coop(env)
    num_action_types = _num_action_types_coop(env)

    # --- Network + optimizer (same architecture as Mixed PPO) ---
    net = SharedPolicyNetwork(obs_dim, num_action_types).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5)

    # --- Optional TensorBoard ---
    writer = None
    if ppo_cfg.tb_log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=ppo_cfg.tb_log_dir)

    # --- Optional progress bar ---
    try:
        from tqdm import tqdm
        pbar = tqdm(total=ppo_cfg.total_timesteps, desc="Cooperative PPO training")
    except ImportError:
        pbar = None

    global_step = 0
    num_updates = ppo_cfg.total_timesteps // ppo_cfg.rollout_steps
    episode_returns: list[float] = []

    # ---- main loop ---------------------------------------------------------

    for update in range(1, num_updates + 1):
        # Buffer sized to hold at most rollout_steps transitions * num_agents
        max_transitions = ppo_cfg.rollout_steps * len(env.possible_agents)
        buf = RolloutBuffer(max_transitions, obs_dim, device=str(device))

        # Start a fresh episode at the beginning of each rollout
        observations, _ = env.reset(seed=ppo_cfg.seed + update)
        ep_rewards: dict[str, float] = {a: 0.0 for a in env.agents}

        for _step in range(ppo_cfg.rollout_steps):
            # Handle episode boundary within a rollout
            if not env.agents:
                if ep_rewards:
                    episode_returns.append(
                        sum(ep_rewards.values()) / max(len(ep_rewards), 1)
                    )
                observations, _ = env.reset(
                    seed=ppo_cfg.seed + update + _step + 1000
                )
                ep_rewards = {a: 0.0 for a in env.agents}
                if not env.agents:
                    break

            # Build actions for every active agent
            gym_actions: dict[str, dict[str, Any]] = {}
            agent_data: dict[str, tuple] = {}

            for agent_id in env.agents:
                # Cooperative adapter returns flat np.ndarray — no extra flattening needed
                obs_arr = observations[agent_id]
                obs_t = torch.from_numpy(obs_arr).to(device)

                with torch.no_grad():
                    at, amt, lp, _, val = net.get_action_and_value(obs_t.unsqueeze(0))

                at_item = at.item()
                amt_item = amt.item()

                # Cooperative action keys: task_type / effort_amount
                gym_actions[agent_id] = {
                    "task_type": at_item,
                    "effort_amount": np.array([amt_item], dtype=np.float32),
                }
                agent_data[agent_id] = (obs_t, at_item, amt_item, lp.item(), val.item())

            next_obs, rewards, terminations, truncations, infos = env.step(gym_actions)

            for agent_id, (obs_t, at_item, amt_item, lp, val) in agent_data.items():
                r = rewards.get(agent_id, 0.0)
                done = terminations.get(agent_id, False) or truncations.get(agent_id, False)
                buf.add(obs_t, at_item, amt_item, lp, r, float(done), val)
                ep_rewards[agent_id] = ep_rewards.get(agent_id, 0.0) + r

            observations = next_obs
            global_step += len(agent_data)

        if pbar:
            pbar.update(ppo_cfg.rollout_steps)

        if buf.ptr == 0:
            continue

        # Bootstrap value for GAE
        if env.agents and observations:
            first_agent = env.agents[0]
            obs_arr = observations[first_agent]
            obs_t = torch.from_numpy(obs_arr).to(device)
            with torch.no_grad():
                last_val = net.get_value(obs_t.unsqueeze(0)).item()
        else:
            last_val = 0.0

        buf.compute_gae(last_val, ppo_cfg.gamma, ppo_cfg.gae_lambda)

        # PPO update (same math as ppo_shared.py's inline update)
        for _epoch in range(ppo_cfg.ppo_epochs):
            for (mb_obs, mb_at, mb_amt, mb_old_lp, mb_adv, mb_ret) in buf.get_batches(
                ppo_cfg.num_minibatches
            ):
                _, _, new_lp, entropy, new_val = net.get_action_and_value(
                    mb_obs, mb_at, mb_amt
                )
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1 - ppo_cfg.clip_eps, 1 + ppo_cfg.clip_eps
                ) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((new_val - mb_ret) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + ppo_cfg.vf_coef * value_loss
                    + ppo_cfg.ent_coef * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()

        if writer and episode_returns:
            writer.add_scalar(
                "charts/mean_episode_return",
                np.mean(episode_returns[-10:]),
                global_step,
            )

    if pbar:
        pbar.close()
    if writer:
        writer.close()

    save_path = _save_cooperative_artifacts(
        net, env_config, ppo_cfg, obs_dim, global_step, num_action_types
    )
    return save_path


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------

def _save_cooperative_artifacts(
    net: SharedPolicyNetwork,
    env_config: CooperativeEnvironmentConfig,
    ppo_cfg: PPOConfig,
    obs_dim: int,
    training_steps: int,
    num_action_types: int,
) -> Path:
    """Save policy.pt and metadata.json to storage/agents/cooperative/ppo_shared/."""
    save_dir = Path(ppo_cfg.save_dir) / ppo_cfg.agent_id
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(net.state_dict(), save_dir / "policy.pt")

    config_json = env_config.model_dump_json()
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

    T = env_config.population.num_task_types
    action_mapping: dict[str, str] = {str(i): f"task_{i}" for i in range(T)}
    action_mapping[str(T)] = "idle"

    metadata: dict[str, Any] = {
        "algo": "ppo_shared",
        "archetype": "cooperative",
        "obs_dim": obs_dim,
        "num_action_types": num_action_types,
        "action_mapping": action_mapping,
        "amount_distribution": "Beta",
        "config_hash": config_hash,
        "training_steps": training_steps,
        "seed": ppo_cfg.seed,
        "hyperparameters": {
            "total_timesteps": ppo_cfg.total_timesteps,
            "rollout_steps": ppo_cfg.rollout_steps,
            "ppo_epochs": ppo_cfg.ppo_epochs,
            "num_minibatches": ppo_cfg.num_minibatches,
            "learning_rate": ppo_cfg.learning_rate,
            "gamma": ppo_cfg.gamma,
            "gae_lambda": ppo_cfg.gae_lambda,
            "clip_eps": ppo_cfg.clip_eps,
            "vf_coef": ppo_cfg.vf_coef,
            "ent_coef": ppo_cfg.ent_coef,
        },
    }

    (save_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return save_dir


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from simulation.config.cooperative_defaults import default_cooperative_config

    cfg = default_cooperative_config(seed=42)
    save_path = train(cfg)
    print(f"Artifacts saved to: {save_path}")
