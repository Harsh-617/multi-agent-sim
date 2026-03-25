"""Shared-policy PPO training on CompetitivePettingZooParallelEnv.

Minimal CleanRL-style PPO with:
- Shared policy network for learning agents
- Discrete action_type (Categorical) + continuous amount (Beta)
- Baseline opponents sampled from COMPETITIVE_POLICY_REGISTRY
- Rollout collection across learning agents, then PPO update for K epochs
- Deterministic seeding
- TensorBoard logging (optional)
- Artifact export to storage/agents/competitive_ppo/
- Periodic league snapshots to storage/agents/competitive_league/
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta, Categorical

from simulation.adapters.competitive_pettingzoo import (
    ACTION_TYPE_ORDER,
    CompetitivePettingZooParallelEnv,
)
from simulation.agents.competitive_baselines import (
    COMPETITIVE_POLICY_REGISTRY,
    create_competitive_agent,
)
from simulation.config.competitive_schema import CompetitiveEnvironmentConfig


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class CompetitivePPOConfig:
    """PPO training hyper-parameters for the Competitive archetype."""

    total_timesteps: int = 50_000
    rollout_steps: int = 256
    ppo_epochs: int = 4
    num_minibatches: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    seed: int = 42
    device: str = "cpu"
    tb_log_dir: str | None = None
    save_dir: str = "storage/agents"
    agent_id: str = "competitive_ppo"

    # Competitive-specific: which baseline to use as opponent during training
    opponent_policy: str = "random"

    # Periodic league snapshots
    snapshot_every_timesteps: int = 10_000
    max_league_members: int = 50


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

OBS_KEYS = (
    "step", "own_score", "own_resources", "own_rank",
    "num_active_agents", "opponents_scores",
    "own_action_history", "opponents_recent_actions",
)


def _flat_obs_dim(env: CompetitivePettingZooParallelEnv) -> int:
    """Compute flattened observation dimension from the env spaces."""
    agent = env.possible_agents[0]
    space = env.observation_space(agent)
    total = 0
    for key, sub in space.spaces.items():
        total += int(np.prod(sub.shape))
    return total


def _flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten a Dict observation into a 1-D float32 array."""
    parts = []
    for key in OBS_KEYS:
        arr = obs[key]
        parts.append(arr.astype(np.float32).flatten())
    return np.concatenate(parts)


class CompetitivePolicyNetwork(nn.Module):
    """Actor-critic network with discrete action head + Beta amount head."""

    def __init__(self, obs_dim: int, num_action_types: int = 4, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        # Discrete action type head
        self.action_type_head = nn.Linear(hidden, num_action_types)
        # Beta distribution params for amount (alpha, beta > 0)
        self.amount_alpha_head = nn.Linear(hidden, 1)
        self.amount_beta_head = nn.Linear(hidden, 1)
        # Value head
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        logits = self.action_type_head(h)
        alpha = torch.nn.functional.softplus(self.amount_alpha_head(h)) + 1.0
        beta = torch.nn.functional.softplus(self.amount_beta_head(h)) + 1.0
        value = self.value_head(h).squeeze(-1)
        return logits, alpha, beta, value

    def get_action_and_value(
        self, x: torch.Tensor, action_type: torch.Tensor | None = None,
        amount: torch.Tensor | None = None,
    ):
        logits, alpha, beta_param, value = self.forward(x)
        cat_dist = Categorical(logits=logits)
        beta_dist = Beta(alpha.squeeze(-1), beta_param.squeeze(-1))

        if action_type is None:
            action_type = cat_dist.sample()
        if amount is None:
            amount = beta_dist.sample()

        log_prob_type = cat_dist.log_prob(action_type)
        log_prob_amount = beta_dist.log_prob(amount.clamp(1e-6, 1 - 1e-6))
        log_prob = log_prob_type + log_prob_amount

        entropy_type = cat_dist.entropy()
        entropy_amount = beta_dist.entropy()
        entropy = entropy_type + entropy_amount

        return action_type, amount, log_prob, entropy, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        return self.value_head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Flat buffer collecting transitions from learning agents."""

    def __init__(self, capacity: int, obs_dim: int, device: str = "cpu"):
        self.device = device
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.action_types = torch.zeros(capacity, dtype=torch.long, device=device)
        self.amounts = torch.zeros(capacity, device=device)
        self.log_probs = torch.zeros(capacity, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.dones = torch.zeros(capacity, device=device)
        self.values = torch.zeros(capacity, device=device)
        self.advantages = torch.zeros(capacity, device=device)
        self.returns = torch.zeros(capacity, device=device)
        self.ptr = 0
        self.capacity = capacity

    def add(self, obs, action_type, amount, log_prob, reward, done, value):
        i = self.ptr
        self.obs[i] = obs
        self.action_types[i] = action_type
        self.amounts[i] = amount
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        n = self.ptr
        last_adv = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = last_value
                next_non_terminal = 1.0 - self.dones[t].item()
            else:
                next_val = self.values[t + 1].item()
                next_non_terminal = 1.0 - self.dones[t].item()
            delta = self.rewards[t].item() + gamma * next_val * next_non_terminal - self.values[t].item()
            last_adv = delta + gamma * lam * next_non_terminal * last_adv
            self.advantages[t] = last_adv
        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, num_minibatches: int):
        n = self.ptr
        indices = torch.randperm(n, device=self.device)
        batch_size = n // num_minibatches
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield (
                self.obs[idx],
                self.action_types[idx],
                self.amounts[idx],
                self.log_probs[idx],
                self.advantages[idx],
                self.returns[idx],
            )

    def reset(self):
        self.ptr = 0


# ---------------------------------------------------------------------------
# League helpers
# ---------------------------------------------------------------------------

def _trim_league_members(registry: Any, max_members: int) -> None:
    """Delete oldest league members if count exceeds *max_members*."""
    members = registry.list_members()
    if len(members) <= max_members:
        return
    to_remove = members[: len(members) - max_members]
    for meta in to_remove:
        member_dir = registry.root / meta["member_id"]
        if member_dir.is_dir():
            shutil.rmtree(member_dir)


def _action_to_gym(action: Any) -> dict[str, Any]:
    """Convert a baseline agent's Action to a gymnasium Dict action."""
    idx = 0
    for i, at in enumerate(ACTION_TYPE_ORDER):
        if at == action.type:
            idx = i
            break
    return {
        "action_type": idx,
        "amount": np.array([action.amount], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    env_config: CompetitiveEnvironmentConfig,
    ppo_cfg: CompetitivePPOConfig | None = None,
) -> Path:
    """Run PPO training and return the path to saved artifacts."""
    if ppo_cfg is None:
        ppo_cfg = CompetitivePPOConfig()

    # Seeding
    torch.manual_seed(ppo_cfg.seed)
    np.random.seed(ppo_cfg.seed)

    device = torch.device(ppo_cfg.device)

    # Environment
    env = CompetitivePettingZooParallelEnv(env_config)
    obs_dim = _flat_obs_dim(env)
    num_action_types = len(ACTION_TYPE_ORDER)

    # Identify the learning agent (agent_0) vs opponents
    learning_agent = env.possible_agents[0]

    # Create baseline opponent
    opponent_agents: dict[str, Any] = {}

    def _reset_opponents(seed_base: int) -> None:
        opponent_agents.clear()
        for i, aid in enumerate(env.possible_agents):
            if aid != learning_agent:
                opp = create_competitive_agent(ppo_cfg.opponent_policy)
                opp.reset(aid, seed_base + i)
                opponent_agents[aid] = opp

    # Network + optimizer
    net = CompetitivePolicyNetwork(obs_dim, num_action_types).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5)

    # League registry for periodic snapshots
    from simulation.league.registry import LeagueRegistry
    league_root = str(Path(ppo_cfg.save_dir) / "competitive_league")
    registry = LeagueRegistry(league_root=league_root)

    # TensorBoard
    writer = None
    if ppo_cfg.tb_log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=ppo_cfg.tb_log_dir)

    # Progress bar
    try:
        from tqdm import tqdm
        pbar = tqdm(total=ppo_cfg.total_timesteps, desc="Competitive PPO training")
    except ImportError:
        pbar = None

    global_step = 0
    num_updates = ppo_cfg.total_timesteps // ppo_cfg.rollout_steps
    episode_returns: list[float] = []

    # Snapshot tracking
    next_snapshot_step = ppo_cfg.snapshot_every_timesteps
    last_snapshot_id: str | None = None
    snapshots_created = 0

    for update in range(1, num_updates + 1):
        # Allocate buffer for this rollout
        buf = RolloutBuffer(ppo_cfg.rollout_steps, obs_dim, device=str(device))

        # Collect rollout
        observations, _ = env.reset(seed=ppo_cfg.seed + update)
        _reset_opponents(ppo_cfg.seed + update)
        ep_reward_learning = 0.0

        for _step in range(ppo_cfg.rollout_steps):
            if not env.agents:
                # Episode ended, record and reset
                episode_returns.append(ep_reward_learning)
                observations, _ = env.reset(seed=ppo_cfg.seed + update + _step + 1000)
                _reset_opponents(ppo_cfg.seed + update + _step + 1000)
                ep_reward_learning = 0.0
                if not env.agents:
                    break

            gym_actions: dict[str, dict[str, Any]] = {}
            learning_data = None

            for agent_id in env.agents:
                if agent_id == learning_agent:
                    # PPO policy
                    obs_flat = _flatten_obs(observations[agent_id])
                    obs_t = torch.from_numpy(obs_flat).to(device)

                    with torch.no_grad():
                        at, amt, lp, _, val = net.get_action_and_value(obs_t.unsqueeze(0))

                    at_item = at.item()
                    amt_item = amt.item()

                    gym_actions[agent_id] = {
                        "action_type": at_item,
                        "amount": np.array([amt_item], dtype=np.float32),
                    }
                    learning_data = (obs_t, at_item, amt_item, lp.item(), val.item())
                else:
                    # Baseline opponent
                    if agent_id in opponent_agents:
                        action = opponent_agents[agent_id].act(observations[agent_id])
                        gym_actions[agent_id] = _action_to_gym(action)
                    else:
                        # Fallback: random action
                        gym_actions[agent_id] = {
                            "action_type": int(np.random.randint(num_action_types)),
                            "amount": np.array([np.random.uniform()], dtype=np.float32),
                        }

            next_obs, rewards, terminations, truncations, infos = env.step(gym_actions)

            # Store transition only for the learning agent
            if learning_data is not None and learning_agent in rewards:
                obs_t, at_item, amt_item, lp, val = learning_data
                r = rewards[learning_agent]
                done = terminations.get(learning_agent, False) or truncations.get(learning_agent, False)
                buf.add(obs_t, at_item, amt_item, lp, r, float(done), val)
                ep_reward_learning += r

            observations = next_obs
            global_step += 1

        if pbar:
            pbar.update(ppo_cfg.rollout_steps)

        if buf.ptr == 0:
            continue

        # Compute last value for GAE
        if env.agents and learning_agent in observations and learning_agent in env.agents:
            obs_flat = _flatten_obs(observations[learning_agent])
            obs_t = torch.from_numpy(obs_flat).to(device)
            with torch.no_grad():
                last_val = net.get_value(obs_t.unsqueeze(0)).item()
        else:
            last_val = 0.0

        buf.compute_gae(last_val, ppo_cfg.gamma, ppo_cfg.gae_lambda)

        # PPO update
        pl = vl = ent = 0.0
        for _epoch in range(ppo_cfg.ppo_epochs):
            for (mb_obs, mb_at, mb_amt, mb_old_lp, mb_adv, mb_ret) in buf.get_batches(
                ppo_cfg.num_minibatches
            ):
                _, _, new_lp, entropy, new_val = net.get_action_and_value(
                    mb_obs, mb_at, mb_amt
                )
                # Normalize advantages
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - ppo_cfg.clip_eps, 1 + ppo_cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((new_val - mb_ret) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + ppo_cfg.vf_coef * value_loss + ppo_cfg.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()

                pl = policy_loss.item()
                vl = value_loss.item()
                ent = -entropy_loss.item()

        # Logging
        if writer and episode_returns:
            writer.add_scalar("charts/mean_episode_return", np.mean(episode_returns[-10:]), global_step)
            writer.add_scalar("losses/policy_loss", pl, global_step)
            writer.add_scalar("losses/value_loss", vl, global_step)
            writer.add_scalar("losses/entropy", ent, global_step)

        # Periodic league snapshot
        if global_step >= next_snapshot_step:
            _save_artifacts(net, env_config, ppo_cfg, obs_dim, global_step)
            notes = f"checkpoint @ {global_step}"
            last_snapshot_id = registry.save_snapshot(
                source_dir=Path(ppo_cfg.save_dir) / ppo_cfg.agent_id,
                parent_id=last_snapshot_id,
                notes=notes,
            )
            snapshots_created += 1
            _trim_league_members(registry, ppo_cfg.max_league_members)
            next_snapshot_step += ppo_cfg.snapshot_every_timesteps

    if pbar:
        pbar.close()
    if writer:
        writer.close()

    # Save final artifacts
    save_path = _save_artifacts(
        net, env_config, ppo_cfg, obs_dim, global_step,
        last_league_snapshot_id=last_snapshot_id,
        snapshots_created=snapshots_created,
    )
    return save_path


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------

def _save_artifacts(
    net: CompetitivePolicyNetwork,
    env_config: CompetitiveEnvironmentConfig,
    ppo_cfg: CompetitivePPOConfig,
    obs_dim: int,
    training_steps: int,
    *,
    last_league_snapshot_id: str | None = None,
    snapshots_created: int = 0,
) -> Path:
    """Save policy.pt and metadata.json to storage/agents/{agent_id}/."""
    save_dir = Path(ppo_cfg.save_dir) / ppo_cfg.agent_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # policy.pt
    torch.save(net.state_dict(), save_dir / "policy.pt")

    # Config hash for reproducibility tracking
    config_json = env_config.model_dump_json()
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

    metadata: dict[str, Any] = {
        "algo": "competitive_ppo",
        "obs_keys": list(OBS_KEYS),
        "obs_dim": obs_dim,
        "action_mapping": {str(i): at.value for i, at in enumerate(ACTION_TYPE_ORDER)},
        "amount_distribution": "Beta",
        "config_hash": config_hash,
        "training_steps": training_steps,
        "seed": ppo_cfg.seed,
        "opponent_policy": ppo_cfg.opponent_policy,
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
        "snapshot_every_timesteps": ppo_cfg.snapshot_every_timesteps,
        "max_league_members": ppo_cfg.max_league_members,
        "last_league_snapshot_id": last_league_snapshot_id,
        "snapshots_created": snapshots_created,
    }

    (save_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return save_dir


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from simulation.config.competitive_defaults import default_competitive_config

    cfg = default_competitive_config(seed=42)
    save_path = train(cfg)
    print(f"Artifacts saved to: {save_path}")
