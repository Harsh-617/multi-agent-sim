"""Shared-policy PPO training on MixedPettingZooParallelEnv.

Minimal CleanRL-style PPO with:
- Shared policy network for all agents
- Discrete action_type (Categorical) + continuous amount (Beta)
- Rollout collection across all agents, then PPO update for K epochs
- Deterministic seeding
- TensorBoard logging (optional)
- Artifact export to storage/agents/{agent_id}/
- League-backed self-play: train against opponents sampled from
  OpponentSampler (baseline / league / fixed policies).
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

from simulation.adapters.pettingzoo_mixed import (
    ACTION_TYPE_ORDER,
    MixedPettingZooParallelEnv,
)
from simulation.config.schema import MixedEnvironmentConfig


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """PPO training hyper-parameters."""

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
    agent_id: str = "ppo_shared"

    # League / self-play knobs
    league_training: bool = False
    opponent_mix_baseline_weight: float = 0.6
    opponent_mix_league_weight: float = 0.3
    opponent_mix_fixed_weight: float = 0.1
    recent_vs_old: float = 0.7
    include_fixed_ppo_shared: bool = True
    evaluated_agent_index: int = 0

    # Periodic league snapshots
    snapshot_every_timesteps: int = 10_000
    max_league_members: int = 50
    snapshot_notes: str | None = None


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def _flat_obs_dim(env: MixedPettingZooParallelEnv) -> int:
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
    for key in ("step", "shared_pool", "own_resources",
                "num_active_agents", "cooperation_scores", "action_history"):
        arr = obs[key]
        parts.append(arr.astype(np.float32).flatten())
    return np.concatenate(parts)


class SharedPolicyNetwork(nn.Module):
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
    """Flat buffer collecting transitions from all agents."""

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
# League training helpers
# ---------------------------------------------------------------------------

def _compute_obs_dim(env_config: MixedEnvironmentConfig) -> int:
    """Compute obs_dim from config, matching PettingZoo adapter layout."""
    n = env_config.population.num_agents
    mem = env_config.agents.observation_memory_steps
    # step(1) + shared_pool(1) + own_resources(1) + num_active(1)
    # + cooperation_scores(n-1) + action_history(mem * 5)
    return 4 + (n - 1) + (mem * 5)


def _convert_raw_obs_for_network(
    raw_obs: dict[str, Any],
    num_agents: int,
    mem_steps: int,
) -> np.ndarray:
    """Convert raw MixedEnvironment observation to flattened float32 array.

    Produces the same layout as PettingZoo ``_convert_obs`` + ``_flatten_obs``.
    """
    parts: list[np.ndarray] = []
    parts.append(np.array([raw_obs["step"]], dtype=np.float32))
    parts.append(np.array([raw_obs["shared_pool"]], dtype=np.float32))
    parts.append(np.array([raw_obs["own_resources"]], dtype=np.float32))
    parts.append(np.array([raw_obs["num_active_agents"]], dtype=np.float32))

    coop_dict: dict = raw_obs["cooperation_scores"]
    coop_arr = np.zeros(num_agents - 1, dtype=np.float32)
    for i, v in enumerate(coop_dict.values()):
        if i >= num_agents - 1:
            break
        coop_arr[i] = v
    parts.append(coop_arr)

    hist = raw_obs["action_history"]
    hist_arr = np.zeros((mem_steps, 5), dtype=np.float32)
    for i, entry in enumerate(hist):
        if i >= mem_steps:
            break
        for j, at in enumerate(ACTION_TYPE_ORDER):
            if at.value == entry["type"]:
                hist_arr[i, j] = 1.0
                break
        hist_arr[i, 4] = entry["amount"]
    parts.append(hist_arr.flatten().astype(np.float32))

    return np.concatenate(parts)


def _create_opponent_from_spec(spec: Any, registry: Any) -> Any:
    """Create a BaseAgent from an OpponentSpec."""
    from simulation.agents import create_agent

    if spec.source == "baseline":
        return create_agent(spec.policy)
    if spec.source == "league":
        member_dir = registry.load_member(spec.policy)
        return create_agent("league_snapshot", member_dir=member_dir)
    if spec.source == "fixed":
        return create_agent(spec.policy)
    raise ValueError(f"Unknown opponent source: {spec.source}")


def _trim_league_members(registry: Any, max_members: int) -> None:
    """Delete oldest league members if count exceeds *max_members*."""
    members = registry.list_members()
    if len(members) <= max_members:
        return
    # members are sorted by id (ascending) from list_members
    to_remove = members[: len(members) - max_members]
    for meta in to_remove:
        member_dir = registry.root / meta["member_id"]
        if member_dir.is_dir():
            shutil.rmtree(member_dir)


def _ppo_update(
    net: "SharedPolicyNetwork",
    optimizer: torch.optim.Optimizer,
    buf: "RolloutBuffer",
    ppo_cfg: PPOConfig,
) -> tuple[float, float, float]:
    """Run K epochs of clipped PPO on the rollout buffer.

    Returns ``(policy_loss, value_loss, entropy)`` from the last minibatch.
    """
    pl = vl = ent = 0.0
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
    return pl, vl, ent


# ---------------------------------------------------------------------------
# League-backed training loop
# ---------------------------------------------------------------------------

def _train_league(
    env_config: MixedEnvironmentConfig,
    ppo_cfg: PPOConfig,
) -> Path:
    """PPO training with opponent sampling from league / baseline / fixed."""
    from simulation.core.seeding import derive_seed
    from simulation.envs.mixed.env import MixedEnvironment
    from simulation.league.registry import LeagueRegistry
    from simulation.league.sampling import (
        OpponentSampler,
        OpponentSpec,
        SamplingWeights,
    )

    # Seeding
    torch.manual_seed(ppo_cfg.seed)
    np.random.seed(ppo_cfg.seed)
    device = torch.device(ppo_cfg.device)

    # Environment (raw MixedEnvironment — opponents need raw observations)
    env = MixedEnvironment(env_config)
    num_agents = env_config.population.num_agents
    mem_steps = env_config.agents.observation_memory_steps
    obs_dim = _compute_obs_dim(env_config)
    num_action_types = len(ACTION_TYPE_ORDER)

    # Network + optimizer
    net = SharedPolicyNetwork(obs_dim, num_action_types).to(device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5,
    )

    # League / opponent sampler
    league_root = str(Path(ppo_cfg.save_dir) / "league")
    registry = LeagueRegistry(league_root=league_root)

    sampler_weights = SamplingWeights(
        league_weight=ppo_cfg.opponent_mix_league_weight,
        baseline_weight=ppo_cfg.opponent_mix_baseline_weight,
        recent_vs_old=ppo_cfg.recent_vs_old,
    )
    include_fixed = ["ppo_shared"] if ppo_cfg.include_fixed_ppo_shared else []
    sampler = OpponentSampler(
        registry=registry,
        seed=ppo_cfg.seed,
        weights=sampler_weights,
        include_fixed=include_fixed,
    )

    # TensorBoard
    writer = None
    if ppo_cfg.tb_log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=ppo_cfg.tb_log_dir)

    # Progress bar
    try:
        from tqdm import tqdm
        pbar = tqdm(total=ppo_cfg.total_timesteps, desc="PPO league training")
    except ImportError:
        pbar = None

    global_step = 0
    num_updates = ppo_cfg.total_timesteps // ppo_cfg.rollout_steps
    episode_returns: list[float] = []
    opponent_source_counts: dict[str, int] = {
        "baseline": 0, "league": 0, "fixed": 0,
    }
    source_return_sums: dict[str, float] = {
        "baseline": 0.0, "league": 0.0, "fixed": 0.0,
    }
    source_return_counts: dict[str, int] = {
        "baseline": 0, "league": 0, "fixed": 0,
    }

    learning_agent = f"agent_{ppo_cfg.evaluated_agent_index}"
    episode_idx = 0

    # Snapshot tracking
    next_snapshot_step = ppo_cfg.snapshot_every_timesteps
    last_snapshot_id: str | None = None
    snapshots_created = 0

    # ---- helper closures --------------------------------------------------

    def _sample_and_create_opponent() -> tuple[Any, str]:
        """Sample opponent spec, create agent, return (agent, source)."""
        spec = sampler.sample_opponent_policy()
        source = spec.source
        opponent_source_counts[source] += 1
        try:
            agent = _create_opponent_from_spec(spec, registry)
        except (FileNotFoundError, KeyError):
            agent = _create_opponent_from_spec(
                OpponentSpec(source="baseline", policy="random"), registry,
            )
            # count correction
            opponent_source_counts[source] -= 1
            opponent_source_counts["baseline"] += 1
            source = "baseline"
        return agent, source

    def _reset_episode(seed: int) -> tuple[dict, list[str]]:
        observations = env.reset(seed=seed)
        agent_ids = env.active_agents()
        return observations, agent_ids

    # ---- main loop --------------------------------------------------------

    for update in range(1, num_updates + 1):
        buf = RolloutBuffer(ppo_cfg.rollout_steps, obs_dim, device=str(device))

        # Initial episode setup
        opponent_agent, current_source = _sample_and_create_opponent()
        observations, agent_ids = _reset_episode(ppo_cfg.seed + update)

        for i, aid in enumerate(agent_ids):
            if aid != learning_agent:
                opp_seed = derive_seed(ppo_cfg.seed, episode_idx * 100 + i)
                opponent_agent.reset(aid, opp_seed)
        ep_reward_learning = 0.0
        episode_idx += 1

        for _step in range(ppo_cfg.rollout_steps):
            if env.is_done():
                # Record completed episode
                episode_returns.append(ep_reward_learning)
                source_return_sums[current_source] += ep_reward_learning
                source_return_counts[current_source] += 1

                # New opponent, new episode
                opponent_agent, current_source = _sample_and_create_opponent()
                observations, agent_ids = _reset_episode(
                    ppo_cfg.seed + update + _step + 1000,
                )
                if not agent_ids:
                    break
                for i, aid in enumerate(agent_ids):
                    if aid != learning_agent:
                        opp_seed = derive_seed(
                            ppo_cfg.seed, episode_idx * 100 + i,
                        )
                        opponent_agent.reset(aid, opp_seed)
                ep_reward_learning = 0.0
                episode_idx += 1

            agent_ids = env.active_agents()
            if not agent_ids:
                break

            # Build actions for all agents
            actions: dict[str, Any] = {}
            learning_data = None

            for aid in agent_ids:
                obs = observations[aid]
                if aid == learning_agent:
                    obs_flat = _convert_raw_obs_for_network(
                        obs, num_agents, mem_steps,
                    )
                    obs_t = torch.from_numpy(obs_flat).to(device)
                    with torch.no_grad():
                        at, amt, lp, _, val = net.get_action_and_value(
                            obs_t.unsqueeze(0),
                        )
                    at_item = at.item()
                    amt_item = amt.item()
                    action_type = ACTION_TYPE_ORDER[at_item]
                    from simulation.envs.mixed.actions import Action
                    actions[aid] = Action(type=action_type, amount=amt_item)
                    learning_data = (
                        obs_t, at_item, amt_item, lp.item(), val.item(),
                    )
                else:
                    actions[aid] = opponent_agent.act(obs)

            step_results = env.step(actions)

            # Store transition ONLY for the learning agent
            if learning_data is not None and learning_agent in step_results:
                obs_t, at_item, amt_item, lp, val = learning_data
                sr = step_results[learning_agent]
                buf.add(obs_t, at_item, amt_item, lp, sr.reward, float(sr.done), val)
                ep_reward_learning += sr.reward

            # Update observations from step results
            observations = {
                aid: sr.observation for aid, sr in step_results.items()
            }
            global_step += 1

        if pbar:
            pbar.update(ppo_cfg.rollout_steps)

        if buf.ptr == 0:
            continue

        # Compute last value for GAE
        if not env.is_done() and learning_agent in observations:
            obs_flat = _convert_raw_obs_for_network(
                observations[learning_agent], num_agents, mem_steps,
            )
            obs_t = torch.from_numpy(obs_flat).to(device)
            with torch.no_grad():
                last_val = net.get_value(obs_t.unsqueeze(0)).item()
        else:
            last_val = 0.0

        buf.compute_gae(last_val, ppo_cfg.gamma, ppo_cfg.gae_lambda)
        pl, vl, ent = _ppo_update(net, optimizer, buf, ppo_cfg)

        # Logging
        if writer and episode_returns:
            writer.add_scalar(
                "charts/mean_episode_return",
                np.mean(episode_returns[-10:]),
                global_step,
            )
            writer.add_scalar("losses/policy_loss", pl, global_step)
            writer.add_scalar("losses/value_loss", vl, global_step)
            writer.add_scalar("losses/entropy", ent, global_step)

        # Periodic league snapshot
        if global_step >= next_snapshot_step:
            # Save current artifacts to the agent dir first
            _save_artifacts(
                net, env_config, ppo_cfg, obs_dim, global_step,
                opponent_source_counts=opponent_source_counts,
            )
            notes = ppo_cfg.snapshot_notes or f"checkpoint @ {global_step}"
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

    # Save artifacts
    save_path = _save_artifacts(
        net, env_config, ppo_cfg, obs_dim, global_step,
        opponent_source_counts=opponent_source_counts,
        last_league_snapshot_id=last_snapshot_id,
        snapshots_created=snapshots_created,
    )
    return save_path


# ---------------------------------------------------------------------------
# Training loop (standard / entry point)
# ---------------------------------------------------------------------------

def train(
    env_config: MixedEnvironmentConfig,
    ppo_cfg: PPOConfig | None = None,
) -> Path:
    """Run PPO training and return the path to saved artifacts."""
    if ppo_cfg is None:
        ppo_cfg = PPOConfig()

    if ppo_cfg.league_training:
        return _train_league(env_config, ppo_cfg)

    # Seeding
    torch.manual_seed(ppo_cfg.seed)
    np.random.seed(ppo_cfg.seed)

    device = torch.device(ppo_cfg.device)

    # Environment
    env = MixedPettingZooParallelEnv(env_config)
    obs_dim = _flat_obs_dim(env)
    num_action_types = len(ACTION_TYPE_ORDER)

    # Network + optimizer
    net = SharedPolicyNetwork(obs_dim, num_action_types).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5)

    # TensorBoard
    writer = None
    if ppo_cfg.tb_log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=ppo_cfg.tb_log_dir)

    # Progress bar
    try:
        from tqdm import tqdm
        pbar = tqdm(total=ppo_cfg.total_timesteps, desc="PPO training")
    except ImportError:
        pbar = None

    global_step = 0
    num_updates = ppo_cfg.total_timesteps // ppo_cfg.rollout_steps
    episode_returns: list[float] = []

    for update in range(1, num_updates + 1):
        # Allocate buffer for this rollout (worst-case: rollout_steps * num_agents)
        max_transitions = ppo_cfg.rollout_steps * len(env.possible_agents)
        buf = RolloutBuffer(max_transitions, obs_dim, device=str(device))

        # Collect rollout
        observations, _ = env.reset(seed=ppo_cfg.seed + update)
        ep_rewards: dict[str, float] = {a: 0.0 for a in env.agents}

        for _step in range(ppo_cfg.rollout_steps):
            if not env.agents:
                # Episode ended, reset
                if ep_rewards:
                    episode_returns.append(
                        sum(ep_rewards.values()) / max(len(ep_rewards), 1)
                    )
                observations, _ = env.reset(seed=ppo_cfg.seed + update + _step + 1000)
                ep_rewards = {a: 0.0 for a in env.agents}
                if not env.agents:
                    break

            gym_actions: dict[str, dict[str, Any]] = {}
            agent_data: dict[str, tuple] = {}

            for agent_id in env.agents:
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

        # Compute last value for GAE
        if env.agents and observations:
            first_agent = env.agents[0]
            obs_flat = _flatten_obs(observations[first_agent])
            obs_t = torch.from_numpy(obs_flat).to(device)
            with torch.no_grad():
                last_val = net.get_value(obs_t.unsqueeze(0)).item()
        else:
            last_val = 0.0

        buf.compute_gae(last_val, ppo_cfg.gamma, ppo_cfg.gae_lambda)

        # PPO update
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

        # Logging
        if writer and episode_returns:
            writer.add_scalar("charts/mean_episode_return", np.mean(episode_returns[-10:]), global_step)
            writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
            writer.add_scalar("losses/entropy", -entropy_loss.item(), global_step)

    if pbar:
        pbar.close()
    if writer:
        writer.close()

    # Save artifacts
    save_path = _save_artifacts(net, env_config, ppo_cfg, obs_dim, global_step)
    return save_path


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------

def _save_artifacts(
    net: SharedPolicyNetwork,
    env_config: MixedEnvironmentConfig,
    ppo_cfg: PPOConfig,
    obs_dim: int,
    training_steps: int,
    *,
    opponent_source_counts: dict[str, int] | None = None,
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
        "algo": "ppo_shared",
        "obs_keys": [
            "step", "shared_pool", "own_resources",
            "num_active_agents", "cooperation_scores", "action_history",
        ],
        "obs_dim": obs_dim,
        "action_mapping": {str(i): at.value for i, at in enumerate(ACTION_TYPE_ORDER)},
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

    if ppo_cfg.league_training:
        metadata["training_mode"] = "league_self_play"
        metadata["opponent_mix"] = {
            "baseline_weight": ppo_cfg.opponent_mix_baseline_weight,
            "league_weight": ppo_cfg.opponent_mix_league_weight,
            "fixed_weight": ppo_cfg.opponent_mix_fixed_weight,
            "recent_vs_old": ppo_cfg.recent_vs_old,
            "include_fixed_ppo_shared": ppo_cfg.include_fixed_ppo_shared,
        }
        if opponent_source_counts is not None:
            metadata["opponent_source_counts"] = opponent_source_counts
        metadata["last_league_snapshot_id"] = last_league_snapshot_id
        metadata["snapshots_created"] = snapshots_created
        metadata["snapshot_every_timesteps"] = ppo_cfg.snapshot_every_timesteps
        metadata["max_league_members"] = ppo_cfg.max_league_members

    (save_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return save_dir


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from simulation.config.defaults import default_config

    cfg = default_config(seed=42)
    save_path = train(cfg)
    print(f"Artifacts saved to: {save_path}")
