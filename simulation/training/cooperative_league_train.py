"""League self-play training for the Cooperative archetype.

Mirrors ppo_shared.py's league training structure — adapted for:
  - CooperativePettingZooParallelEnv (flat Box obs, Dict action)
  - CooperativeEnvironmentConfig
  - Periodic snapshots saved to CooperativeLeagueRegistry
  - Opponent sampling with recent_vs_old bias (same as Mixed)
  - Artifact storage under storage/agents/cooperative/league/

Does NOT modify ppo_shared.py (ADR-003, ADR-013).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from simulation.adapters.cooperative_pettingzoo import CooperativePettingZooParallelEnv
from simulation.config.cooperative_schema import CooperativeEnvironmentConfig
from simulation.league.cooperative_registry import CooperativeLeagueRegistry
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
# Snapshot helpers
# ---------------------------------------------------------------------------

def _save_snapshot_metadata(
    net: SharedPolicyNetwork,
    env_config: CooperativeEnvironmentConfig,
    ppo_cfg: PPOConfig,
    obs_dim: int,
    training_steps: int,
    num_action_types: int,
    snap_dir: Path,
) -> None:
    """Write policy.pt + metadata.json to snap_dir."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), snap_dir / "policy.pt")

    config_json = env_config.model_dump_json()
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]
    T = env_config.population.num_task_types
    action_mapping = {str(i): f"task_{i}" for i in range(T)}
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
    (snap_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Opponent sampling (recent_vs_old bias, mirrors OpponentSampler pattern)
# ---------------------------------------------------------------------------

def _sample_snapshot_net(
    registry: CooperativeLeagueRegistry,
    obs_dim: int,
    num_action_types: int,
    recent_vs_old: float = 0.7,
    rng: np.random.RandomState | None = None,
) -> SharedPolicyNetwork | None:
    """Load a snapshot policy with recent-vs-old weighting.

    Returns None if no snapshots are available yet.
    """
    members = registry.list_members()
    if not members:
        return None

    if rng is None:
        rng = np.random.RandomState(0)

    n = len(members)
    # Recent bias: later members get higher weight
    if n == 1:
        chosen = members[0]
    else:
        # Weight recent members more heavily (recent_vs_old fraction to last half)
        weights = np.ones(n, dtype=np.float64)
        half = n // 2
        weights[half:] *= (recent_vs_old / max(1 - recent_vs_old, 1e-9))
        weights /= weights.sum()
        idx = rng.choice(n, p=weights)
        chosen = members[idx]

    member_dir = registry.load_member(chosen["member_id"])
    snap_net = SharedPolicyNetwork(obs_dim, num_action_types)
    snap_net.load_state_dict(
        torch.load(member_dir / "policy.pt", weights_only=True)
    )
    snap_net.eval()
    return snap_net


# ---------------------------------------------------------------------------
# League training loop
# ---------------------------------------------------------------------------

def train_cooperative_league(
    env_config: CooperativeEnvironmentConfig,
    ppo_cfg: PPOConfig | None = None,
    league_root: str | Path | None = None,
) -> Path:
    """Run league PPO training for cooperative archetype and return artifact path.

    Periodic snapshots are saved to the cooperative league registry.
    Opponent sampling uses recent-vs-old bias (same as Mixed).

    Parameters
    ----------
    env_config:
        Cooperative environment configuration.
    ppo_cfg:
        PPO hyperparameters. Defaults to league-mode config.
    league_root:
        Path to the cooperative league root directory.
        Defaults to storage/agents/cooperative/league/.
    """
    if ppo_cfg is None:
        ppo_cfg = PPOConfig(
            save_dir="storage/agents/cooperative",
            agent_id="ppo_shared",
            league_training=True,
            snapshot_every_timesteps=10_000,
            max_league_members=50,
        )

    if league_root is None:
        league_root = Path(ppo_cfg.save_dir) / "league"

    registry = CooperativeLeagueRegistry(league_root=league_root)

    torch.manual_seed(ppo_cfg.seed)
    np.random.seed(ppo_cfg.seed)
    rng = np.random.RandomState(ppo_cfg.seed)
    device = torch.device(ppo_cfg.device)

    # --- Environment & spaces ---
    env = CooperativePettingZooParallelEnv(env_config)
    obs_dim = _flat_obs_dim_coop(env)
    num_action_types = _num_action_types_coop(env)

    # --- Learner network + optimizer ---
    net = SharedPolicyNetwork(obs_dim, num_action_types).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5)

    # Progress bar (optional)
    try:
        from tqdm import tqdm
        pbar = tqdm(total=ppo_cfg.total_timesteps, desc="Coop League PPO")
    except ImportError:
        pbar = None

    global_step = 0
    num_updates = ppo_cfg.total_timesteps // ppo_cfg.rollout_steps
    next_snapshot_at = ppo_cfg.snapshot_every_timesteps
    snapshots_created = 0
    last_member_id: str | None = None
    episode_returns: list[float] = []

    # Temp dir for staging snapshots before registry registration
    snap_stage = Path(ppo_cfg.save_dir) / "_coop_snap_stage"

    # ---- main loop ---------------------------------------------------------

    for update in range(1, num_updates + 1):
        max_transitions = ppo_cfg.rollout_steps * len(env.possible_agents)
        buf = RolloutBuffer(max_transitions, obs_dim, device=str(device))

        observations, _ = env.reset(seed=ppo_cfg.seed + update)
        ep_rewards: dict[str, float] = {a: 0.0 for a in env.agents}

        for _step in range(ppo_cfg.rollout_steps):
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

            gym_actions: dict[str, dict[str, Any]] = {}
            agent_data: dict[str, tuple] = {}

            for agent_id in env.agents:
                obs_arr = observations[agent_id]
                obs_t = torch.from_numpy(obs_arr).to(device)

                with torch.no_grad():
                    at, amt, lp, _, val = net.get_action_and_value(obs_t.unsqueeze(0))

                gym_actions[agent_id] = {
                    "task_type": at.item(),
                    "effort_amount": np.array([amt.item()], dtype=np.float32),
                }
                agent_data[agent_id] = (obs_t, at.item(), amt.item(), lp.item(), val.item())

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

        # Bootstrap value
        if env.agents and observations:
            first_agent = env.agents[0]
            obs_arr = observations[first_agent]
            obs_t = torch.from_numpy(obs_arr).to(device)
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

        # Periodic snapshot
        if ppo_cfg.league_training and global_step >= next_snapshot_at:
            _save_snapshot_metadata(
                net, env_config, ppo_cfg, obs_dim, global_step,
                num_action_types, snap_stage,
            )
            member_id = registry.save_snapshot(
                snap_stage,
                parent_id=last_member_id,
                notes=f"step_{global_step}",
            )
            last_member_id = member_id
            snapshots_created += 1

            # Trim oldest if over limit
            members = registry.list_members()
            while len(members) > ppo_cfg.max_league_members:
                oldest = members.pop(0)
                oldest_dir = registry.root / oldest["member_id"]
                if oldest_dir.is_dir():
                    shutil.rmtree(oldest_dir, ignore_errors=True)
                members = registry.list_members()

            next_snapshot_at += ppo_cfg.snapshot_every_timesteps

    if pbar:
        pbar.close()

    # Clean up staging dir
    if snap_stage.exists():
        shutil.rmtree(snap_stage, ignore_errors=True)

    # Save final artifacts
    save_dir = Path(ppo_cfg.save_dir) / ppo_cfg.agent_id
    _save_snapshot_metadata(
        net, env_config, ppo_cfg, obs_dim, global_step,
        num_action_types, save_dir,
    )

    return save_dir


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from simulation.config.cooperative_defaults import default_cooperative_config

    cfg = default_cooperative_config(seed=42)
    ppo_cfg = PPOConfig(
        save_dir="storage/agents/cooperative",
        agent_id="ppo_shared",
        league_training=True,
        snapshot_every_timesteps=10_000,
        total_timesteps=50_000,
    )
    save_path = train_cooperative_league(cfg, ppo_cfg)
    print(f"Artifacts saved to: {save_path}")
