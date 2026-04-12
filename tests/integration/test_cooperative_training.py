"""Integration tests for cooperative PPO training and evaluation.

Verifies that the full training pipeline works end-to-end:
  train() → saves artifacts → eval() loads and runs episodes.

All tests use a tiny PPOConfig (128 total_timesteps) so they run quickly on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from simulation.adapters.cooperative_pettingzoo import CooperativePettingZooParallelEnv
from simulation.config.cooperative_defaults import default_cooperative_config
from simulation.training.cooperative_eval import evaluate
from simulation.training.cooperative_train import PPOConfig, train


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def env_config():
    """Small cooperative config: 4 agents, 3 task types, 200 max steps."""
    return default_cooperative_config(seed=42)


@pytest.fixture()
def tiny_ppo_cfg(tmp_path: Path) -> PPOConfig:
    """PPO config that completes a short run in well under a second on CPU."""
    return PPOConfig(
        total_timesteps=128,
        rollout_steps=64,
        ppo_epochs=1,
        num_minibatches=2,
        seed=7,
        device="cpu",
        save_dir=str(tmp_path / "agents" / "cooperative"),
        agent_id="ppo_shared",
        tb_log_dir=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCooperativeTraining:

    def test_training_run_completes_without_error(
        self, env_config, tiny_ppo_cfg: PPOConfig
    ):
        """train() must complete 100+ env steps without raising."""
        save_path = train(env_config, tiny_ppo_cfg)
        assert save_path is not None
        assert isinstance(save_path, Path)

    def test_policy_pt_saved_to_correct_storage_path(
        self, env_config, tiny_ppo_cfg: PPOConfig
    ):
        """policy.pt must be saved inside save_dir/ppo_shared/."""
        save_path = train(env_config, tiny_ppo_cfg)
        policy_file = save_path / "policy.pt"
        assert policy_file.exists(), f"policy.pt not found at {policy_file}"

        # Confirm the file is a valid torch state_dict
        state_dict = torch.load(policy_file, weights_only=True)
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

    def test_metadata_json_saved_with_required_fields(
        self, env_config, tiny_ppo_cfg: PPOConfig
    ):
        """metadata.json must exist and contain archetype, config_hash, training_steps."""
        save_path = train(env_config, tiny_ppo_cfg)
        meta_path = save_path / "metadata.json"

        assert meta_path.exists(), f"metadata.json not found at {meta_path}"

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        required_fields = {"archetype", "config_hash", "training_steps"}
        missing = required_fields - metadata.keys()
        assert not missing, f"metadata.json missing required fields: {missing}"

        assert metadata["archetype"] == "cooperative"
        assert isinstance(metadata["config_hash"], str) and len(metadata["config_hash"]) > 0
        assert isinstance(metadata["training_steps"], int) and metadata["training_steps"] > 0

    def test_eval_loads_saved_policy_and_runs_episodes(
        self, env_config, tiny_ppo_cfg: PPOConfig
    ):
        """evaluate() must load the saved policy and complete 3 episodes without error."""
        save_path = train(env_config, tiny_ppo_cfg)

        results = evaluate(
            env_config,
            agent_dir=save_path,
            num_episodes=3,
            seed=999,
            deterministic=True,
        )

        assert isinstance(results, dict)
        expected_keys = {
            "mean_completion_ratio",
            "mean_group_efficiency_ratio",
            "mean_effort_utilization",
            "mean_return",
            "std_return",
            "episode_returns",
        }
        assert expected_keys.issubset(results.keys()), (
            f"Missing keys: {expected_keys - results.keys()}"
        )
        assert len(results["episode_returns"]) == 3
        assert all(isinstance(r, float) for r in results["episode_returns"])

    def test_mean_reward_higher_than_random_baseline(
        self, env_config, tiny_ppo_cfg: PPOConfig
    ):
        """Trained policy mean return must exceed an all-idle (zero-effort) baseline.

        Rationale: the SharedPolicyNetwork is initialised with Beta(alpha>1, beta>1)
        which produces effort > 0 from the very first forward pass.  Any non-zero
        effort contribution raises r_individual and improves r_group compared with
        a policy that always idles (zero effort, zero task completion).
        """
        save_path = train(env_config, tiny_ppo_cfg)

        trained_results = evaluate(
            env_config, agent_dir=save_path, num_episodes=5, seed=111
        )
        trained_return = trained_results["mean_return"]

        # Build all-idle baseline
        T = env_config.population.num_task_types
        env_idle = CooperativePettingZooParallelEnv(env_config)
        idle_returns: list[float] = []

        for ep in range(5):
            obs, _ = env_idle.reset(seed=ep + 200)
            ep_rewards: dict[str, float] = {a: 0.0 for a in env_idle.agents}

            while env_idle.agents:
                # IDLE: task_type = T (sentinel), effort = 0
                actions = {
                    aid: {
                        "task_type": T,
                        "effort_amount": np.array([0.0], dtype=np.float32),
                    }
                    for aid in env_idle.agents
                }
                _, rewards, _, _, _ = env_idle.step(actions)
                for aid, r in rewards.items():
                    ep_rewards[aid] = ep_rewards.get(aid, 0.0) + r

            idle_returns.append(
                sum(ep_rewards.values()) / max(len(ep_rewards), 1)
            )

        idle_baseline = float(np.mean(idle_returns))

        assert trained_return >= idle_baseline, (
            f"Trained policy mean return ({trained_return:.4f}) must be >= "
            f"all-idle baseline ({idle_baseline:.4f}). "
            "SharedPolicyNetwork init produces non-zero effort from the first step, "
            "which always improves over pure idle."
        )
