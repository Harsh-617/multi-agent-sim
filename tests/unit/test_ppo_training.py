"""Smoke tests for PPO shared-policy training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from simulation.config.defaults import default_config
from simulation.training.ppo_shared import (
    PPOConfig,
    SharedPolicyNetwork,
    _flat_obs_dim,
    _flatten_obs,
    train,
)


@pytest.fixture()
def tiny_ppo_config(tmp_path: Path) -> PPOConfig:
    """PPO config that trains for a tiny number of steps."""
    return PPOConfig(
        total_timesteps=128,
        rollout_steps=32,
        ppo_epochs=2,
        num_minibatches=2,
        seed=123,
        save_dir=str(tmp_path / "agents"),
        agent_id="test_ppo",
        tb_log_dir=None,
    )


class TestPPOTraining:
    """Smoke tests ensuring the training loop runs and produces artifacts."""

    def test_train_produces_artifacts(self, tiny_ppo_config: PPOConfig, tmp_path: Path):
        cfg = default_config(seed=42)
        save_path = train(cfg, tiny_ppo_config)

        assert save_path.exists()
        assert (save_path / "policy.pt").exists()
        assert (save_path / "metadata.json").exists()

    def test_metadata_has_required_fields(self, tiny_ppo_config: PPOConfig):
        cfg = default_config(seed=42)
        save_path = train(cfg, tiny_ppo_config)

        metadata = json.loads((save_path / "metadata.json").read_text(encoding="utf-8"))
        required_keys = {
            "algo", "obs_keys", "obs_dim", "action_mapping",
            "config_hash", "training_steps", "seed",
        }
        assert required_keys.issubset(metadata.keys())
        assert metadata["algo"] == "ppo_shared"
        assert metadata["obs_dim"] > 0

    def test_policy_loadable(self, tiny_ppo_config: PPOConfig):
        cfg = default_config(seed=42)
        save_path = train(cfg, tiny_ppo_config)

        metadata = json.loads((save_path / "metadata.json").read_text(encoding="utf-8"))
        obs_dim = metadata["obs_dim"]

        net = SharedPolicyNetwork(obs_dim)
        net.load_state_dict(torch.load(save_path / "policy.pt", weights_only=True))
        net.eval()

        # Forward pass with dummy input
        dummy = torch.randn(1, obs_dim)
        logits, alpha, beta, value = net(dummy)
        assert logits.shape == (1, 4)
        assert alpha.shape == (1, 1)
        assert value.shape == (1,)


class TestEvalPolicy:
    """Tests for the evaluation script."""

    def test_eval_runs(self, tiny_ppo_config: PPOConfig):
        from simulation.training.eval_policy import evaluate

        cfg = default_config(seed=42)
        save_path = train(cfg, tiny_ppo_config)

        results = evaluate(cfg, agent_dir=save_path, num_episodes=2, seed=999)
        assert "mean_return" in results
        assert "mean_final_pool" in results
        assert len(results["episode_returns"]) == 2
