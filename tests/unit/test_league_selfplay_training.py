"""Tests for league-backed self-play PPO training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.config.defaults import default_config
from simulation.training.ppo_shared import PPOConfig, train


@pytest.fixture()
def league_ppo_config(tmp_path: Path) -> PPOConfig:
    """PPO config for league self-play with minimal timesteps."""
    return PPOConfig(
        total_timesteps=256,
        rollout_steps=64,
        ppo_epochs=2,
        num_minibatches=2,
        seed=123,
        save_dir=str(tmp_path / "agents"),
        agent_id="test_league_ppo",
        tb_log_dir=None,
        league_training=True,
        # Baseline-only mix (league dir is empty, fixed may not exist)
        opponent_mix_baseline_weight=1.0,
        opponent_mix_league_weight=0.0,
        opponent_mix_fixed_weight=0.0,
        include_fixed_ppo_shared=False,
    )


class TestLeagueSelfPlayTraining:
    """Smoke tests for league-backed PPO training."""

    def test_smoke_train_completes(self, league_ppo_config: PPOConfig):
        cfg = default_config(seed=42)
        save_path = train(cfg, league_ppo_config)

        assert save_path.exists()
        assert (save_path / "policy.pt").exists()
        assert (save_path / "metadata.json").exists()

    def test_metadata_has_league_keys(self, league_ppo_config: PPOConfig):
        cfg = default_config(seed=42)
        save_path = train(cfg, league_ppo_config)

        metadata = json.loads(
            (save_path / "metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["training_mode"] == "league_self_play"
        assert "opponent_mix" in metadata
        assert "opponent_source_counts" in metadata

        mix = metadata["opponent_mix"]
        assert mix["baseline_weight"] == 1.0
        assert mix["league_weight"] == 0.0

    def test_baseline_only_empty_league(self, league_ppo_config: PPOConfig):
        """Training with baseline-only weights works even with empty league."""
        cfg = default_config(seed=42)
        save_path = train(cfg, league_ppo_config)

        metadata = json.loads(
            (save_path / "metadata.json").read_text(encoding="utf-8")
        )
        counts = metadata["opponent_source_counts"]
        # All opponents should be baseline (league and fixed are 0 weight)
        assert counts["baseline"] > 0
        assert counts["league"] == 0
        assert counts["fixed"] == 0

    def test_determinism_same_seed(self, tmp_path: Path):
        """Two runs with the same seed produce identical opponent_source_counts."""
        cfg = default_config(seed=42)

        results = []
        for run_idx in range(2):
            ppo_cfg = PPOConfig(
                total_timesteps=256,
                rollout_steps=64,
                ppo_epochs=2,
                num_minibatches=2,
                seed=999,
                save_dir=str(tmp_path / f"agents_run{run_idx}"),
                agent_id="det_test",
                tb_log_dir=None,
                league_training=True,
                opponent_mix_baseline_weight=1.0,
                opponent_mix_league_weight=0.0,
                opponent_mix_fixed_weight=0.0,
                include_fixed_ppo_shared=False,
            )
            save_path = train(cfg, ppo_cfg)
            metadata = json.loads(
                (save_path / "metadata.json").read_text(encoding="utf-8")
            )
            results.append(metadata["opponent_source_counts"])

        assert results[0] == results[1]

    def test_standard_training_unaffected(self, tmp_path: Path):
        """Standard (non-league) training still works and has no league keys."""
        cfg = default_config(seed=42)
        ppo_cfg = PPOConfig(
            total_timesteps=128,
            rollout_steps=32,
            ppo_epochs=2,
            num_minibatches=2,
            seed=42,
            save_dir=str(tmp_path / "agents"),
            agent_id="standard_test",
            tb_log_dir=None,
            league_training=False,
        )
        save_path = train(cfg, ppo_cfg)

        metadata = json.loads(
            (save_path / "metadata.json").read_text(encoding="utf-8")
        )
        assert "training_mode" not in metadata
        assert "opponent_mix" not in metadata
