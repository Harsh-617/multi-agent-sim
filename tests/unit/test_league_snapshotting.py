"""Tests for periodic league snapshotting during PPO training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.config.defaults import default_config
from simulation.league.registry import LeagueRegistry
from simulation.training.ppo_shared import PPOConfig, train


def _league_ppo_cfg(
    tmp_path: Path,
    *,
    total_timesteps: int = 1500,
    rollout_steps: int = 256,
    snapshot_every: int = 500,
    max_league_members: int = 50,
    agent_id: str = "snap_test",
) -> PPOConfig:
    return PPOConfig(
        total_timesteps=total_timesteps,
        rollout_steps=rollout_steps,
        ppo_epochs=2,
        num_minibatches=2,
        seed=42,
        save_dir=str(tmp_path / "agents"),
        agent_id=agent_id,
        tb_log_dir=None,
        league_training=True,
        opponent_mix_baseline_weight=1.0,
        opponent_mix_league_weight=0.0,
        opponent_mix_fixed_weight=0.0,
        include_fixed_ppo_shared=False,
        snapshot_every_timesteps=snapshot_every,
        max_league_members=max_league_members,
    )


class TestPeriodicLeagueSnapshots:
    """Verify that league snapshots are created periodically during training."""

    def test_snapshots_created(self, tmp_path: Path):
        """Training with snapshot_every=500 and total=1500 creates >=2 members."""
        cfg = default_config(seed=42)
        ppo_cfg = _league_ppo_cfg(tmp_path, total_timesteps=1500, rollout_steps=256, snapshot_every=500)
        save_path = train(cfg, ppo_cfg)

        league_root = tmp_path / "agents" / "league"
        registry = LeagueRegistry(league_root)
        members = registry.list_members()

        assert len(members) >= 2, (
            f"Expected >=2 league snapshots but got {len(members)}"
        )

    def test_metadata_has_snapshot_fields(self, tmp_path: Path):
        """Final metadata.json contains snapshot tracking fields."""
        cfg = default_config(seed=42)
        ppo_cfg = _league_ppo_cfg(tmp_path, total_timesteps=1500, rollout_steps=256, snapshot_every=500)
        save_path = train(cfg, ppo_cfg)

        metadata = json.loads(
            (save_path / "metadata.json").read_text(encoding="utf-8")
        )

        assert "snapshots_created" in metadata
        assert metadata["snapshots_created"] >= 2
        assert "last_league_snapshot_id" in metadata
        assert metadata["last_league_snapshot_id"] is not None
        assert "snapshot_every_timesteps" in metadata
        assert metadata["snapshot_every_timesteps"] == 500
        assert "max_league_members" in metadata

    def test_snapshot_parent_chain(self, tmp_path: Path):
        """Each snapshot after the first should reference the previous as parent."""
        cfg = default_config(seed=42)
        ppo_cfg = _league_ppo_cfg(tmp_path, total_timesteps=1500, rollout_steps=256, snapshot_every=500)
        train(cfg, ppo_cfg)

        league_root = tmp_path / "agents" / "league"
        registry = LeagueRegistry(league_root)
        members = registry.list_members()

        if len(members) >= 2:
            # First member has no parent
            assert members[0].get("parent_id") is None
            # Second member references first
            assert members[1]["parent_id"] == members[0]["member_id"]

    def test_trimming_max_league_members(self, tmp_path: Path):
        """With max_league_members=2, only 2 members remain even if 3+ are created."""
        cfg = default_config(seed=42)
        # Use small rollout to ensure many snapshots
        ppo_cfg = _league_ppo_cfg(
            tmp_path,
            total_timesteps=2048,
            rollout_steps=128,
            snapshot_every=128,
            max_league_members=2,
            agent_id="trim_test",
        )
        save_path = train(cfg, ppo_cfg)

        league_root = tmp_path / "agents" / "league"
        registry = LeagueRegistry(league_root)
        members = registry.list_members()

        metadata = json.loads(
            (save_path / "metadata.json").read_text(encoding="utf-8")
        )

        # More than 2 snapshots were created over the course of training
        assert metadata["snapshots_created"] >= 3, (
            f"Expected >=3 snapshots created, got {metadata['snapshots_created']}"
        )
        # But only 2 remain on disk
        assert len(members) <= 2, (
            f"Expected <=2 league members after trimming, got {len(members)}"
        )

    def test_standard_training_no_snapshots(self, tmp_path: Path):
        """Non-league training does not create any league snapshots."""
        cfg = default_config(seed=42)
        ppo_cfg = PPOConfig(
            total_timesteps=256,
            rollout_steps=64,
            ppo_epochs=2,
            num_minibatches=2,
            seed=42,
            save_dir=str(tmp_path / "agents"),
            agent_id="no_snap_test",
            tb_log_dir=None,
            league_training=False,
            snapshot_every_timesteps=64,
        )
        train(cfg, ppo_cfg)

        league_root = tmp_path / "agents" / "league"
        # League dir may or may not exist; if it does, no members
        if league_root.exists():
            registry = LeagueRegistry(league_root)
            assert len(registry.list_members()) == 0
