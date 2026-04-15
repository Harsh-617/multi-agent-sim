"""Integration tests for /api/transfer/* endpoints.

These tests use FastAPI's TestClient and isolate storage to tmp_path.
Fake league members with random-init torch weights are created on demand.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_member(
    league_dir: Path,
    member_id: str = "league_000001",
    obs_dim: int = 33,
    num_action_types: int = 4,
) -> Path:
    """Write a minimal policy.pt + metadata.json into league_dir/member_id/."""
    from simulation.training.ppo_shared import SharedPolicyNetwork

    member_dir = league_dir / member_id
    member_dir.mkdir(parents=True, exist_ok=True)

    net = SharedPolicyNetwork(obs_dim, num_action_types)
    torch.save(net.state_dict(), member_dir / "policy.pt")

    meta = {
        "obs_dim": obs_dim,
        "num_action_types": num_action_types,
        "member_id": member_id,
    }
    (member_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return member_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_transfer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect routes_transfer storage references and reset manager state."""
    import backend.api.routes_transfer as rt
    import simulation.transfer.transfer_runner as tr

    # Set up all directories first
    reports_dir = tmp_path / "reports"
    mixed_league = tmp_path / "agents/league"
    comp_league = tmp_path / "agents/competitive_league"
    coop_league = tmp_path / "agents/cooperative/league"
    configs_dir = tmp_path / "configs"

    for d in (reports_dir, mixed_league, comp_league, coop_league, configs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Patch routes module
    monkeypatch.setattr(rt, "REPORTS_ROOT", reports_dir)
    monkeypatch.setattr(rt, "CONFIGS_DIR", configs_dir)

    # Patch runner module
    monkeypatch.setattr(tr, "_MIXED_LEAGUE_ROOT", mixed_league)
    monkeypatch.setattr(tr, "_COMPETITIVE_LEAGUE_ROOT", comp_league)
    monkeypatch.setattr(tr, "_COOPERATIVE_LEAGUE_ROOT", coop_league)
    monkeypatch.setattr(tr, "_CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(tr, "_REPORTS_ROOT", reports_dir)

    # Reset transfer manager state before each test
    rt.transfer_manager.reset_state()
    yield {
        "reports_dir": reports_dir,
        "mixed_league": mixed_league,
        "comp_league": comp_league,
        "coop_league": coop_league,
        "configs_dir": configs_dir,
        "tmp_path": tmp_path,
    }
    rt.transfer_manager.reset_state()


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/transfer/run
# ---------------------------------------------------------------------------

class TestTransferRunEndpoint:
    def test_returns_200_with_transfer_id(self, client, _isolate_transfer):
        dirs = _isolate_transfer
        _make_fake_member(dirs["mixed_league"])

        resp = client.post("/api/transfer/run", json={
            "source_archetype": "mixed",
            "source_member_id": "league_000001",
            "target_archetype": "cooperative",
            "target_config_id": "default",
            "episodes": 1,
            "seed": 42,
        })

        assert resp.status_code == 200
        body = resp.json()
        assert "transfer_id" in body
        assert body["status"] == "pending"
        assert len(body["transfer_id"]) == 12

    def test_returns_422_when_source_equals_target_archetype(self, client, _isolate_transfer):
        resp = client.post("/api/transfer/run", json={
            "source_archetype": "mixed",
            "source_member_id": "league_000001",
            "target_archetype": "mixed",
            "target_config_id": "default",
            "episodes": 1,
            "seed": 42,
        })
        assert resp.status_code == 422
        body = resp.json()
        # FastAPI validation errors nest differently; check detail contains the message
        detail = body.get("detail", "")
        assert "differ" in detail or "same" in detail.lower() or "mixed" in detail

    def test_returns_422_when_config_archetype_mismatches_target(self, client, _isolate_transfer):
        dirs = _isolate_transfer

        # Write a mixed config but claim target is cooperative
        mixed_config = {
            "identity": {
                "environment_type": "mixed",
                "environment_version": "0.1.0",
                "seed": 42,
            },
            "population": {
                "num_agents": 4,
                "max_steps": 50,
                "initial_shared_pool": 100.0,
                "initial_agent_resources": 10.0,
                "collapse_threshold": 0.0,
            },
            "layers": {
                "information_asymmetry": 0.0,
                "temporal_memory_depth": 5,
                "reputation_sensitivity": 0.0,
                "incentive_softness": 1.0,
                "uncertainty_intensity": 0.0,
            },
            "rewards": {
                "individual_weight": 1.0,
                "group_weight": 0.0,
                "relational_weight": 0.0,
                "penalty_scaling": 1.0,
            },
            "agents": {"observation_memory_steps": 5},
        }
        config_path = dirs["configs_dir"] / "wrong_type.json"
        config_path.write_text(json.dumps(mixed_config), encoding="utf-8")

        resp = client.post("/api/transfer/run", json={
            "source_archetype": "mixed",
            "source_member_id": "league_000001",
            "target_archetype": "cooperative",
            "target_config_id": "wrong_type",
            "episodes": 1,
            "seed": 42,
        })
        assert resp.status_code == 422
        body = resp.json()
        detail = body.get("detail", "")
        assert "environment_type" in detail or "mismatch" in detail.lower() or "cooperative" in detail

    def test_returns_422_for_invalid_source_archetype(self, client, _isolate_transfer):
        resp = client.post("/api/transfer/run", json={
            "source_archetype": "invalid_type",
            "source_member_id": "league_000001",
            "target_archetype": "cooperative",
            "target_config_id": "default",
            "episodes": 1,
            "seed": 42,
        })
        assert resp.status_code == 422

    def test_returns_422_for_invalid_target_archetype(self, client, _isolate_transfer):
        resp = client.post("/api/transfer/run", json={
            "source_archetype": "mixed",
            "source_member_id": "league_000001",
            "target_archetype": "bad_archetype",
            "target_config_id": "default",
            "episodes": 1,
            "seed": 42,
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/transfer/status/{transfer_id}
# ---------------------------------------------------------------------------

class TestTransferStatusEndpoint:
    def test_returns_valid_status_after_submission(self, client, _isolate_transfer):
        dirs = _isolate_transfer
        _make_fake_member(dirs["mixed_league"])

        # Start run
        post_resp = client.post("/api/transfer/run", json={
            "source_archetype": "mixed",
            "source_member_id": "league_000001",
            "target_archetype": "cooperative",
            "target_config_id": "default",
            "episodes": 1,
            "seed": 42,
        })
        assert post_resp.status_code == 200
        transfer_id = post_resp.json()["transfer_id"]

        # Poll status — with TestClient the background task runs synchronously
        # so it may already be done
        status_resp = client.get(f"/api/transfer/status/{transfer_id}")
        assert status_resp.status_code == 200
        body = status_resp.json()
        assert body["transfer_id"] == transfer_id
        assert body["status"] in (
            "pending", "running_transfer", "running_baseline", "saving", "done", "error"
        )

    def test_returns_404_for_unknown_transfer_id(self, client, _isolate_transfer):
        resp = client.get("/api/transfer/status/nonexistent_id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/transfer/reports
# ---------------------------------------------------------------------------

class TestTransferReportsListEndpoint:
    def test_returns_empty_list_when_no_reports(self, client, _isolate_transfer):
        resp = client.get("/api/transfer/reports")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_list_with_correct_schema(self, client, _isolate_transfer):
        dirs = _isolate_transfer
        # Write a fake report
        report_dir = dirs["reports_dir"] / "transfer_mixed_cooperative_abcd1234_20260101_120000"
        report_dir.mkdir()
        summary = {
            "report_type": "transfer",
            "report_id": report_dir.name,
            "source_archetype": "mixed",
            "source_member_id": "league_000001",
            "target_archetype": "cooperative",
            "target_config_hash": "abcd1234",
            "episodes": 2,
            "transferred_mean": 0.5,
            "baseline_mean": 0.3,
            "vs_baseline_delta": 0.2,
            "vs_baseline_pct": 66.67,
        }
        (report_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        resp = client.get("/api/transfer/reports")
        assert resp.status_code == 200
        reports = resp.json()
        assert len(reports) == 1
        r = reports[0]
        assert r["report_id"] == report_dir.name
        assert r["source_archetype"] == "mixed"
        assert r["target_archetype"] == "cooperative"

    def test_ignores_non_transfer_dirs(self, client, _isolate_transfer):
        dirs = _isolate_transfer
        # Write a robustness report (should be ignored)
        robust_dir = dirs["reports_dir"] / "robust_abc123"
        robust_dir.mkdir()
        (robust_dir / "report.json").write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

        resp = client.get("/api/transfer/reports")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /api/transfer/reports/{report_id}
# ---------------------------------------------------------------------------

class TestTransferReportDetailEndpoint:
    def test_returns_correct_schema_after_completion(self, client, _isolate_transfer):
        dirs = _isolate_transfer
        report_id = "transfer_cooperative_mixed_deadbeef_20260115_100000"
        report_dir = dirs["reports_dir"] / report_id
        report_dir.mkdir()
        summary = {
            "report_type": "transfer",
            "report_id": report_id,
            "source_archetype": "cooperative",
            "source_member_id": "league_000001",
            "source_obs_dim": 35,
            "source_strategy_label": None,
            "source_elo": None,
            "target_archetype": "mixed",
            "target_config_hash": "deadbeef",
            "target_obs_dim": 33,
            "obs_mismatch_strategy": "pad",
            "episodes": 3,
            "seed": 7,
            "transferred_results": [
                {"cooperation_rate": 0.4},
                {"cooperation_rate": 0.5},
                {"cooperation_rate": 0.3},
            ],
            "baseline_results": [
                {"cooperation_rate": 0.25},
                {"cooperation_rate": 0.25},
                {"cooperation_rate": 0.25},
            ],
            "transferred_mean": 0.4,
            "baseline_mean": 0.25,
            "vs_baseline_delta": 0.15,
            "vs_baseline_pct": 60.0,
        }
        (report_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        resp = client.get(f"/api/transfer/reports/{report_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_id"] == report_id
        assert data["report_type"] == "transfer"
        assert data["transferred_mean"] == 0.4
        assert data["baseline_mean"] == 0.25
        assert len(data["transferred_results"]) == 3
        assert len(data["baseline_results"]) == 3

    def test_returns_404_for_unknown_report_id(self, client, _isolate_transfer):
        resp = client.get("/api/transfer/reports/nonexistent_report_xyz")
        assert resp.status_code == 404

    def test_returns_404_when_summary_missing(self, client, _isolate_transfer):
        dirs = _isolate_transfer
        # Dir exists but no summary.json
        orphan = dirs["reports_dir"] / "transfer_mixed_coop_00000000_20260101_000000"
        orphan.mkdir()

        resp = client.get(f"/api/transfer/reports/{orphan.name}")
        assert resp.status_code == 404
