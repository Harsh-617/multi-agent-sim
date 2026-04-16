"""Unit tests for simulation.export.policy_exporter."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
import torch

from simulation.export.policy_exporter import (
    build_export_zip,
    generate_policy_py,
    generate_readme,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_metadata(
    archetype: str,
    obs_dim: int | None = None,
    num_action_types: int | None = None,
) -> dict:
    defaults = {
        "mixed":       {"obs_dim": 33, "num_action_types": 4},
        "competitive": {"obs_dim": 93, "num_action_types": 4},
        "cooperative": {"obs_dim": 40, "num_action_types": 4},
    }
    d = defaults[archetype].copy()
    if obs_dim is not None:
        d["obs_dim"] = obs_dim
    if num_action_types is not None:
        d["num_action_types"] = num_action_types
    d.update({
        "member_id": "league_000001",
        "config_hash": "abc12345",
        "training_steps": 50000,
        "seed": 42,
        "rating": 1234.5,
    })
    return d


def _write_fake_policy_pt(member_dir: Path, obs_dim: int, num_action_types: int) -> None:
    """Write a randomly-initialised policy.pt so build_export_zip can pack it."""
    from simulation.training.ppo_shared import SharedPolicyNetwork

    net = SharedPolicyNetwork(obs_dim, num_action_types)
    torch.save(net.state_dict(), member_dir / "policy.pt")


# ---------------------------------------------------------------------------
# generate_policy_py — valid Python string
# ---------------------------------------------------------------------------

class TestGeneratePolicyPy:
    def test_mixed_is_valid_python_string(self):
        meta = _make_metadata("mixed")
        code = generate_policy_py(meta, "mixed")
        assert isinstance(code, str)
        assert len(code) > 100

    def test_competitive_is_valid_python_string(self):
        meta = _make_metadata("competitive")
        code = generate_policy_py(meta, "competitive")
        assert isinstance(code, str)
        assert len(code) > 100

    def test_cooperative_is_valid_python_string(self):
        meta = _make_metadata("cooperative")
        code = generate_policy_py(meta, "cooperative")
        assert isinstance(code, str)
        assert len(code) > 100

    def test_contains_policy_network_class(self):
        meta = _make_metadata("mixed")
        code = generate_policy_py(meta, "mixed")
        assert "class PolicyNetwork" in code

    def test_contains_predict_function(self):
        meta = _make_metadata("mixed")
        code = generate_policy_py(meta, "mixed")
        assert "def predict(" in code

    def test_mixed_obs_dim_embedded(self):
        meta = _make_metadata("mixed", obs_dim=33)
        code = generate_policy_py(meta, "mixed")
        assert "obs_dim=33" in code

    def test_competitive_obs_dim_embedded(self):
        meta = _make_metadata("competitive", obs_dim=93)
        code = generate_policy_py(meta, "competitive")
        assert "obs_dim=93" in code

    def test_cooperative_obs_dim_embedded(self):
        meta = _make_metadata("cooperative", obs_dim=40)
        code = generate_policy_py(meta, "cooperative")
        assert "obs_dim=40" in code

    def test_mixed_num_action_types_embedded(self):
        meta = _make_metadata("mixed", num_action_types=4)
        code = generate_policy_py(meta, "mixed")
        assert "num_action_types=4" in code

    def test_competitive_num_action_types_embedded(self):
        meta = _make_metadata("competitive", num_action_types=4)
        code = generate_policy_py(meta, "competitive")
        assert "num_action_types=4" in code

    def test_cooperative_num_action_types_embedded(self):
        meta = _make_metadata("cooperative", num_action_types=4)
        code = generate_policy_py(meta, "cooperative")
        assert "num_action_types=4" in code

    def test_action_mapping_mixed_in_docstring(self):
        meta = _make_metadata("mixed")
        code = generate_policy_py(meta, "mixed")
        assert "cooperate" in code
        assert "extract" in code

    def test_action_mapping_competitive_in_docstring(self):
        meta = _make_metadata("competitive")
        code = generate_policy_py(meta, "competitive")
        assert "build" in code
        assert "attack" in code

    def test_action_mapping_cooperative_in_docstring(self):
        meta = _make_metadata("cooperative")
        code = generate_policy_py(meta, "cooperative")
        assert "idle" in code.lower()

    def test_uses_action_mapping_from_metadata_if_present(self):
        meta = _make_metadata("mixed")
        # Remove num_action_types so action_mapping is used as the source
        meta.pop("num_action_types", None)
        meta["action_mapping"] = {"0": "a", "1": "b"}  # 2 types
        code = generate_policy_py(meta, "mixed")
        assert "num_action_types=2" in code

    def test_uses_num_action_types_from_metadata_if_present(self):
        meta = _make_metadata("cooperative")
        meta["num_action_types"] = 5
        code = generate_policy_py(meta, "cooperative")
        assert "num_action_types=5" in code

    def test_load_state_dict_present(self):
        meta = _make_metadata("mixed")
        code = generate_policy_py(meta, "mixed")
        assert "load_state_dict" in code

    def test_policy_pt_reference_present(self):
        meta = _make_metadata("mixed")
        code = generate_policy_py(meta, "mixed")
        assert "policy.pt" in code

    def test_keeps_same_directory_warning_present(self):
        meta = _make_metadata("mixed")
        code = generate_policy_py(meta, "mixed")
        assert "same directory" in code.lower()


# ---------------------------------------------------------------------------
# generate_readme
# ---------------------------------------------------------------------------

class TestGenerateReadme:
    def test_contains_requirements_section(self):
        meta = _make_metadata("mixed")
        readme = generate_readme(meta, "mixed")
        assert "Requirements" in readme

    def test_contains_torch_requirement(self):
        meta = _make_metadata("mixed")
        readme = generate_readme(meta, "mixed")
        assert "torch" in readme.lower()

    def test_contains_usage_example(self):
        meta = _make_metadata("mixed")
        readme = generate_readme(meta, "mixed")
        assert "predict" in readme
        assert "from policy import predict" in readme or "predict(obs)" in readme

    def test_mixed_action_mapping(self):
        meta = _make_metadata("mixed")
        readme = generate_readme(meta, "mixed")
        assert "cooperate" in readme

    def test_competitive_action_mapping(self):
        meta = _make_metadata("competitive")
        readme = generate_readme(meta, "competitive")
        assert "build" in readme
        assert "attack" in readme

    def test_cooperative_action_mapping(self):
        meta = _make_metadata("cooperative")
        readme = generate_readme(meta, "cooperative")
        assert "idle" in readme.lower()

    def test_keep_same_directory_warning(self):
        meta = _make_metadata("mixed")
        readme = generate_readme(meta, "mixed")
        assert "same directory" in readme.lower()

    def test_config_hash_present(self):
        meta = _make_metadata("mixed")
        readme = generate_readme(meta, "mixed")
        assert "abc12345" in readme

    def test_seed_present(self):
        meta = _make_metadata("mixed")
        readme = generate_readme(meta, "mixed")
        assert "42" in readme

    def test_reproducibility_section_present(self):
        meta = _make_metadata("mixed")
        readme = generate_readme(meta, "mixed")
        assert "Reproducib" in readme


# ---------------------------------------------------------------------------
# build_export_zip
# ---------------------------------------------------------------------------

class TestBuildExportZip:
    def test_raises_file_not_found_when_policy_pt_missing(self, tmp_path: Path):
        member_dir = tmp_path / "league_000001"
        member_dir.mkdir()
        # No policy.pt — only metadata
        meta = _make_metadata("mixed")
        with pytest.raises(FileNotFoundError):
            build_export_zip(member_dir, meta, "mixed")

    def test_returns_bytes_when_policy_pt_exists(self, tmp_path: Path):
        member_dir = tmp_path / "league_000001"
        member_dir.mkdir()
        meta = _make_metadata("mixed")
        _write_fake_policy_pt(member_dir, obs_dim=33, num_action_types=4)

        result = build_export_zip(member_dir, meta, "mixed")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_zip_contains_exactly_three_files(self, tmp_path: Path):
        member_dir = tmp_path / "league_000001"
        member_dir.mkdir()
        meta = _make_metadata("mixed")
        _write_fake_policy_pt(member_dir, obs_dim=33, num_action_types=4)

        zip_bytes = build_export_zip(member_dir, meta, "mixed")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = set(zf.namelist())
        assert names == {"policy.py", "policy.pt", "README.md"}

    def test_zip_policy_py_contains_policy_network(self, tmp_path: Path):
        member_dir = tmp_path / "league_000001"
        member_dir.mkdir()
        meta = _make_metadata("mixed")
        _write_fake_policy_pt(member_dir, obs_dim=33, num_action_types=4)

        zip_bytes = build_export_zip(member_dir, meta, "mixed")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            py_src = zf.read("policy.py").decode("utf-8")
        assert "class PolicyNetwork" in py_src

    def test_zip_readme_contains_requirements(self, tmp_path: Path):
        member_dir = tmp_path / "league_000001"
        member_dir.mkdir()
        meta = _make_metadata("cooperative")
        _write_fake_policy_pt(member_dir, obs_dim=40, num_action_types=4)

        zip_bytes = build_export_zip(member_dir, meta, "cooperative")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            readme = zf.read("README.md").decode("utf-8")
        assert "Requirements" in readme

    def test_competitive_zip_correct_obs_dim(self, tmp_path: Path):
        member_dir = tmp_path / "league_000001"
        member_dir.mkdir()
        meta = _make_metadata("competitive")
        _write_fake_policy_pt(member_dir, obs_dim=93, num_action_types=4)

        zip_bytes = build_export_zip(member_dir, meta, "competitive")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            py_src = zf.read("policy.py").decode("utf-8")
        assert "obs_dim=93" in py_src
