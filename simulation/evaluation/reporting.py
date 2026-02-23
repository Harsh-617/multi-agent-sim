"""Write evaluation report artifacts (JSON + Markdown)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluator import PolicyResult
from .robustness import RobustnessResult

_REPORTS_ROOT = Path("storage/reports")


def config_hash(config_dict: dict[str, Any]) -> str:
    """Stable SHA-256 of canonically-serialized config (first 12 hex chars)."""
    canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _generate_report_id(cfg_hash: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"eval_{cfg_hash}_{ts}"


def write_report(
    policy_results: list[PolicyResult],
    *,
    config_dict: dict[str, Any],
    seeds: list[int],
    episodes_per_seed: int,
    report_root: Path = _REPORTS_ROOT,
) -> Path:
    """Write JSON and Markdown report artifacts.

    Returns the directory where the report was saved.
    """
    cfg_hash = config_hash(config_dict)
    report_id = _generate_report_id(cfg_hash)
    report_dir = report_root / report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    # --- report.json ---
    report_data = {
        "report_id": report_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg_hash,
        "seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "config": config_dict,
        "results": [pr.to_dict() for pr in policy_results],
    }
    json_path = report_dir / "report.json"
    json_path.write_text(
        json.dumps(report_data, indent=2, default=str), encoding="utf-8"
    )

    # --- report.md ---
    md_path = report_dir / "report.md"
    md_path.write_text(_build_markdown(report_data), encoding="utf-8")

    return report_dir


def _build_markdown(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Evaluation Report `{data['report_id']}`\n")
    lines.append(f"**Generated:** {data['timestamp']}  ")
    lines.append(f"**Config hash:** `{data['config_hash']}`  ")
    lines.append(f"**Seeds:** {data['seeds']}  ")
    lines.append(f"**Episodes per seed:** {data['episodes_per_seed']}\n")

    # Results table
    lines.append("## Results\n")
    lines.append(
        "| Policy | Source | Reward (mean +/- std) | Pool (mean) | Collapse % | Ep. Length | N |"
    )
    lines.append(
        "|--------|--------|-----------------------|-------------|------------|------------|---|"
    )

    skipped: list[dict[str, Any]] = []

    for r in data["results"]:
        if not r["available"]:
            skipped.append(r)
            continue
        reward_str = f"{r['mean_total_reward']:.4f} +/- {r['std_total_reward']:.4f}"
        lines.append(
            f"| {r['policy_name']} | {r['source']} | {reward_str} "
            f"| {r['mean_final_shared_pool']:.2f} "
            f"| {r['collapse_rate'] * 100:.1f} "
            f"| {r['mean_episode_length']:.1f} "
            f"| {r['n_episodes']} |"
        )

    if skipped:
        lines.append("\n## Skipped Policies\n")
        for s in skipped:
            lines.append(f"- **{s['policy_name']}** ({s['source']}): {s['skip_reason']}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Robustness report
# ---------------------------------------------------------------------------


def write_robustness_report(
    robustness_result: RobustnessResult,
    *,
    config_dict: dict[str, Any],
    report_root: Path = _REPORTS_ROOT,
) -> Path:
    """Write robustness sweep report (JSON + Markdown).

    Returns the directory where the report was saved.
    """
    cfg_hash = config_hash(config_dict)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_id = f"robust_{cfg_hash}_{ts}"
    report_dir = report_root / report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "report_id": report_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg_hash,
        "config": config_dict,
        **robustness_result.to_dict(),
    }

    json_path = report_dir / "report.json"
    json_path.write_text(
        json.dumps(report_data, indent=2, default=str), encoding="utf-8"
    )

    md_path = report_dir / "report.md"
    md_path.write_text(
        _build_robustness_markdown(report_data), encoding="utf-8"
    )

    return report_dir


def _build_robustness_markdown(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Robustness Report `{data['report_id']}`\n")
    lines.append(f"**Generated:** {data['timestamp']}  ")
    lines.append(f"**Config hash:** `{data['config_hash']}`  ")

    meta = data.get("metadata", {})
    lines.append(f"**Seeds:** {meta.get('seeds', [])}  ")
    lines.append(f"**Episodes per seed:** {meta.get('episodes_per_seed', 0)}  ")
    lines.append(f"**Sweeps:** {meta.get('sweep_count', 0)}\n")

    # Per-policy robustness table
    robustness = data.get("per_policy_robustness", {})
    available = {
        k: v for k, v in robustness.items() if v.get("n_sweeps_evaluated", 0) > 0
    }
    skipped = {
        k: v for k, v in robustness.items() if v.get("n_sweeps_evaluated", 0) == 0
    }

    lines.append("## Policy Robustness Summary\n")
    lines.append(
        "| Policy | Robustness Score | Mean Reward | Worst-Case Reward "
        "| Collapse % | Sweeps |"
    )
    lines.append(
        "|--------|-----------------|-------------|-------------------"
        "|------------|--------|"
    )

    # Sort by robustness_score descending
    ranked = sorted(
        available.items(),
        key=lambda kv: kv[1].get("robustness_score", 0),
        reverse=True,
    )

    for name, pr in ranked:
        lines.append(
            f"| {name} "
            f"| {pr['robustness_score']:.4f} "
            f"| {pr['overall_mean_reward']:.4f} "
            f"| {pr['worst_case_mean_reward']:.4f} "
            f"| {pr['collapse_rate_overall'] * 100:.1f} "
            f"| {pr['n_sweeps_evaluated']} |"
        )

    # Top-3 callout
    if ranked:
        lines.append("\n### Top-3 Most Robust Policies\n")
        for i, (name, pr) in enumerate(ranked[:3], 1):
            lines.append(
                f"{i}. **{name}** — robustness score {pr['robustness_score']:.4f}"
            )

    # Hardest sweep
    per_sweep = data.get("per_sweep_results", {})
    if per_sweep:
        sweep_avg: dict[str, float] = {}
        for sweep_name, policy_data in per_sweep.items():
            rewards = [
                v["mean_total_reward"]
                for v in policy_data.values()
                if v.get("available", True) and v.get("n_episodes", 0) > 0
            ]
            if rewards:
                sweep_avg[sweep_name] = sum(rewards) / len(rewards)

        if sweep_avg:
            hardest = min(sweep_avg, key=lambda k: sweep_avg[k])
            lines.append(
                f"\n### Hardest Sweep\n\n"
                f"**{hardest}** (avg reward across policies: "
                f"{sweep_avg[hardest]:.4f})"
            )

    # Skipped policies
    if skipped:
        lines.append("\n## Skipped Policies\n")
        for name, pr in skipped.items():
            lines.append(f"- **{name}**: No episodes evaluated (unavailable)")

    lines.append("")
    return "\n".join(lines)
