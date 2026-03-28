"""Write competitive evaluation report artifacts (JSON + Markdown)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPORTS_ROOT = Path("storage/reports")


def _config_hash(config_dict: dict[str, Any]) -> str:
    """Stable SHA-256 of canonically-serialized config (first 12 hex chars)."""
    canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def write_competitive_report(
    results: dict[str, Any],
    output_dir: Path | None = None,
) -> Path:
    """Write competitive robustness report (JSON + Markdown).

    Parameters
    ----------
    results:
        Output from ``run_competitive_robustness``.
    output_dir:
        If provided, write report here. Otherwise auto-generate under
        storage/reports/competitive_{hash}_{timestamp}/.

    Returns the directory where the report was saved.
    """
    cfg_hash = _config_hash(results.get("metadata", {}))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_id = f"competitive_{cfg_hash}_{ts}"

    if output_dir is not None:
        report_dir = output_dir / report_id
    else:
        report_dir = _REPORTS_ROOT / report_id

    report_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "report_id": report_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg_hash,
        **results,
    }

    # --- summary.json ---
    json_path = report_dir / "summary.json"
    json_path.write_text(
        json.dumps(report_data, indent=2, default=str), encoding="utf-8"
    )

    # --- summary.md ---
    md_path = report_dir / "summary.md"
    md_path.write_text(_build_markdown(report_data), encoding="utf-8")

    return report_dir


def _build_markdown(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Competitive Robustness Report `{data['report_id']}`\n")
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
        "| Winner Rate | Best Sweep | Worst Sweep | Sweeps |"
    )
    lines.append(
        "|--------|-----------------|-------------|-------------------"
        "|-------------|------------|-------------|--------|"
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
            f"| {pr['mean_winner_rate']:.4f} "
            f"| {pr.get('best_sweep', 'N/A')} "
            f"| {pr.get('worst_sweep', 'N/A')} "
            f"| {pr['n_sweeps_evaluated']} |"
        )

    # Overall ranking
    if ranked:
        lines.append("\n### Overall Ranking\n")
        for i, (name, pr) in enumerate(ranked, 1):
            lines.append(
                f"{i}. **{name}** — robustness score {pr['robustness_score']:.4f}, "
                f"winner rate {pr['mean_winner_rate']:.4f}"
            )

    # Hardest sweep
    per_sweep = data.get("per_sweep_results", {})
    if per_sweep:
        sweep_avg: dict[str, float] = {}
        for sweep_name, policy_data in per_sweep.items():
            rewards = [
                v["mean_reward"]
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

    # Best and worst sweep per policy
    if ranked:
        lines.append("\n### Per-Policy Best/Worst Sweeps\n")
        for name, pr in ranked:
            best = pr.get("best_sweep", "N/A")
            worst = pr.get("worst_sweep", "N/A")
            lines.append(f"- **{name}**: best = {best}, worst = {worst}")

    # Skipped policies
    if skipped:
        lines.append("\n## Skipped Policies\n")
        for name, pr in skipped.items():
            lines.append(f"- **{name}**: No episodes evaluated (unavailable)")

    lines.append("")
    return "\n".join(lines)
