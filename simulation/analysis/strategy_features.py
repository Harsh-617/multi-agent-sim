"""Extract per-policy behavioral features from a report dict."""

from __future__ import annotations

from typing import Dict, Optional


Features = Dict[str, Optional[float]]
PolicyFeatures = Dict[str, Features]


def extract_features(report: dict) -> PolicyFeatures:
    """Return per-policy feature dicts from an evaluation or robustness report.

    Handles both report kinds:
    - Evaluation reports have a ``results`` list of policy dicts.
    - Robustness reports have ``per_policy_robustness`` and ``per_sweep_results``.

    Missing fields are set to ``None``.
    """
    kind = report.get("kind", "")
    if kind == "robust" or "per_policy_robustness" in report:
        return _extract_robust(report)
    return _extract_eval(report)


def _extract_eval(report: dict) -> PolicyFeatures:
    out: PolicyFeatures = {}
    for entry in report.get("results", []):
        name = entry.get("policy_name", "")
        if not name or not entry.get("available", True):
            continue
        out[name] = {
            "mean_return": entry.get("mean_total_reward"),
            "worst_case_return": None,
            "collapse_rate": entry.get("collapse_rate"),
            "mean_final_pool": entry.get("mean_final_shared_pool"),
            "robustness_score": None,
        }
    return out


def _extract_robust(report: dict) -> PolicyFeatures:
    robustness = report.get("per_policy_robustness", {})

    # Compute mean_final_pool from per_sweep_results if available.
    pool_means: dict[str, float] = {}
    per_sweep = report.get("per_sweep_results", {})
    if per_sweep:
        accum: dict[str, list[float]] = {}
        for sweep_policies in per_sweep.values():
            for pname, pdata in sweep_policies.items():
                val = pdata.get("mean_final_shared_pool")
                if val is not None:
                    accum.setdefault(pname, []).append(val)
        for pname, vals in accum.items():
            pool_means[pname] = sum(vals) / len(vals)

    out: PolicyFeatures = {}
    for name, entry in robustness.items():
        out[name] = {
            "mean_return": entry.get("overall_mean_reward"),
            "worst_case_return": entry.get("worst_case_mean_reward"),
            "collapse_rate": entry.get("collapse_rate_overall"),
            "mean_final_pool": pool_means.get(name),
            "robustness_score": entry.get("robustness_score"),
        }
    return out
