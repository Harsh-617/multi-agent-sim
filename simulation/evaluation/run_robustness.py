"""CLI entrypoint: python -m simulation.evaluation.run_robustness

Usage:
    python -m simulation.evaluation.run_robustness \
        --config-id default --seeds 3 --episodes-per-seed 2 \
        --top-k 2 --max-steps 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from simulation.config.defaults import default_config
from simulation.config.schema import MixedEnvironmentConfig

from .policy_set import resolve_policy_set
from .reporting import write_robustness_report
from .robustness import evaluate_robustness
from .sweeps import build_default_sweeps

CONFIGS_DIR = Path("storage/configs")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run robustness sweep evaluation across environment variants."
    )
    parser.add_argument(
        "--config-id",
        default="default",
        help='Config id in storage/configs/ or "default" for default_config().',
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of root seeds (generates base_seed .. base_seed+N-1).",
    )
    parser.add_argument(
        "--episodes-per-seed",
        type=int,
        default=2,
        help="Episodes to run per seed per policy per sweep.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of top-rated league members to include.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps for faster evaluation.",
    )
    parser.add_argument(
        "--sweeps",
        type=str,
        default=None,
        help="Comma-separated filter: sweep names or tags to include.",
    )
    parser.add_argument(
        "--limit-sweeps",
        type=int,
        default=None,
        help="Maximum number of sweeps to run (caps runtime).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Load config
    if args.config_id == "default":
        config = default_config()
        config_dict = json.loads(config.model_dump_json())
    else:
        config_path = CONFIGS_DIR / f"{args.config_id}.json"
        if not config_path.exists():
            print(f"ERROR: config not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        raw = config_path.read_text(encoding="utf-8")
        config = MixedEnvironmentConfig.model_validate_json(raw)
        config_dict = json.loads(raw)

    base_seed = config.identity.seed
    seeds = [base_seed + i for i in range(args.seeds)]

    # Build and filter sweeps
    all_sweeps = build_default_sweeps()

    if args.sweeps:
        filters = {f.strip() for f in args.sweeps.split(",")}
        all_sweeps = [
            s for s in all_sweeps
            if s.name in filters or any(t in filters for t in s.tags)
        ]

    if args.limit_sweeps is not None:
        all_sweeps = all_sweeps[: args.limit_sweeps]

    if not all_sweeps:
        print("ERROR: No sweeps matched the filter.", file=sys.stderr)
        sys.exit(1)

    print(f"Sweeps selected: {len(all_sweeps)}")
    for s in all_sweeps:
        print(f"  - {s.name}: {s.description}")

    # Resolve policies
    print(f"\nResolving policy set (top_k={args.top_k}) ...")
    specs = resolve_policy_set(top_k=args.top_k)
    available = [s for s in specs if s.available]
    skipped = [s for s in specs if not s.available]
    print(f"  {len(available)} available, {len(skipped)} skipped")
    for s in skipped:
        print(f"    SKIP {s.name}: {s.skip_reason}")

    total_evals = len(all_sweeps) * len(available) * len(seeds) * args.episodes_per_seed
    print(
        f"\nRunning robustness sweep: {len(all_sweeps)} sweeps x "
        f"{len(available)} policies x {len(seeds)} seeds x "
        f"{args.episodes_per_seed} episodes = {total_evals} total episodes ..."
    )

    result = evaluate_robustness(
        config,
        specs,
        all_sweeps,
        seeds=seeds,
        episodes_per_seed=args.episodes_per_seed,
        max_steps_override=args.max_steps,
    )

    print("\nWriting robustness report ...")
    report_dir = write_robustness_report(
        result,
        config_dict=config_dict,
    )
    print(f"Report saved to: {report_dir}")

    # Print quick summary
    ranked = sorted(
        result.per_policy_robustness.values(),
        key=lambda pr: pr.robustness_score,
        reverse=True,
    )
    print("\n--- Quick Summary ---")
    for pr in ranked:
        if pr.n_sweeps_evaluated > 0:
            print(
                f"  {pr.policy_name}: "
                f"robustness={pr.robustness_score:.4f}  "
                f"mean={pr.overall_mean_reward:.4f}  "
                f"worst={pr.worst_case_mean_reward:.4f}  "
                f"collapse={pr.collapse_rate_overall * 100:.1f}%"
            )


if __name__ == "__main__":
    main()
