"""CLI entrypoint: python -m simulation.evaluation.run_eval

Usage:
    python -m simulation.evaluation.run_eval --config-id default --seeds 5 --episodes-per-seed 3 --top-k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from simulation.config.defaults import default_config
from simulation.config.schema import MixedEnvironmentConfig

from .evaluator import evaluate_policies
from .policy_set import resolve_policy_set
from .reporting import write_report

CONFIGS_DIR = Path("storage/configs")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run research-grade policy evaluation."
    )
    parser.add_argument(
        "--config-id",
        default="default",
        help='Config id in storage/configs/ or "default" for default_config().',
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of root seeds (1..seeds).",
    )
    parser.add_argument(
        "--episodes-per-seed",
        type=int,
        default=3,
        help="Episodes to run per seed per policy.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top-rated league members to include.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps for faster evaluation.",
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

    seeds = list(range(1, args.seeds + 1))

    print(f"Resolving policy set (top_k={args.top_k}) ...")
    specs = resolve_policy_set(top_k=args.top_k)
    available = [s for s in specs if s.available]
    skipped = [s for s in specs if not s.available]
    print(f"  {len(available)} available, {len(skipped)} skipped")
    for s in skipped:
        print(f"    SKIP {s.name}: {s.skip_reason}")

    print(
        f"Running evaluation: {len(seeds)} seeds x {args.episodes_per_seed} episodes ..."
    )
    results = evaluate_policies(
        config,
        specs,
        seeds=seeds,
        episodes_per_seed=args.episodes_per_seed,
        max_steps_override=args.max_steps,
    )

    print("Writing report ...")
    report_dir = write_report(
        results,
        config_dict=config_dict,
        seeds=seeds,
        episodes_per_seed=args.episodes_per_seed,
    )
    print(f"Report saved to: {report_dir}")


if __name__ == "__main__":
    main()
