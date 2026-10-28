"""Full benchmark runner wrapper for Jev (backward compatibility shim for openrouter_only)."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import openrouter_only.run_openrouter_full as _base
from openrouter_only.run_openrouter_full import (
    evaluate_one,
    timestamp,
    EvalItem,
    append_jsonl,
    load_dataset,
    load_latest_records,
    summarize_predictions,
    validate_cached_inputs,
)
from openrouter_only.openrouter_common import get_api_key

DEFAULT_MODEL = "typesafe/jev-1.13"
async def run_full(args: argparse.Namespace) -> int:
    import openrouter_only.run_openrouter_full as _runner
    orig_attrs = {}
    for attr in ("load_dataset", "get_api_key", "summarize_predictions", "default_specs", "question_for"):
        if attr in globals():
            orig_attrs[attr] = getattr(_runner, attr)
            setattr(_runner, attr, globals()[attr])
    try:
        return await _runner.run_full(args)
    finally:
        for attr, orig in orig_attrs.items():
            setattr(_runner, attr, orig)



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=None, help="Prefer OPENROUTER_API_KEY instead of this flag.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    from dataset_registry import available_datasets
    _available = available_datasets()
    parser.add_argument("--datasets", nargs="+", default=_available, help=f"Available: {_available}")
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-appended-data", action="store_true",
                        help="Validate cached IDs, texts and labels before accepting a changed dataset hash.")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Rebuild metrics from saved predictions without API calls.")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.samples_per_class is not None and args.samples_per_class <= 0:
        parser.error("--samples-per-class must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.retries < 0:
        parser.error("--retries must not be negative")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.resume and args.clean:
        parser.error("--resume and --clean cannot be used together")
    if args.summarize_only and (args.clean or args.dry_run):
        parser.error("--summarize-only cannot be combined with --clean or --dry-run")
    if args.summarize_only and args.output is None:
        parser.error("--summarize-only requires --output")
    if args.allow_appended_data and not (args.resume or args.summarize_only):
        parser.error("--allow-appended-data requires --resume or --summarize-only")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run_full(parse_args())))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
