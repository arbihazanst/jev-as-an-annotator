"""Smoke test wrapper for Jev (backward compatibility shim for openrouter_only)."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import openrouter_only.run_openrouter_smoke as _base
from openrouter_only.run_openrouter_smoke import run_smoke
from jev_only.jev_common import DEFAULT_MODEL


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _base.parse_args(argv)
    return args


# Override the default model in parser for run_jev_smoke
def _patched_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=None, help="Prefer OPENROUTER_API_KEY instead of this flag.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    from dataset_registry import available_datasets
    _available = available_datasets()
    parser.add_argument("--datasets", nargs="+", default=_available, help=f"Available: {_available}")
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.samples_per_class <= 0:
        parser.error("--samples-per-class must be positive")
    if args.retries < 0:
        parser.error("--retries must not be negative")
    return args


parse_args = _patched_parse_args


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run_smoke(parse_args())))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
