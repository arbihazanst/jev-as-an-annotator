"""Smoke test for any decision model via OpenRouter Decisions API.

Dry-run validates datasets and payload construction but never writes fake predictions.
Real smoke predictions are isolated from the full-run directory by default.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
import time

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluate_decision_models import append_jsonl, load_dataset, summarize_predictions
from openrouter_only.openrouter_common import (
    DECISIONS_API_URL,
    DEFAULT_MODEL,
    RESULTS_DIR,
    build_record,
    call_openrouter,
    default_specs,
    get_api_key,
    question_for,
    run_manifest,
    validate_openrouter_response,
    write_json,
)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


async def run_smoke(args: argparse.Namespace) -> int:
    specs = default_specs()
    selected_by_dataset = {}
    quality_by_dataset = {}
    selected_items = []
    for dataset in args.datasets:
        items, quality = load_dataset(
            specs[dataset],
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
        selected_by_dataset[dataset] = len(items)
        quality_by_dataset[dataset] = quality
        selected_items.extend(items)
        print(f"Loaded {len(items)} smoke items for {dataset} ({args.samples_per_class}/class)")

    model_slug = args.model.replace("/", "_").replace(":", "_")
    prefix = f"dryrun_openrouter_smoke_{model_slug}" if args.dry_run else f"openrouter_smoke_{model_slug}"
    output_dir = args.output or (RESULTS_DIR / f"{prefix}_{timestamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "data_quality.json", quality_by_dataset)
    write_json(
        output_dir / "run_config.json",
        run_manifest(
            mode="dry_run" if args.dry_run else "smoke",
            model=args.model,
            datasets=args.datasets,
            seed=args.seed,
            selected_rows=selected_by_dataset,
            specs=specs,
            extra={
                "samples_per_class": args.samples_per_class,
                "retries": args.retries,
                "timeout_seconds": args.timeout,
            },
        ),
    )

    if args.dry_run:
        preview = []
        for item in selected_items[: min(2, len(selected_items))]:
            preview.append(
                {
                    "model": args.model,
                    "state": item.text,
                    "questions": question_for(item.dataset),
                    "dataset": item.dataset,
                    "id": item.record_id,
                }
            )
        write_json(output_dir / "payload_preview.json", preview)
        print(f"Dry-run passed for {len(selected_items)} items. No API calls and no predictions were written.")
        print(f"Artifacts: {output_dir.resolve()}")
        return 0

    api_key = get_api_key(args.api_key)
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (env var, .env, or --api-key)")

    predictions_path = output_dir / "predictions.jsonl"
    errors_path = output_dir / "errors.jsonl"
    if predictions_path.exists() and predictions_path.stat().st_size and not args.clean:
        raise RuntimeError(f"{predictions_path} already exists; use --clean or a new --output")
    if args.clean:
        for path in (predictions_path, errors_path):
            if path.exists():
                path.unlink()

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout = httpx.Timeout(args.timeout)
    errors = 0

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        for index, item in enumerate(selected_items, start=1):
            spec = specs[item.dataset]
            questions = question_for(item.dataset)
            started = time.perf_counter()
            record = None
            last_error: Exception | None = None

            for attempt in range(1, args.retries + 2):
                try:
                    data = await call_openrouter(client, args.model, item.text, questions)
                    choice, probabilities, confidence, usage = validate_openrouter_response(data, spec)
                    record = build_record(
                        model=args.model,
                        item=item,
                        status="ok",
                        latency_ms=(time.perf_counter() - started) * 1000,
                        attempts=attempt,
                        predicted_label=choice,
                        probabilities=probabilities,
                        confidence=confidence,
                        usage=usage,
                        error=None,
                    )
                    break
                except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                    last_error = exc
                    retryable = True
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    retryable = exc.response.status_code in {408, 409, 425, 429} or exc.response.status_code >= 500
                except Exception as exc:
                    last_error = exc
                    retryable = False

                if not retryable or attempt > args.retries:
                    record = build_record(
                        model=args.model,
                        item=item,
                        status="error",
                        latency_ms=(time.perf_counter() - started) * 1000,
                        attempts=attempt,
                        predicted_label=None,
                        probabilities=None,
                        confidence=None,
                        usage=None,
                        error=f"{type(last_error).__name__}: {last_error}",
                    )
                    break
                await asyncio.sleep(args.retry_base_seconds * (2 ** (attempt - 1)) + random.random() * 0.25)

            assert record is not None
            append_jsonl(predictions_path, record)
            if record["status"] == "error":
                errors += 1
                append_jsonl(errors_path, record)
            print(
                f"[{index:02d}/{len(selected_items):02d}] {item.dataset}#{item.record_id} "
                f"gold={item.gold_label} pred={record['predicted_label']} status={record['status']}"
            )

    summarize_predictions(
        predictions_path,
        output_dir,
        specs,
        models=[args.model],
        datasets=args.datasets,
        calibration_bins=5,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(f"Smoke test complete. errors={errors}. Results: {output_dir.resolve()}")
    return 2 if errors else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=None, help="Prefer OPENROUTER_API_KEY instead of this flag.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter decision model (default: {DEFAULT_MODEL}). Any OpenRouter decision model can be specified.",
    )
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


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run_smoke(parse_args())))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
