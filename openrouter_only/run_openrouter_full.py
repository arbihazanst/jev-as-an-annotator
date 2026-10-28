"""Full, resumable benchmark for any decision model via OpenRouter Decisions API.

The runner shares dataset loading, task definitions, and metric code with the SI-Lab
benchmark. Dry-run performs validation only and never writes simulated predictions.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
import typing as t

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluate_decision_models import (
    EvalItem,
    append_jsonl,
    load_dataset,
    load_latest_records,
    summarize_predictions,
    validate_cached_inputs,
)
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


async def evaluate_one(
    client: httpx.AsyncClient,
    *,
    model: str,
    item: EvalItem,
    spec,
    retries: int,
    retry_base_seconds: float,
) -> dict[str, t.Any]:
    started = time.perf_counter()
    last_error: Exception | None = None

    for attempt in range(1, retries + 2):
        try:
            data = await call_openrouter(client, model, item.text, question_for(item.dataset))
            choice, probabilities, confidence, usage = validate_openrouter_response(data, spec)
            return build_record(
                model=model,
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
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            last_error = exc
            retryable = True
            retry_after = None
        except httpx.HTTPStatusError as exc:
            last_error = exc
            retryable = exc.response.status_code in {408, 409, 425, 429} or exc.response.status_code >= 500
            retry_after = exc.response.headers.get("retry-after")
        except Exception as exc:
            last_error = exc
            retryable = False
            retry_after = None

        if not retryable or attempt > retries:
            break
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = retry_base_seconds * (2 ** (attempt - 1))
        else:
            delay = retry_base_seconds * (2 ** (attempt - 1))
        delay += random.uniform(0.0, min(1.0, delay * 0.25))
        await asyncio.sleep(delay)

    return build_record(
        model=model,
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


async def run_full(args: argparse.Namespace) -> int:
    specs = default_specs()
    items_by_dataset: dict[str, list[EvalItem]] = {}
    quality_by_dataset = {}
    for dataset in args.datasets:
        items, quality = load_dataset(
            specs[dataset],
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
        items_by_dataset[dataset] = items
        quality_by_dataset[dataset] = quality
        print(f"Loaded {len(items)} items for {dataset}")

    all_items = [item for dataset in args.datasets for item in items_by_dataset[dataset]]
    if args.limit is not None:
        all_items = all_items[: args.limit]
        print(f"Capped total items to {len(all_items)}")

    model_slug = args.model.replace("/", "_").replace(":", "_")
    prefix = f"dryrun_openrouter_full_{model_slug}" if args.dry_run else f"openrouter_full_{model_slug}"
    if args.resume and args.output is None:
        raise RuntimeError("--resume requires --output so the exact prior run can be resumed")
    output_dir = args.output or (RESULTS_DIR / f"{prefix}_{timestamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = {
        dataset: sum(1 for item in all_items if item.dataset == dataset) for dataset in args.datasets
    }
    comparable = {
        "mode": "full",
        "model": args.model,
        "datasets": args.datasets,
        "samples_per_class": args.samples_per_class,
        "limit": args.limit,
        "seed": args.seed,
        "dataset_sha256": {
            dataset: quality_by_dataset[dataset]["sha256"] for dataset in args.datasets
        },
    }
    config_path = output_dir / "run_config.json"
    predictions_path = output_dir / "predictions.jsonl"
    validated = set()
    if args.resume or args.summarize_only:
        validated = validate_cached_inputs(
            predictions_path,
            {ds: [item for item in all_items if item.dataset == ds] for ds in args.datasets},
            [args.model],
        )

    if (args.resume or args.summarize_only) and config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        old_comp = existing.get("comparable_config", {})
        for key in ("mode", "model", "seed"):
            if old_comp.get(key) != comparable.get(key):
                raise RuntimeError(
                    f"resume configuration mismatch on {key!r}: "
                    f"existing={old_comp.get(key)!r}, new={comparable.get(key)!r}"
                )
        old_sha = old_comp.get("dataset_sha256", {})
        for ds, sha in comparable.get("dataset_sha256", {}).items():
            if (ds in old_sha and old_sha[ds] != sha
                    and not (args.allow_appended_data and ds in validated)):
                raise RuntimeError(
                    f"resume SHA mismatch for {ds!r}: data has changed since the original run"
                )

    write_json(output_dir / "data_quality.json", quality_by_dataset)
    manifest = run_manifest(
        mode="dry_run" if args.dry_run else "full",
        model=args.model,
        datasets=args.datasets,
        seed=args.seed,
        selected_rows=selected_rows,
        specs=specs,
        extra={
            "comparable_config": comparable,
            "runtime": {
                "concurrency": args.concurrency,
                "timeout_seconds": args.timeout,
                "retries": args.retries,
                "retry_base_seconds": args.retry_base_seconds,
                "calibration_bins": args.calibration_bins,
                "bootstrap_samples": args.bootstrap_samples,
            },
        },
    )
    if not args.summarize_only:
        write_json(config_path, manifest)

    if args.dry_run:
        preview = [
            {
                "model": args.model,
                "dataset": item.dataset,
                "id": item.record_id,
                "state": item.text,
                "questions": question_for(item.dataset),
            }
            for item in all_items[: min(2, len(all_items))]
        ]
        write_json(output_dir / "payload_preview.json", preview)
        print(f"Dry-run passed for {len(all_items)} items. No API calls and no predictions were written.")
        print(f"Artifacts: {output_dir.resolve()}")
        return 0

    errors_path = output_dir / "errors.jsonl"
    if args.clean:
        for path in (predictions_path, errors_path):
            if path.exists():
                path.unlink()
    elif predictions_path.exists() and predictions_path.stat().st_size and not (args.resume or args.summarize_only):
        raise RuntimeError(f"{predictions_path} already has results; use --resume, --clean, or another --output")

    latest = load_latest_records(predictions_path) if args.resume or args.summarize_only else {}
    successful = {key for key, record in latest.items() if record.get("status") == "ok"}
    pending = [
        item for item in all_items if (args.model, item.dataset, item.record_id) not in successful
    ]
    print(
        f"Total={len(all_items)} already_completed={len(all_items)-len(pending)} pending={len(pending)}"
    )
    if args.summarize_only or not pending:
        if not latest:
            raise RuntimeError("--summarize-only requires existing predictions.jsonl records")
        summarize_predictions(
            predictions_path, output_dir, specs, models=[args.model], datasets=args.datasets,
            calibration_bins=args.calibration_bins, bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        print(f"Metrics rebuilt without API calls. Results: {output_dir.resolve()}")
        return 0

    api_key = get_api_key(args.api_key)
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (env var, .env, or --api-key)")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout = httpx.Timeout(args.timeout)
    queue: asyncio.Queue[EvalItem | None] = asyncio.Queue(maxsize=max(args.concurrency * 4, 8))
    write_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    counters = {"completed": 0, "errors": 0}

    async def producer() -> None:
        for item in pending:
            await queue.put(item)
        for _ in range(args.concurrency):
            await queue.put(None)

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        async def worker() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    record = await evaluate_one(
                        client,
                        model=args.model,
                        item=item,
                        spec=specs[item.dataset],
                        retries=args.retries,
                        retry_base_seconds=args.retry_base_seconds,
                    )
                    async with write_lock:
                        append_jsonl(predictions_path, record)
                        if record["status"] == "error":
                            append_jsonl(errors_path, record)
                    async with progress_lock:
                        counters["completed"] += 1
                        if record["status"] == "error":
                            counters["errors"] += 1
                        completed = counters["completed"]
                        if completed % args.progress_every == 0 or completed == len(pending):
                            print(
                                f"Progress {completed}/{len(pending)} errors={counters['errors']}",
                                flush=True,
                            )
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
        producer_task = asyncio.create_task(producer())
        try:
            await asyncio.gather(producer_task, *workers)
        except Exception:
            producer_task.cancel()
            for task in workers:
                task.cancel()
            await asyncio.gather(producer_task, *workers, return_exceptions=True)
            raise

    if predictions_path.exists() and predictions_path.stat().st_size:
        summarize_predictions(
            predictions_path,
            output_dir,
            specs,
            models=[args.model],
            datasets=args.datasets,
            calibration_bins=args.calibration_bins,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    print(f"Full OpenRouter run complete. errors={counters['errors']}. Results: {output_dir.resolve()}")
    return 2 if counters["errors"] else 0


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
