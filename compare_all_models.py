"""Combine model prediction files and build one fair cross-model evaluation report.

By default, all compared models must have identical successful example coverage within
each dataset. Use --require-full-dataset for the final benchmark to additionally require
coverage of every row in the source datasets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import typing as t

import pandas as pd

from dataset_registry import DatasetSpec, available_datasets, get_specs
from evaluate_decision_models import (
    file_sha256,
    load_latest_records,
    stable_id,
    summarize_predictions,
    write_json,
)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def specs() -> dict[str, DatasetSpec]:
    return get_specs()


def expected_rows(spec: DatasetSpec) -> dict[str, tuple[str, str]]:
    frame = pd.read_csv(spec.path)
    result: dict[str, tuple[str, str]] = {}
    for _, row in frame.iterrows():
        record_id = stable_id(row["id"])
        text = str(row[spec.text_column])
        label = str(row[spec.label_column])
        result[record_id] = (label, hashlib.sha256(text.encode("utf-8")).hexdigest())
    return result


def records_from_inputs(paths: list[Path]) -> tuple[dict[tuple[str, str, str], dict[str, t.Any]], list[str]]:
    merged: dict[tuple[str, str, str], dict[str, t.Any]] = {}
    model_order: list[str] = []
    for path in paths:
        if not path.exists() or not path.stat().st_size:
            raise FileNotFoundError(f"prediction file is missing or empty: {path}")
        latest = load_latest_records(path)
        for key, record in latest.items():
            model = key[0]
            if model not in model_order:
                model_order.append(model)
            if key in merged:
                previous = merged[key]
                comparable_fields = (
                    "status",
                    "gold_label",
                    "predicted_label",
                    "probabilities",
                    "text_sha256",
                )
                if any(previous.get(field) != record.get(field) for field in comparable_fields):
                    raise ValueError(f"conflicting duplicate record across input files: {key}")
                continue
            merged[key] = record
    return merged, model_order


def validate_records_against_datasets(
    records: dict[tuple[str, str, str], dict[str, t.Any]],
    selected_datasets: list[str],
    selected_models: list[str],
    benchmark_specs: dict[str, DatasetSpec],
) -> dict[str, dict[str, tuple[str, str]]]:
    expected = {dataset: expected_rows(benchmark_specs[dataset]) for dataset in selected_datasets}
    for (model, dataset, record_id), record in records.items():
        if model not in selected_models or dataset not in selected_datasets:
            continue
        if record_id not in expected[dataset]:
            raise ValueError(f"{model}/{dataset}/{record_id}: ID not present in source dataset")
        expected_gold, expected_hash = expected[dataset][record_id]
        if str(record.get("gold_label")) != expected_gold:
            raise ValueError(
                f"{model}/{dataset}/{record_id}: gold label mismatch "
                f"({record.get('gold_label')!r} != {expected_gold!r})"
            )
        record_hash = record.get("text_sha256")
        if record_hash is not None and record_hash != expected_hash:
            raise ValueError(f"{model}/{dataset}/{record_id}: text hash does not match source dataset")
    return expected


def validate_coverage(
    records: dict[tuple[str, str, str], dict[str, t.Any]],
    *,
    models: list[str],
    datasets: list[str],
    expected: dict[str, dict[str, tuple[str, str]]],
    require_full_dataset: bool,
) -> dict[str, dict[str, int]]:
    coverage_report: dict[str, dict[str, int]] = {}
    for dataset in datasets:
        successful_sets: dict[str, set[str]] = {}
        for model in models:
            successful_sets[model] = {
                record_id
                for (record_model, record_dataset, record_id), record in records.items()
                if record_model == model
                and record_dataset == dataset
                and record.get("status") == "ok"
            }
        reference_model = models[0]
        reference = successful_sets[reference_model]
        for model in models[1:]:
            if successful_sets[model] != reference:
                missing = sorted(reference - successful_sets[model])[:10]
                extra = sorted(successful_sets[model] - reference)[:10]
                raise ValueError(
                    f"incomparable successful coverage for {dataset}: {reference_model} has "
                    f"{len(reference)} IDs, {model} has {len(successful_sets[model])}. "
                    f"missing_from_{model}={missing}, extra_in_{model}={extra}"
                )
        if require_full_dataset and reference != set(expected[dataset]):
            missing = sorted(set(expected[dataset]) - reference)[:10]
            raise ValueError(
                f"{dataset} is not a full-dataset comparison: successful={len(reference)} "
                f"expected={len(expected[dataset])}; first missing IDs={missing}"
            )
        coverage_report[dataset] = {model: len(successful_sets[model]) for model in models}
    return coverage_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=None, help="Optional explicit model order/subset.")
    _available = available_datasets()
    parser.add_argument("--datasets", nargs="+", default=_available, help=f"Available: {_available}")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-full-dataset", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    benchmark_specs = specs()
    merged, discovered_models = records_from_inputs(args.inputs)
    models = args.models or discovered_models
    if len(models) < 2:
        raise ValueError("comparison requires at least two models")
    unknown = [model for model in models if model not in discovered_models]
    if unknown:
        raise ValueError(f"requested models not present in inputs: {unknown}")

    expected = validate_records_against_datasets(merged, args.datasets, models, benchmark_specs)
    coverage = validate_coverage(
        merged,
        models=models,
        datasets=args.datasets,
        expected=expected,
        require_full_dataset=args.require_full_dataset,
    )

    output_dir = args.output or (RESULTS_DIR / f"all_models_{timestamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "combined_predictions.jsonl"
    with combined_path.open("w", encoding="utf-8") as handle:
        for model in models:
            for dataset in args.datasets:
                rows = [
                    record
                    for (record_model, record_dataset, _), record in merged.items()
                    if record_model == model and record_dataset == dataset
                ]
                rows.sort(key=lambda record: str(record["id"]))
                for record in rows:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in args.inputs
        ],
        "models": models,
        "datasets": args.datasets,
        "require_full_dataset": args.require_full_dataset,
        "coverage_successful": coverage,
        "dataset_sha256": {
            dataset: file_sha256(benchmark_specs[dataset].path) for dataset in args.datasets
        },
        "combined_predictions": str(combined_path.resolve()),
    }
    write_json(output_dir / "comparison_manifest.json", manifest)

    summarize_predictions(
        combined_path,
        output_dir,
        benchmark_specs,
        models=models,
        datasets=args.datasets,
        calibration_bins=args.calibration_bins,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(f"Comparison complete for {len(models)} models: {', '.join(models)}")
    print(f"Results: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
