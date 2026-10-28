"""Evaluate SI-Lab decision models on the local sentiment/emotion datasets.

The runner is intentionally resumable: every completed example is appended to
``predictions.jsonl`` before the next one is scheduled. API credentials are read
only from ``SI_LAB_API_KEY`` and are never written to the output directory.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import typing as t

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from si_lab_decision_client import (
    DEFAULT_OLLAMA_BASE_URL,
    DecisionAnswer,
    SiLabDecisionError,
    SiLabEndpointUnavailable,
    SiLabOllamaDecisionClient,
    SiLabRequestError,
)
from dataset_registry import (
    DatasetSpec,
    available_datasets,
    get_specs,
    question_for,
)


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

SCRIPT_VERSION = "1.2.0"
DEFAULT_MODELS = ["nimble:9b", "tev1:0.8b", "tev1:4b"]
RETRYABLE_NETWORK_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


# DatasetSpec is imported from dataset_registry (re-exported here for backward compat)
# DatasetSpec = DatasetSpec  # already in namespace via import above


@dataclass(frozen=True)
class EvalItem:
    dataset: str
    record_id: str
    text: str
    gold_label: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(value: t.Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: t.Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, t.Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def _check_resume_compat(
    old: dict[str, t.Any], new: dict[str, t.Any], *, context: str = "resume",
    validated_datasets: set[str] | None = None,
) -> None:
    """Validate that a new run is compatible with an existing run_config.

    Checks model, seed, and transport match.  For overlapping datasets,
    verifies the data SHA is unchanged.  Allows *adding* new datasets.
    """
    for key in ("models", "transport", "seed"):
        if old.get(key) != new.get(key):
            raise RuntimeError(
                f"{context} configuration mismatch on {key!r}: "
                f"existing={old.get(key)!r}, new={new.get(key)!r}"
            )
    old_sha = old.get("dataset_sha256", {})
    for ds, sha in new.get("dataset_sha256", {}).items():
        if ds in old_sha and old_sha[ds] != sha and ds not in (validated_datasets or set()):
            raise RuntimeError(
                f"{context} SHA mismatch for {ds!r}: data has changed since the original run"
            )


def validate_cached_inputs(
    predictions_path: Path,
    items_by_dataset: dict[str, list[EvalItem]],
    models: list[str],
) -> set[str]:
    """Prove cached IDs still refer to the same text and gold label."""
    current = {
        (dataset, item.record_id): item
        for dataset, items in items_by_dataset.items() for item in items
    }
    validated: set[str] = set()
    for (model, dataset, record_id), record in load_latest_records(predictions_path).items():
        if model not in models or dataset not in items_by_dataset:
            continue
        item = current.get((dataset, record_id))
        if item is None or (
            record.get("text_sha256") != hashlib.sha256(item.text.encode("utf-8")).hexdigest()
            or record.get("gold_label") != item.gold_label
        ):
            raise RuntimeError(
                f"Cannot reuse cached prediction for {model}/{dataset}/{record_id}: "
                "existing row was removed or its text/label changed. Use a new output directory."
            )
        validated.add(dataset)
    return validated


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()



def load_dataset(
    spec: DatasetSpec,
    *,
    samples_per_class: int | None,
    seed: int,
    deduplicate_text: bool = False,
) -> tuple[list[EvalItem], dict[str, t.Any]]:
    if not spec.path.exists():
        raise FileNotFoundError(f"dataset not found: {spec.path}")

    frame = pd.read_csv(spec.path)
    if spec.text_column not in frame.columns:
        for candidate in ("text", "tweet", "comment", "review", "sentence", "content", "body", "ulasan", "yorum"):
            if candidate in frame.columns:
                spec = replace(spec, text_column=candidate)
                break
    required = {"id", spec.text_column, spec.label_column}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{spec.name} is missing required columns: {missing_columns}"
        )

    missing_ids = int(frame["id"].isna().sum())
    missing_texts = int(frame[spec.text_column].isna().sum())
    missing_labels = int(frame[spec.label_column].isna().sum())
    duplicate_ids = int(frame["id"].duplicated(keep=False).sum())

    label_values = frame[spec.label_column].dropna().astype(str)
    if not spec.labels or getattr(spec, "auto_labels", False):
        spec = replace(spec, labels=tuple(sorted(label_values.unique())))
    unexpected_labels = sorted(set(label_values) - set(spec.labels))
    class_counts = {
        label: int((label_values == label).sum()) for label in spec.labels
    }

    text_values = frame[spec.text_column].fillna("").astype(str)
    normalized = text_values.map(normalized_text)
    duplicate_text_rows = int(normalized.duplicated(keep=False).sum())
    quality_frame = pd.DataFrame(
        {"normalized_text": normalized, "label": label_values.reindex(frame.index)}
    )
    conflicting_groups = int(
        (quality_frame.groupby("normalized_text", dropna=False)["label"].nunique() > 1).sum()
    )

    quality = {
        "dataset": spec.name,
        "path": str(spec.path.resolve()),
        "sha256": file_sha256(spec.path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "class_counts": class_counts,
        "missing_ids": missing_ids,
        "missing_texts": missing_texts,
        "missing_labels": missing_labels,
        "duplicate_id_rows": duplicate_ids,
        "duplicate_text_rows": duplicate_text_rows,
        "conflicting_duplicate_text_groups": conflicting_groups,
        "unexpected_labels": unexpected_labels,
    }

    fatal_issues: list[str] = []
    if missing_ids:
        fatal_issues.append(f"{missing_ids} missing IDs")
    if missing_texts:
        fatal_issues.append(f"{missing_texts} missing texts")
    if missing_labels:
        fatal_issues.append(f"{missing_labels} missing labels")
    if duplicate_ids:
        fatal_issues.append(f"{duplicate_ids} rows with duplicate IDs")
    if unexpected_labels:
        fatal_issues.append(f"unexpected labels: {unexpected_labels}")
    if fatal_issues:
        raise ValueError(f"invalid {spec.name} dataset: " + "; ".join(fatal_issues))

    selected = frame.copy()
    quality["deduplicate_text"] = deduplicate_text
    quality["deduplication_removed_rows"] = 0
    if deduplicate_text:
        selected["_normalized_text"] = normalized
        conflicting_texts = set(
            quality_frame.groupby("normalized_text", dropna=False)["label"]
            .nunique()
            .loc[lambda values: values > 1]
            .index
        )
        original_rows = len(selected)
        selected = selected[
            ~selected["_normalized_text"].isin(conflicting_texts)
        ].drop_duplicates(subset=["_normalized_text"], keep="first")
        selected = selected.drop(columns=["_normalized_text"])
        quality["deduplication_removed_rows"] = original_rows - len(selected)

    if samples_per_class is not None:
        pieces: list[pd.DataFrame] = []
        for label in spec.labels:
            group = selected[selected[spec.label_column].astype(str) == label]
            sample_size = min(samples_per_class, len(group))
            pieces.append(group.sample(n=sample_size, random_state=seed))
        selected = pd.concat(pieces, ignore_index=True)
        selected = selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    items = [
        EvalItem(
            dataset=spec.name,
            record_id=stable_id(row["id"]),
            text=str(row[spec.text_column]),
            gold_label=str(row[spec.label_column]),
        )
        for _, row in selected.iterrows()
    ]
    quality["selected_rows"] = len(items)
    quality["samples_per_class"] = samples_per_class
    return items, quality


def load_latest_records(path: Path) -> dict[tuple[str, str, str], dict[str, t.Any]]:
    latest: dict[tuple[str, str, str], dict[str, t.Any]] = {}
    if not path.exists():
        return latest

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = (
                    str(record["model"]),
                    str(record["dataset"]),
                    str(record["id"]),
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"invalid JSONL record at {path}:{line_number}"
                ) from exc
            latest[key] = record
    return latest


# question_for() is imported from dataset_registry


def answer_to_record(
    *,
    model: str,
    item: EvalItem,
    answer: DecisionAnswer,
    latency_ms: float,
    attempts: int,
) -> dict[str, t.Any]:
    return {
        "timestamp": utc_now(),
        "status": "ok",
        "model": model,
        "dataset": item.dataset,
        "id": item.record_id,
        "text_sha256": hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
        "gold_label": item.gold_label,
        "predicted_label": answer.choice,
        "probabilities": answer.probabilities,
        "systemone_confidence": answer.confidence,
        "transport": answer.transport,
        "latency_ms": round(latency_ms, 3),
        "attempts": attempts,
        "usage": answer.usage,
        "error": None,
    }


async def evaluate_item(
    client: SiLabOllamaDecisionClient,
    *,
    model: str,
    item: EvalItem,
    expected_labels: tuple[str, ...],
    answer_name: str,
    retries: int,
    retry_base_seconds: float,
) -> dict[str, t.Any]:
    started = time.perf_counter()
    attempts = 0
    last_error: Exception | None = None

    for attempt in range(1, retries + 2):
        attempts = attempt
        try:
            answers = await client.adecide(item.text, question_for(item.dataset))
            answer = answers[answer_name]
            if answer.choice not in expected_labels:
                raise SiLabDecisionError(
                    f"invalid label {answer.choice!r}; expected one of {expected_labels}"
                )
            return answer_to_record(
                model=model,
                item=item,
                answer=answer,
                latency_ms=(time.perf_counter() - started) * 1000,
                attempts=attempts,
            )
        except SiLabEndpointUnavailable:
            raise
        except SiLabRequestError as exc:
            last_error = exc
            retryable = exc.retryable
        except RETRYABLE_NETWORK_ERRORS as exc:
            last_error = exc
            retryable = True
        except Exception as exc:  # malformed model output is a final record, not a crash
            last_error = exc
            retryable = False

        if not retryable or attempt > retries:
            break
        delay = retry_base_seconds * (2 ** (attempt - 1))
        delay += random.uniform(0.0, min(1.0, delay * 0.25))
        await asyncio.sleep(delay)

    return {
        "timestamp": utc_now(),
        "status": "error",
        "model": model,
        "dataset": item.dataset,
        "id": item.record_id,
        "text_sha256": hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
        "gold_label": item.gold_label,
        "predicted_label": None,
        "probabilities": None,
        "systemone_confidence": None,
        "transport": client.transport,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "attempts": attempts,
        "usage": None,
        "error": f"{type(last_error).__name__}: {last_error}",
    }


async def preflight(
    clients: dict[str, SiLabOllamaDecisionClient],
    *,
    requested_models: list[str],
    sample_item: EvalItem,
    specs: dict[str, DatasetSpec],
    skip_model_check: bool,
) -> list[str]:
    visible_models: list[str] = []
    first_client = clients[requested_models[0]]
    if not skip_model_check:
        visible_models = await first_client.alist_models()
        missing = [model for model in requested_models if model not in visible_models]
        if missing:
            raise RuntimeError(
                "requested model IDs are not visible at the configured model-list endpoint: "
                f"{missing}. Visible IDs: {visible_models}"
            )

    for model in requested_models:
        answer_name = specs[sample_item.dataset].answer_name
        answers = await clients[model].adecide(
            sample_item.text,
            question_for(sample_item.dataset),
        )
        answer = answers[answer_name]
        if answer.choice not in specs[sample_item.dataset].labels:
            raise RuntimeError(
                f"preflight returned invalid label for {model}: {answer.choice!r}"
            )
        if answer.transport == "systemone" and answer.probabilities is None:
            raise RuntimeError(
                f"native System-One response for {model} has no probabilities"
            )
    return visible_models


async def run_evaluation(
    *,
    args: argparse.Namespace,
    items_by_dataset: dict[str, list[EvalItem]],
    specs: dict[str, DatasetSpec],
    predictions_path: Path,
    errors_path: Path,
) -> tuple[int, int, list[str]]:
    api_key = os.getenv("SI_LAB_API_KEY")

    clients = {
        model: SiLabOllamaDecisionClient(
            base_url=args.base_url,
            model=model,
            api_key=api_key or None,
            system_one_url=f"{args.ollama_base_url.rstrip('/')}/v1/systemone",
            transport=args.transport,
            timeout=args.timeout,
        )
        for model in args.models
    }

    try:
        visible_models: list[str] = []
        latest = load_latest_records(predictions_path) if args.resume else {}
        successful = {
            key for key, record in latest.items() if record.get("status") == "ok"
        }
        total_selected = sum(len(items_by_dataset[name]) for name in args.datasets)
        total_requested = total_selected * len(args.models)
        total_skipped = sum(
            1
            for model in args.models
            for dataset in args.datasets
            for item in items_by_dataset[dataset]
            if (model, dataset, item.record_id) in successful
        )
        total_pending = total_requested - total_skipped

        print(
            f"requested={total_requested} "
            f"resumed={total_skipped} pending={total_pending}",
            flush=True,
        )
        if total_pending == 0:
            return 0, total_skipped, visible_models

        sample_item = next(item for items in items_by_dataset.values() for item in items)
        visible_models = await preflight(
            clients, requested_models=args.models, sample_item=sample_item,
            specs=specs, skip_model_check=args.skip_model_check,
        )

        queue: asyncio.Queue[tuple[str, EvalItem] | None] = asyncio.Queue(
            maxsize=max(args.concurrency * 4, 8)
        )
        write_lock = asyncio.Lock()
        progress_lock = asyncio.Lock()
        counters = {"completed": 0, "errors": 0}

        async def producer() -> None:
            for model in args.models:
                for dataset in args.datasets:
                    for item in items_by_dataset[dataset]:
                        if (model, dataset, item.record_id) in successful:
                            continue
                        await queue.put((model, item))
            for _ in range(args.concurrency):
                await queue.put(None)

        async def worker() -> None:
            while True:
                work = await queue.get()
                try:
                    if work is None:
                        return
                    model, item = work
                    spec = specs[item.dataset]
                    record = await evaluate_item(
                        clients[model],
                        model=model,
                        item=item,
                        expected_labels=spec.labels,
                        answer_name=spec.answer_name,
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
                        if (
                            completed % args.progress_every == 0
                            or completed == total_pending
                        ):
                            print(
                                f"Progress {completed}/{total_pending} "
                                f"errors={counters['errors']}",
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

        return counters["errors"], total_skipped, visible_models
    finally:
        await asyncio.gather(
            *(client.aclose() for client in clients.values()),
            return_exceptions=True,
        )


def probability_rows(
    frame: pd.DataFrame,
    labels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, float]:
    row_indices: list[int] = []
    vectors: list[list[float]] = []
    max_sum_error = 0.0
    for position, probabilities in enumerate(frame["probabilities"]):
        if not isinstance(probabilities, dict):
            continue
        try:
            vector = np.asarray(
                [float(probabilities[label]) for label in labels], dtype=float
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(vector).all() or np.any(vector < 0):
            continue
        total = float(vector.sum())
        if total <= 0:
            continue
        max_sum_error = max(max_sum_error, abs(total - 1.0))
        vectors.append((vector / total).tolist())
        row_indices.append(position)
    return np.asarray(row_indices, dtype=int), np.asarray(vectors), max_sum_error


def calibration_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int,
) -> tuple[float, list[dict[str, t.Any]]]:
    confidence = probabilities.max(axis=1)
    correctness = (y_true == y_pred).astype(float)
    assignments = np.minimum((confidence * bins).astype(int), bins - 1)
    rows: list[dict[str, t.Any]] = []
    ece = 0.0
    for bin_index in range(bins):
        mask = assignments == bin_index
        count = int(mask.sum())
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        if count:
            mean_confidence = float(confidence[mask].mean())
            bin_accuracy = float(correctness[mask].mean())
            gap = abs(mean_confidence - bin_accuracy)
            ece += (count / len(y_true)) * gap
        else:
            mean_confidence = math.nan
            bin_accuracy = math.nan
            gap = math.nan
        rows.append(
            {
                "bin": bin_index + 1,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": bin_accuracy,
                "absolute_gap": gap,
            }
        )
    return ece, rows


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[str, ...],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | None]:
    if samples <= 0 or len(y_true) == 0:
        return {
            "accuracy_ci_low": None,
            "accuracy_ci_high": None,
            "macro_f1_ci_low": None,
            "macro_f1_ci_high": None,
        }

    rng = np.random.default_rng(seed)
    grouped = [np.flatnonzero(y_true == label) for label in labels]
    accuracy_values: list[float] = []
    f1_values: list[float] = []
    for _ in range(samples):
        sampled_groups = [
            rng.choice(indices, size=len(indices), replace=True)
            for indices in grouped
            if len(indices)
        ]
        indices = np.concatenate(sampled_groups)
        accuracy_values.append(float(accuracy_score(y_true[indices], y_pred[indices])))
        f1_values.append(
            float(
                f1_score(
                    y_true[indices],
                    y_pred[indices],
                    labels=list(labels),
                    average="macro",
                    zero_division=0,
                )
            )
        )
    return {
        "accuracy_ci_low": float(np.percentile(accuracy_values, 2.5)),
        "accuracy_ci_high": float(np.percentile(accuracy_values, 97.5)),
        "macro_f1_ci_low": float(np.percentile(f1_values, 2.5)),
        "macro_f1_ci_high": float(np.percentile(f1_values, 97.5)),
    }


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def generate_visualizations(
    output_dir: Path,
    specs: dict[str, DatasetSpec],
    models: list[str],
    datasets: list[str],
    predictions_path: Path,
) -> None:
    """Generate PNG charts from evaluation results."""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted")

    # -- 1. Confusion matrix heatmaps --
    confusion_dir = output_dir / "confusion_matrices"
    for model in models:
        for dataset in datasets:
            cm_path = confusion_dir / f"{safe_name(model)}__{dataset}.csv"
            if not cm_path.exists():
                continue
            cm_df = pd.read_csv(cm_path, index_col=0)
            fig, ax = plt.subplots(figsize=(max(4, len(cm_df.columns)), max(3, len(cm_df))))
            sns.heatmap(
                cm_df, annot=True, fmt="d", cmap="Blues",
                xticklabels=[c.replace("pred_", "") for c in cm_df.columns],
                yticklabels=[r.replace("gold_", "") for r in cm_df.index],
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Gold")
            ax.set_title(f"Confusion Matrix — {model} / {dataset}")
            fig.tight_layout()
            fig.savefig(viz_dir / f"confusion_{safe_name(model)}__{dataset}.png", dpi=150)
            plt.close(fig)

    # -- 2. Prediction distribution (gold vs predicted) --
    latest = load_latest_records(predictions_path)
    records = [
        r for key, r in latest.items()
        if key[0] in models and key[1] in datasets and r.get("status") == "ok"
    ]
    if records:
        pred_df = pd.DataFrame(records)
        for model in models:
            for dataset in datasets:
                group = pred_df[
                    (pred_df["model"] == model) & (pred_df["dataset"] == dataset)
                ]
                if group.empty:
                    continue
                spec = specs[dataset]
                labels = list(spec.labels)
                gold_counts = group["gold_label"].value_counts().reindex(labels, fill_value=0)
                pred_counts = group["predicted_label"].value_counts().reindex(labels, fill_value=0)
                x = np.arange(len(labels))
                width = 0.35
                fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.2), 4))
                ax.bar(x - width / 2, gold_counts.values, width, label="Gold")
                ax.bar(x + width / 2, pred_counts.values, width, label="Predicted")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right")
                ax.set_ylabel("Count")
                ax.set_title(f"Label Distribution — {model} / {dataset}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(
                    viz_dir / f"predictions_{safe_name(model)}__{dataset}.png", dpi=150,
                )
                plt.close(fig)

    # -- 3. Summary metric comparison across models --
    summary_path = output_dir / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        metrics = ["accuracy", "macro_f1", "weighted_f1"]
        for dataset in datasets:
            ds = summary[summary["dataset"] == dataset].copy()
            if ds.empty:
                continue
            available = [m for m in metrics if m in ds.columns and ds[m].notna().any()]
            if not available:
                continue
            x = np.arange(len(ds))
            width = 0.8 / max(len(available), 1)
            fig, ax = plt.subplots(figsize=(max(5, len(ds) * 1.5), 4))
            for i, metric in enumerate(available):
                ax.bar(x + i * width, ds[metric].values, width, label=metric)
            ax.set_xticks(x + width * (len(available) - 1) / 2)
            ax.set_xticklabels(ds["model"].values, rotation=30, ha="right")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.set_title(f"Model Comparison — {dataset}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(viz_dir / f"summary_{dataset}.png", dpi=150)
            plt.close(fig)

    print(f"Visualizations saved to {viz_dir.resolve()}", flush=True)


def summarize_predictions(
    predictions_path: Path,
    output_dir: Path,
    specs: dict[str, DatasetSpec],
    *,
    models: list[str],
    datasets: list[str],
    calibration_bins: int,
    bootstrap_samples: int,
    seed: int,
) -> None:
    latest = load_latest_records(predictions_path)
    records = [
        record
        for key, record in latest.items()
        if key[0] in models and key[1] in datasets
    ]
    if not records:
        print("No prediction records are available for summarization.", flush=True)
        return

    frame = pd.DataFrame(records)
    summary_rows: list[dict[str, t.Any]] = []
    per_class_rows: list[dict[str, t.Any]] = []
    calibration_rows: list[dict[str, t.Any]] = []
    coverage_rows: list[dict[str, t.Any]] = []
    confusion_dir = output_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        for dataset in datasets:
            spec = specs[dataset]
            group = frame[(frame["model"] == model) & (frame["dataset"] == dataset)]
            if group.empty:
                continue
            ok = group[group["status"] == "ok"].reset_index(drop=True)
            n_total = int(len(group))
            n_ok = int(len(ok))
            base: dict[str, t.Any] = {
                "model": model,
                "dataset": dataset,
                "n_total": n_total,
                "n_success": n_ok,
                "n_error": n_total - n_ok,
                "error_rate": (n_total - n_ok) / n_total,
                "latency_mean_ms": float(group["latency_ms"].mean()),
                "latency_p50_ms": float(group["latency_ms"].quantile(0.50)),
                "latency_p95_ms": float(group["latency_ms"].quantile(0.95)),
            }
            if not n_ok:
                summary_rows.append(base)
                continue

            y_true = ok["gold_label"].astype(str).to_numpy()
            y_pred = ok["predicted_label"].astype(str).to_numpy()
            base.update(
                {
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y_true, y_pred)
                    ),
                    "macro_f1": float(
                        f1_score(
                            y_true,
                            y_pred,
                            labels=list(spec.labels),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "weighted_f1": float(
                        f1_score(
                            y_true,
                            y_pred,
                            labels=list(spec.labels),
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                }
            )
            base.update(
                stratified_bootstrap_ci(
                    y_true,
                    y_pred,
                    spec.labels,
                    samples=bootstrap_samples,
                    seed=seed,
                )
            )

            report = classification_report(
                y_true,
                y_pred,
                labels=list(spec.labels),
                output_dict=True,
                zero_division=0,
            )
            for label in spec.labels:
                row = t.cast(dict[str, float], report[label])
                per_class_rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "label": label,
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1": row["f1-score"],
                        "support": int(row["support"]),
                    }
                )

            matrix = confusion_matrix(y_true, y_pred, labels=list(spec.labels))
            pd.DataFrame(
                matrix,
                index=[f"gold_{label}" for label in spec.labels],
                columns=[f"pred_{label}" for label in spec.labels],
            ).to_csv(
                confusion_dir / f"{safe_name(model)}__{dataset}.csv",
                index=True,
            )

            probability_indices, probabilities, max_sum_error = probability_rows(
                ok, spec.labels
            )
            base["n_probability"] = int(len(probability_indices))
            base["probability_coverage"] = len(probability_indices) / n_ok
            base["probability_max_sum_error"] = max_sum_error
            if len(probability_indices):
                p_true = y_true[probability_indices]
                p_pred = y_pred[probability_indices]
                log_loss_labels = tuple(sorted(spec.labels))
                log_loss_columns = [
                    spec.labels.index(label) for label in log_loss_labels
                ]
                base["log_loss"] = float(
                    log_loss(
                        p_true,
                        probabilities[:, log_loss_columns],
                        labels=list(log_loss_labels),
                    )
                )
                encoded = np.asarray(
                    [[1.0 if truth == label else 0.0 for label in spec.labels] for truth in p_true]
                )
                base["multiclass_brier"] = float(
                    np.mean(np.sum((probabilities - encoded) ** 2, axis=1))
                )
                ece, bins = calibration_table(
                    p_true,
                    p_pred,
                    probabilities,
                    bins=calibration_bins,
                )
                base["ece"] = ece
                for row in bins:
                    calibration_rows.append(
                        {"model": model, "dataset": dataset, **row}
                    )

                confidence = probabilities.max(axis=1)
                correctness = p_true == p_pred
                probability_choice = np.asarray(spec.labels)[
                    probabilities.argmax(axis=1)
                ]
                base["probability_choice_mismatches"] = int(
                    (probability_choice != p_pred).sum()
                )
                order = np.argsort(-confidence)
                for target_coverage in (1.0, 0.9, 0.75, 0.5):
                    count = max(1, int(math.ceil(len(order) * target_coverage)))
                    selected = order[:count]
                    coverage_rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "target_coverage": target_coverage,
                            "n": count,
                            "accuracy": float(correctness[selected].mean()),
                            "minimum_confidence": float(confidence[selected].min()),
                        }
                    )
            else:
                base.update(
                    {
                        "log_loss": None,
                        "multiclass_brier": None,
                        "ece": None,
                    }
                )
            summary_rows.append(base)

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(
        per_class_rows,
        columns=[
            "model",
            "dataset",
            "label",
            "precision",
            "recall",
            "f1",
            "support",
        ],
    ).to_csv(
        output_dir / "per_class_metrics.csv", index=False
    )
    pd.DataFrame(
        calibration_rows,
        columns=[
            "model",
            "dataset",
            "bin",
            "lower",
            "upper",
            "count",
            "mean_confidence",
            "accuracy",
            "absolute_gap",
        ],
    ).to_csv(
        output_dir / "calibration_bins.csv", index=False
    )
    pd.DataFrame(
        coverage_rows,
        columns=[
            "model",
            "dataset",
            "target_coverage",
            "n",
            "accuracy",
            "minimum_confidence",
        ],
    ).to_csv(
        output_dir / "selective_coverage.csv", index=False
    )

    pairwise_rows: list[dict[str, t.Any]] = []
    ok_frame = frame[frame["status"] == "ok"].copy()
    for dataset in datasets:
        dataset_frame = ok_frame[ok_frame["dataset"] == dataset]
        for model_a, model_b in itertools.combinations(models, 2):
            left = dataset_frame[dataset_frame["model"] == model_a][
                ["id", "gold_label", "predicted_label"]
            ].rename(columns={"predicted_label": "pred_a"})
            right = dataset_frame[dataset_frame["model"] == model_b][
                ["id", "gold_label", "predicted_label"]
            ].rename(columns={"predicted_label": "pred_b"})
            paired = left.merge(
                right,
                on=["id", "gold_label"],
                how="inner",
                validate="one_to_one",
            )
            if paired.empty:
                continue
            correct_a = paired["pred_a"] == paired["gold_label"]
            correct_b = paired["pred_b"] == paired["gold_label"]
            a_only = int((correct_a & ~correct_b).sum())
            b_only = int((~correct_a & correct_b).sum())
            discordant = a_only + b_only
            if discordant:
                chi_square = (max(abs(a_only - b_only) - 1, 0) ** 2) / discordant
                p_value = math.erfc(math.sqrt(chi_square / 2.0))
            else:
                chi_square = 0.0
                p_value = 1.0
            pairwise_rows.append(
                {
                    "dataset": dataset,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_paired": int(len(paired)),
                    "model_a_correct_model_b_wrong": a_only,
                    "model_a_wrong_model_b_correct": b_only,
                    "mcnemar_chi_square_cc": chi_square,
                    "mcnemar_p_value_asymptotic": p_value,
                }
            )
    pd.DataFrame(
        pairwise_rows,
        columns=[
            "dataset",
            "model_a",
            "model_b",
            "n_paired",
            "model_a_correct_model_b_wrong",
            "model_a_wrong_model_b_correct",
            "mcnemar_chi_square_cc",
            "mcnemar_p_value_asymptotic",
        ],
    ).to_csv(
        output_dir / "pairwise_comparisons.csv", index=False
    )

    write_json(
        output_dir / "metrics_metadata.json",
        {
            "generated_at": utc_now(),
            "primary_metric": "macro_f1",
            "multiclass_brier_definition": (
                "mean over examples of the sum of squared differences between "
                "class probabilities and one-hot outcomes"
            ),
            "ece_bins": calibration_bins,
            "bootstrap": {
                "samples": bootstrap_samples,
                "method": "stratified percentile bootstrap",
                "confidence_level": 0.95,
                "seed": seed,
            },
            "pairwise_test": "continuity-corrected asymptotic McNemar test",
            "probability_note": (
                "Probability metrics are computed only for rows with a complete, "
                "finite native probability vector."
            ),
        },
    )

    generate_visualizations(
        output_dir, specs, models, datasets, predictions_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11435",
        help="Open WebUI OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=os.getenv("SI_LAB_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        help=(
            "Direct/tunneled Ollama base URL used by System-One "
            f"(default: {DEFAULT_OLLAMA_BASE_URL}; env: SI_LAB_OLLAMA_BASE_URL)."
        ),
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    _available = available_datasets()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=_available,
        help=f"Dataset names from registry. Available: {_available}",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="Deterministic stratified sample size; omit for the full datasets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument(
        "--transport",
        choices=["systemone", "chat"],
        default="systemone",
        help="Use systemone for probability-aware evaluation; chat is label-only.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-appended-data", action="store_true",
        help="Allow a changed dataset hash only after verifying every cached ID, text, and label.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild metrics from an existing predictions.jsonl without API calls.",
    )
    parser.add_argument(
        "--deduplicate-text",
        action="store_true",
        help=(
            "Sensitivity mode: exclude conflicting duplicate texts and retain "
            "one row from each remaining normalized-text group."
        ),
    )
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory; defaults to results/run_<UTC timestamp>.",
    )
    args = parser.parse_args(argv)

    if args.samples_per_class is not None and args.samples_per_class <= 0:
        parser.error("--samples-per-class must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.retries < 0:
        parser.error("--retries must not be negative")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.calibration_bins <= 1:
        parser.error("--calibration-bins must be greater than 1")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must not be negative")
    if len(set(args.models)) != len(args.models):
        parser.error("--models contains duplicates")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets contains duplicates")
    if args.dry_run and args.summarize_only:
        parser.error("--dry-run and --summarize-only cannot be used together")
    if args.allow_appended_data and not (args.resume or args.summarize_only):
        parser.error("--allow-appended-data requires --resume or --summarize-only")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.output = Path("results") / f"run_{stamp}"
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    errors_path = output_dir / "errors.jsonl"

    if (
        predictions_path.exists()
        and predictions_path.stat().st_size
        and not args.resume
        and not args.summarize_only
    ):
        raise RuntimeError(
            f"{predictions_path} already contains results; use --resume or a new --output"
        )

    specs = get_specs(args.datasets)

    items_by_dataset: dict[str, list[EvalItem]] = {}
    data_quality: dict[str, t.Any] = {}
    for dataset in args.datasets:
        items, quality = load_dataset(
            specs[dataset],
            samples_per_class=args.samples_per_class,
            seed=args.seed,
            deduplicate_text=args.deduplicate_text,
        )
        items_by_dataset[dataset] = items
        data_quality[dataset] = quality
    validated = set()
    if args.resume or args.summarize_only:
        validated = validate_cached_inputs(predictions_path, items_by_dataset, args.models)

    comparable_config = {
        "script_version": SCRIPT_VERSION,
        "base_url": args.base_url,
        "ollama_base_url": (
            args.ollama_base_url if args.transport == "systemone" else None
        ),
        "models": args.models,
        "datasets": args.datasets,
        "transport": args.transport,
        "samples_per_class": args.samples_per_class,
        "deduplicate_text": args.deduplicate_text,
        "seed": args.seed,
        "dataset_sha256": {
            dataset: data_quality[dataset]["sha256"] for dataset in args.datasets
        },
    }

    total_rows = sum(len(items_by_dataset[name]) for name in args.datasets)
    total_requests = total_rows * len(args.models)
    print(
        f"Prepared {total_rows} rows and {total_requests} model requests in {output_dir}",
        flush=True,
    )
    if args.summarize_only:
        if not predictions_path.exists() or not predictions_path.stat().st_size:
            raise RuntimeError(
                f"--summarize-only requires an existing non-empty {predictions_path}"
            )
        existing_config_path = output_dir / "run_config.json"
        if existing_config_path.exists():
            existing_config = json.loads(
                existing_config_path.read_text(encoding="utf-8")
            )
            _check_resume_compat(
                existing_config.get("comparable_config", {}),
                comparable_config,
                context="summarize",
                validated_datasets=validated if args.allow_appended_data else None,
            )
        write_json(output_dir / "data_quality.json", data_quality)
        summarize_predictions(
            predictions_path,
            output_dir,
            specs,
            models=args.models,
            datasets=args.datasets,
            calibration_bins=args.calibration_bins,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        print(f"Metrics rebuilt in {output_dir.resolve()}", flush=True)
        return 0
    config_path = output_dir / "run_config.json"
    if args.resume and config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        _check_resume_compat(
            existing_config.get("comparable_config", {}),
            comparable_config,
            context="resume",
            validated_datasets=validated if args.allow_appended_data else None,
        )
        run_id = existing_config["run_id"]
        created_at = existing_config["created_at"]
    else:
        run_id = hashlib.sha256(
            f"{utc_now()}:{os.getpid()}".encode("utf-8")
        ).hexdigest()[:16]
        created_at = utc_now()

    run_config = {
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": utc_now(),
        "comparable_config": comparable_config,
        "runtime": {
            "concurrency": args.concurrency,
            "timeout": args.timeout,
            "retries": args.retries,
            "retry_base_seconds": args.retry_base_seconds,
            "calibration_bins": args.calibration_bins,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "selected_rows": {
            dataset: len(items_by_dataset[dataset]) for dataset in args.datasets
        },
        "credential_source": "SI_LAB_API_KEY environment variable",
    }
    write_json(output_dir / "data_quality.json", data_quality)
    write_json(config_path, run_config)

    if args.dry_run:
        print("Dry run complete; no API requests were sent.", flush=True)
        return 0

    errors, resumed, visible_models = asyncio.run(
        run_evaluation(
            args=args,
            items_by_dataset=items_by_dataset,
            specs=specs,
            predictions_path=predictions_path,
            errors_path=errors_path,
        )
    )
    run_config["updated_at"] = utc_now()
    run_config["visible_models"] = visible_models
    run_config["resumed_successes"] = resumed
    run_config["final_errors"] = errors
    write_json(config_path, run_config)

    summarize_predictions(
        predictions_path,
        output_dir,
        specs,
        models=args.models,
        datasets=args.datasets,
        calibration_bins=args.calibration_bins,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(f"Evaluation complete. Results: {output_dir.resolve()}", flush=True)
    return 2 if errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Re-run with --resume to continue.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
