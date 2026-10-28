"""Offline tests for the resumable decision-model evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from evaluate_decision_models import (
    DEFAULT_OLLAMA_BASE_URL,
    DatasetSpec,
    append_jsonl,
    load_dataset,
    load_latest_records,
    parse_args,
    summarize_predictions,
)


class DatasetLoadingTests(unittest.TestCase):
    def test_stratified_sample_and_quality_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            pd.DataFrame(
                [
                    {"id": 1, "text": "teks sama", "sentiment": "positive"},
                    {"id": 2, "text": "teks sama", "sentiment": "negative"},
                    {"id": 3, "text": "biasa", "sentiment": "neutral"},
                    {"id": 4, "text": "positif", "sentiment": "positive"},
                ]
            ).to_csv(path, index=False)
            spec = DatasetSpec(
                name="sentiment",
                path=path,
                text_column="text",
                label_column="sentiment",
                labels=("positive", "negative", "neutral"),
                answer_name="sentiment",
            )

            items, quality = load_dataset(spec, samples_per_class=1, seed=42)

            self.assertEqual(len(items), 3)
            self.assertEqual({item.gold_label for item in items}, set(spec.labels))
            self.assertEqual(quality["duplicate_text_rows"], 2)
            self.assertEqual(quality["conflicting_duplicate_text_groups"], 1)

            deduplicated, dedup_quality = load_dataset(
                spec,
                samples_per_class=None,
                seed=42,
                deduplicate_text=True,
            )
            self.assertEqual(len(deduplicated), 2)
            self.assertEqual(dedup_quality["deduplication_removed_rows"], 2)

    def test_auto_label_detection_and_dynamic_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emotions_custom.csv"
            pd.DataFrame(
                [
                    {"id": 1, "tweet": "senang", "label": "happy"},
                    {"id": 2, "tweet": "kaget", "label": "surprise"},
                    {"id": 3, "tweet": "marah", "label": "anger"},
                ]
            ).to_csv(path, index=False)
            spec = DatasetSpec(
                name="emotions_custom",
                path=path,
                text_column="tweet",
                label_column="label"
            )
            detected = spec.resolve_labels()
            self.assertEqual(detected, ("anger", "happy", "surprise"))

            items, quality = load_dataset(spec, samples_per_class=None, seed=42)
            self.assertEqual(len(items), 3)
            self.assertEqual(quality["unexpected_labels"], [])

            from dataset_registry import _emotion_question_id
            q = _emotion_question_id(detected)
            self.assertEqual(
                set(q["emotion"]["criteria"].keys()),
                {"anger", "happy", "surprise"},
            )
            self.assertEqual(q["emotion"]["criteria"]["surprise"], "Terkejut, kaget, heran, atau takjub.")


class CliTests(unittest.TestCase):
    def test_tunnel_is_default_system_one_base_url(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            args = parse_args([])

        self.assertEqual(args.ollama_base_url, DEFAULT_OLLAMA_BASE_URL)

    def test_tunnel_base_url_can_come_from_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {"SI_LAB_OLLAMA_BASE_URL": "http://127.0.0.1:13000"},
            clear=True,
        ):
            args = parse_args([])

        self.assertEqual(args.ollama_base_url, "http://127.0.0.1:13000")

    def test_tunnel_base_url_can_be_overridden(self) -> None:
        args = parse_args(["--ollama-base-url", "http://127.0.0.1:12000"])

        self.assertEqual(args.ollama_base_url, "http://127.0.0.1:12000")


class MetricsTests(unittest.TestCase):
    def _records(self) -> list[dict[str, object]]:
        labels = ["positive", "negative", "neutral"]
        gold = ["positive", "positive", "negative", "negative", "neutral", "neutral"]
        predictions = {
            "model-a": gold,
            "model-b": ["positive", "negative", "negative", "neutral", "neutral", "positive"],
        }
        records: list[dict[str, object]] = []
        for model, predicted in predictions.items():
            for index, (truth, guess) in enumerate(zip(gold, predicted), start=1):
                probabilities = {
                    label: (0.8 if label == guess else 0.1) for label in labels
                }
                records.append(
                    {
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "status": "ok",
                        "model": model,
                        "dataset": "sentiment",
                        "id": str(index),
                        "gold_label": truth,
                        "predicted_label": guess,
                        "probabilities": probabilities,
                        "latency_ms": 10.0 + index,
                    }
                )
        return records

    def test_summary_and_pairwise_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            predictions_path = output / "predictions.jsonl"
            for record in self._records():
                append_jsonl(predictions_path, record)

            specs = {
                "sentiment": DatasetSpec(
                    name="sentiment",
                    path=output / "unused.csv",
                    text_column="text",
                    label_column="sentiment",
                    labels=("positive", "negative", "neutral"),
                    answer_name="sentiment",
                )
            }
            summarize_predictions(
                predictions_path,
                output,
                specs,
                models=["model-a", "model-b"],
                datasets=["sentiment"],
                calibration_bins=5,
                bootstrap_samples=20,
                seed=7,
            )

            summary = pd.read_csv(output / "summary.csv")
            model_a = summary[summary["model"] == "model-a"].iloc[0]
            model_b = summary[summary["model"] == "model-b"].iloc[0]
            self.assertEqual(model_a["accuracy"], 1.0)
            self.assertEqual(model_a["macro_f1"], 1.0)
            self.assertAlmostEqual(model_b["accuracy"], 0.5)
            self.assertEqual(model_a["probability_coverage"], 1.0)
            self.assertTrue((output / "pairwise_comparisons.csv").exists())
            self.assertTrue(
                (output / "confusion_matrices" / "model-a__sentiment.csv").exists()
            )

    def test_latest_checkpoint_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            base = {
                "model": "model-a",
                "dataset": "sentiment",
                "id": "1",
                "gold_label": "positive",
                "predicted_label": None,
                "status": "error",
            }
            append_jsonl(path, base)
            append_jsonl(path, {**base, "status": "ok", "predicted_label": "positive"})

            latest = load_latest_records(path)

            self.assertEqual(len(latest), 1)
            self.assertEqual(latest[("model-a", "sentiment", "1")]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
