from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from compare_all_models import main as compare_main
from jev_only.jev_common import default_specs, validate_jev_response
from jev_only.run_jev_full import parse_args as parse_full_args, run_full
from jev_only.run_jev_smoke import parse_args as parse_smoke_args, run_smoke
from openrouter_only.openrouter_common import validate_openrouter_response
from openrouter_only.run_openrouter_full import (
    parse_args as parse_openrouter_full_args,
    run_full as run_openrouter_full,
)
from openrouter_only.run_openrouter_smoke import (
    parse_args as parse_openrouter_smoke_args,
    run_smoke as run_openrouter_smoke,
)


class JevValidationTests(unittest.TestCase):
    def test_valid_choice_response(self) -> None:
        spec = default_specs()["sentiment"]
        response = {
            "answers": {
                "sentiment": {
                    "choice": "positive",
                    "probabilities": {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
                    "confidence": 0.8,
                }
            },
            "usage": {"input_tokens": 100},
        }
        choice, probabilities, confidence, usage = validate_jev_response(response, spec)
        self.assertEqual(choice, "positive")
        self.assertEqual(set(probabilities), set(spec.labels))
        self.assertEqual(confidence, 0.8)
        self.assertEqual(usage["input_tokens"], 100)

    def test_invalid_choice_is_rejected(self) -> None:
        spec = default_specs()["sentiment"]
        response = {
            "answers": {
                "sentiment": {
                    "choice": "mixed",
                    "probabilities": {"positive": 0.4, "negative": 0.3, "neutral": 0.3},
                }
            }
        }
        with self.assertRaises(ValueError):
            validate_jev_response(response, spec)

    def test_incomplete_probability_vector_is_rejected(self) -> None:
        spec = default_specs()["emotion"]
        response = {
            "answers": {
                "emotion": {
                    "choice": "happy",
                    "probabilities": {"happy": 1.0},
                }
            }
        }
        with self.assertRaises(ValueError):
            validate_jev_response(response, spec)


class DryRunIsolationTests(unittest.TestCase):
    def test_smoke_dry_run_writes_no_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = parse_smoke_args([
                "--dry-run",
                "--samples-per-class", "1",
                "--output", str(output),
                "--bootstrap-samples", "5",
            ])
            result = asyncio.run(run_smoke(args))
            self.assertEqual(result, 0)
            self.assertTrue((output / "run_config.json").exists())
            self.assertTrue((output / "payload_preview.json").exists())
            self.assertFalse((output / "predictions.jsonl").exists())

    def test_full_dry_run_writes_no_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = parse_full_args([
                "--dry-run",
                "--limit", "2",
                "--output", str(output),
                "--bootstrap-samples", "5",
            ])
            result = asyncio.run(run_full(args))
            self.assertEqual(result, 0)
            self.assertTrue((output / "run_config.json").exists())
            self.assertTrue((output / "payload_preview.json").exists())
            self.assertFalse((output / "predictions.jsonl").exists())

    def test_openrouter_dry_run_with_gpt_6_luna(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            custom_model = "openai/gpt-6-luna-decisions"
            args = parse_openrouter_full_args([
                "--model", custom_model,
                "--dry-run",
                "--limit", "2",
                "--output", str(output),
                "--bootstrap-samples", "5",
            ])
            result = asyncio.run(run_openrouter_full(args))
            self.assertEqual(result, 0)
            self.assertTrue((output / "run_config.json").exists())
            config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["model"], custom_model)
            preview = json.loads((output / "payload_preview.json").read_text(encoding="utf-8"))
            self.assertTrue(all(p["model"] == custom_model for p in preview))
            self.assertFalse((output / "predictions.jsonl").exists())



class AggregationTests(unittest.TestCase):
    def _record(self, model: str, dataset: str, row: pd.Series, text_col: str, label_col: str, predicted: str, probabilities: dict[str, float]) -> dict:
        text = str(row[text_col])
        return {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "status": "ok",
            "model": model,
            "dataset": dataset,
            "id": str(int(row["id"])) if float(row["id"]).is_integer() else str(row["id"]),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "gold_label": str(row[label_col]),
            "predicted_label": predicted,
            "probabilities": probabilities,
            "systemone_confidence": max(probabilities.values()),
            "transport": "systemone",
            "latency_ms": 10.0,
            "attempts": 1,
            "usage": None,
            "error": None,
        }

    def test_two_model_aggregation_and_pairwise_output(self) -> None:
        specs = default_specs()
        sentiment = pd.read_csv(specs["sentiment"].path).iloc[0]
        emotion = pd.read_csv(specs["emotion"].path).iloc[0]
        sent_gold = str(sentiment["sentiment"])
        emo_gold = str(emotion["label"])
        sent_probs = {label: (0.8 if label == sent_gold else 0.1) for label in specs["sentiment"].labels}
        # five classes: keep a valid distribution summing to 1
        emo_probs = {label: (0.8 if label == emo_gold else 0.05) for label in specs["emotion"].labels}

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            p1, p2 = base / "a.jsonl", base / "b.jsonl"
            records_a = [
                self._record("model-a", "sentiment", sentiment, "text", "sentiment", sent_gold, sent_probs),
                self._record("model-a", "emotion", emotion, "tweet", "label", emo_gold, emo_probs),
            ]
            records_b = [
                self._record("model-b", "sentiment", sentiment, "text", "sentiment", sent_gold, sent_probs),
                self._record("model-b", "emotion", emotion, "tweet", "label", emo_gold, emo_probs),
            ]
            for path, records in [(p1, records_a), (p2, records_b)]:
                path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

            output = base / "comparison"
            result = compare_main([
                "--inputs", str(p1), str(p2),
                "--output", str(output),
                "--bootstrap-samples", "5",
            ])
            self.assertEqual(result, 0)
            self.assertTrue((output / "summary.csv").exists())
            self.assertTrue((output / "pairwise_comparisons.csv").exists())
            self.assertTrue((output / "comparison_manifest.json").exists())
            pairwise = pd.read_csv(output / "pairwise_comparisons.csv")
            self.assertEqual(len(pairwise), 2)  # one pair for each dataset


if __name__ == "__main__":
    unittest.main()
