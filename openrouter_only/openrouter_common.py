"""Shared utilities for evaluating decision models through OpenRouter Decisions API."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import typing as t

import httpx

from dataset_registry import DatasetSpec, get_specs, question_for
from evaluate_decision_models import EvalItem, utc_now

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_MODEL = os.getenv("OPENROUTER_DECISION_MODEL", "openai/gpt-6-luna-decisions")
DECISIONS_API_URL = os.getenv("OPENROUTER_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions")


def get_api_key(cli_key: str | None = None) -> str:
    if cli_key:
        return cli_key
    if env_key := os.getenv("OPENROUTER_API_KEY"):
        return env_key
    env_file = ROOT_DIR / ".env"
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip('"').strip("'")
    return ""


def default_specs() -> dict[str, DatasetSpec]:
    return get_specs()


def write_json(path: Path, payload: t.Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


async def call_openrouter(
    client: httpx.AsyncClient,
    model: str,
    text: str,
    questions: dict[str, t.Any],
) -> dict[str, t.Any]:
    response = await client.post(
        DECISIONS_API_URL,
        json={"model": model, "state": text, "questions": questions},
    )
    response.raise_for_status()
    return t.cast(dict[str, t.Any], response.json())


# Backwards compatibility alias
call_jev = call_openrouter


def validate_openrouter_response(
    data: t.Any,
    spec: DatasetSpec,
) -> tuple[str, dict[str, float], float | None, dict[str, t.Any] | None]:
    """Validate one OpenRouter decision response without changing its scientific content."""
    if not isinstance(data, dict):
        raise ValueError("OpenRouter decision response is not a JSON object")
    answers = data.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("OpenRouter decision response has no answers object")
    answer = answers.get(spec.answer_name)
    if answer is None and len(answers) == 1:
        answer = next(iter(answers.values()))
    if not isinstance(answer, dict):
        raise ValueError(f"OpenRouter decision response is missing answer {spec.answer_name!r}")

    choice = answer.get("choice")
    if choice is None:
        raise ValueError("OpenRouter decision choice is missing")
    choice = str(choice).strip()

    # Match choice against spec.labels (exact first, case-insensitive fallback)
    matched_choice: str | None = None
    for label in spec.labels:
        if choice == label:
            matched_choice = label
            break
    if matched_choice is None:
        for label in spec.labels:
            if choice.lower() == label.lower():
                matched_choice = label
                break
    if matched_choice is None:
        raise ValueError(f"invalid OpenRouter decision label {choice!r}; expected one of {spec.labels}")
    choice = matched_choice

    raw_probs = answer.get("probabilities")
    if not isinstance(raw_probs, dict):
        raise ValueError("OpenRouter decision response has no probability vector")

    norm_probs = {k.lower(): v for k, v in raw_probs.items()}
    missing = [
        label
        for label in spec.labels
        if label not in raw_probs and label.lower() not in norm_probs
    ]
    if missing:
        raise ValueError(f"OpenRouter decision probability vector is missing labels: {missing}")

    probabilities: dict[str, float] = {}
    for label in spec.labels:
        raw_val = raw_probs[label] if label in raw_probs else norm_probs[label.lower()]
        value = float(raw_val)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid probability for {label}: {value}")
        probabilities[label] = value
    if sum(probabilities.values()) <= 0:
        raise ValueError("OpenRouter decision probability vector sums to zero")

    confidence_raw = answer.get("confidence")
    confidence = None if confidence_raw is None else float(confidence_raw)
    if confidence is not None and not math.isfinite(confidence):
        raise ValueError(f"invalid confidence value: {confidence}")

    usage = data.get("usage")
    parsed_usage = dict(usage) if isinstance(usage, dict) else None
    return choice, probabilities, confidence, parsed_usage


# Backwards compatibility alias
validate_jev_response = validate_openrouter_response


def build_record(
    *,
    model: str,
    item: EvalItem,
    status: str,
    latency_ms: float,
    attempts: int,
    predicted_label: str | None,
    probabilities: dict[str, float] | None,
    confidence: float | None,
    usage: dict[str, t.Any] | None,
    error: str | None,
) -> dict[str, t.Any]:
    return {
        "timestamp": utc_now(),
        "status": status,
        "model": model,
        "dataset": item.dataset,
        "id": item.record_id,
        "text_sha256": hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
        "gold_label": item.gold_label,
        "predicted_label": predicted_label,
        "probabilities": probabilities,
        "systemone_confidence": confidence,
        "transport": "openrouter",
        "latency_ms": round(latency_ms, 3),
        "attempts": attempts,
        "usage": usage,
        "error": error,
    }


def run_manifest(
    *,
    mode: str,
    model: str,
    datasets: list[str],
    seed: int,
    selected_rows: dict[str, int],
    specs: dict[str, DatasetSpec],
    extra: dict[str, t.Any] | None = None,
) -> dict[str, t.Any]:
    payload: dict[str, t.Any] = {
        "created_at": utc_now(),
        "mode": mode,
        "model": model,
        "datasets": datasets,
        "seed": seed,
        "selected_rows": selected_rows,
        "dataset_sha256": {name: file_sha256(specs[name].path) for name in datasets},
        "endpoint": DECISIONS_API_URL,
        "credential_source": "OPENROUTER_API_KEY environment variable or .env",
    }
    if extra:
        payload.update(extra)
    return payload
