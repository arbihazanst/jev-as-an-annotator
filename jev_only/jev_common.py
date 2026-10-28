"""Shared utilities for Jev evaluation (backward compatibility wrapper for openrouter_only)."""
from __future__ import annotations

from openrouter_only.openrouter_common import (
    DECISIONS_API_URL,
    DATA_DIR,
    RESULTS_DIR,
    ROOT_DIR,
    build_record,
    call_openrouter as call_jev,
    default_specs,
    file_sha256,
    get_api_key,
    question_for,
    run_manifest,
    validate_openrouter_response as validate_jev_response,
    write_json,
)

DEFAULT_MODEL = "typesafe/jev-1.13"

__all__ = [
    "DECISIONS_API_URL",
    "DEFAULT_MODEL",
    "DATA_DIR",
    "RESULTS_DIR",
    "ROOT_DIR",
    "build_record",
    "call_jev",
    "default_specs",
    "file_sha256",
    "get_api_key",
    "question_for",
    "run_manifest",
    "validate_jev_response",
    "write_json",
]
