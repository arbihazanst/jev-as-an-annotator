"""Minimal SI-Lab decision-model smoke test.

PowerShell:
    $env:SI_LAB_API_KEY = Read-Host "SI-Lab API key"
    python example_si_lab_decision.py

The key is read from the process environment and is never written to disk.
"""

from __future__ import annotations

import os

from si_lab_decision_client import (
    DEFAULT_OLLAMA_BASE_URL,
    SiLabOllamaDecisionClient,
    sentiment_question,
)


API_BASE_URL = "https://llms.si-lab.org/api"
OLLAMA_BASE_URL = os.getenv("SI_LAB_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
MODEL = os.getenv("SI_LAB_MODEL", "nimble:9b")


def main() -> None:
    api_key = os.environ["SI_LAB_API_KEY"]

    with SiLabOllamaDecisionClient(
        base_url=API_BASE_URL,
        model=MODEL,
        api_key=api_key,
        system_one_url=f"{OLLAMA_BASE_URL.rstrip('/')}/v1/systemone",
        transport="systemone",
        timeout=600.0,
    ) as client:
        print("Visible models:", client.list_models())
        answer = client.decide(
            "Pelayanannya cepat dan saya sangat puas.",
            sentiment_question(),
        )["sentiment"]

    print("Choice:", answer.choice)
    print("Probabilities:", answer.probabilities)
    print("Confidence:", answer.confidence)
    print("Transport:", answer.transport)


if __name__ == "__main__":
    main()
