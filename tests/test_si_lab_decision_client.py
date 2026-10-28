"""Offline tests for :mod:`si_lab_decision_client`."""

from __future__ import annotations

import unittest

import httpx

from si_lab_decision_client import (
    SiLabOllamaDecisionClient,
    emotion_questions,
    sentiment_question,
)


class SiLabDecisionClientTests(unittest.TestCase):
    def test_native_choice_and_model_list(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/models":
                return httpx.Response(
                    200,
                    json={"data": [{"id": "nimble:9b"}, {"id": "tev1:4b"}]},
                )
            self.assertEqual(str(request.url), "http://127.0.0.1:11435/v1/systemone")
            return httpx.Response(
                200,
                json={
                    "answers": {
                        "sentiment": {
                            "type": "choice",
                            "choice": "positive",
                            "probabilities": {
                                "positive": 0.8,
                                "negative": 0.1,
                                "neutral": 0.1,
                            },
                            "confidence": 0.6,
                        }
                    }
                },
            )

        transport = httpx.MockTransport(handler)
        with SiLabOllamaDecisionClient(
            "https://llms.si-lab.org/api",
            "nimble:9b",
            "test-key",
            system_one_url="http://127.0.0.1:11435/v1/systemone",
            http_transport=transport,
        ) as client:
            self.assertEqual(client.list_models(), ["nimble:9b", "tev1:4b"])
            answer = client.decide("bagus sekali", sentiment_question())["sentiment"]

        self.assertEqual(answer.choice, "positive")
        self.assertEqual(answer.probabilities["positive"], 0.8)
        self.assertEqual(answer.transport, "systemone")

    def test_chat_fallback_is_explicit(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if request.url.path == "/v1/systemone":
                return httpx.Response(404, json={"detail": "not found"})
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "A"}}
                    ]
                },
            )

        transport = httpx.MockTransport(handler)
        with SiLabOllamaDecisionClient(
            "https://llms.si-lab.org/api",
            "tev1:4b",
            "test-key",
            transport="auto",
            allow_chat_fallback=True,
            http_transport=transport,
        ) as client:
            answer = client.decide("bagus", sentiment_question())["sentiment"]

        self.assertEqual(
            calls, ["/v1/systemone", "/api/chat/completions"]
        )
        self.assertEqual(answer.choice, "positive")
        self.assertIsNone(answer.probabilities)
        self.assertEqual(answer.transport, "chat")

    def test_native_multilabel_nouls(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            questions = emotion_questions()
            answers = {
                name: {"type": "noul", "noul": 0.9 if name == "happy" else 0.1}
                for name in questions
            }
            return httpx.Response(200, json={"answers": answers})

        transport = httpx.MockTransport(handler)
        with SiLabOllamaDecisionClient(
            "https://llms.si-lab.org/api",
            "nimble:9b",
            "test-key",
            http_transport=transport,
        ) as client:
            answers = client.decide("senang sekali", emotion_questions())

        selected = {name for name, answer in answers.items() if answer.choice == "true"}
        self.assertEqual(selected, {"happy"})


class AsyncSiLabDecisionClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_native_choice(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "answers": {
                        "sentiment": {
                            "type": "choice",
                            "choice": "neutral",
                            "probabilities": {
                                "positive": 0.1,
                                "negative": 0.2,
                                "neutral": 0.7,
                            },
                        }
                    }
                },
            )

        transport = httpx.MockTransport(handler)
        async with SiLabOllamaDecisionClient(
            "https://llms.si-lab.org/api",
            "tev1:0.8b",
            "test-key",
            async_http_transport=transport,
        ) as client:
            answers = await client.adecide("informasi biasa", sentiment_question())

        self.assertEqual(answers["sentiment"].choice, "neutral")


if __name__ == "__main__":
    unittest.main()
