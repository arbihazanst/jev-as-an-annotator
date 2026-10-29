
import os
import json
import requests

BASE = "https://llms.si-lab.org"
API_KEY = os.environ["OPENWEBUI_API_KEY"]

CHAT_MODEL = "openai/gpt-oss-120b"
PPLX_TOOL_ID = "server:pplxdeciderflash"
EMBED_MODEL = "google/embeddinggemma-2"

session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
})


def post(path, payload):
    response = session.post(
        BASE + path,
        json=payload,
        timeout=180,
    )
    print("HTTP status:", response.status_code)
    if not response.ok:
        print("Error:", response.text[:1000])
        return None
    return response.json()


# 1. Discover available models
print("\n=== AVAILABLE MODELS ===")
r = session.get(BASE + "/api/models", timeout=30)
r.raise_for_status()

model_ids = [m["id"] for m in r.json().get("data", [])]
for model in model_ids:
    print(model)


# 2. Test PPLX through Open WebUI tool
print("\n=== PPLX DECIDER ===")

decision = post("/api/chat/completions", {
    "model": CHAT_MODEL,
    "messages": [
        {
            "role": "system",
            "content": (
                "Use the registered PPLX Decider tool "
                "to make the decision. Do not classify "
                "the request yourself. Invoke the tool."
            )
        },
        {
            "role": "user",
            "content": (
                "Classify this request: "
                "A customer cannot authenticate "
                "using their API key. "
                "Choose technical, billing, or sales."
            )
        }
    ],
    "stream": False,
    "params": {"function_calling": "legacy"},
    "tool_ids": [PPLX_TOOL_ID],
})

if decision:
    print("Assistant response:")
    print(json.dumps(
        decision.get("choices", []),
        indent=2
    )[:3000])

    sources = decision.get("sources", [])
    verified = any(
        isinstance(source, dict)
        and source.get("tool_result") is True
        and (
            (source.get("source") or {})
            .get("name", "")
            .startswith(PPLX_TOOL_ID + "/")
        )
        for source in sources
    )
    print("PPLX tool evidence found:", verified)
    if not verified:
        print(
            "WARNING: This does not prove PPLX ran. "
            "Inspect the full response/tool configuration."
        )


# 3. Test EmbeddingGemma through Open WebUI
print("\n=== EMBEDDINGGEMMA 2 ===")

if EMBED_MODEL not in model_ids:
    print(
        "EmbeddingGemma is not listed in Open WebUI. "
        "It may need to be registered as a model connection."
    )

embedding = post("/api/v1/embeddings", {
    "model": EMBED_MODEL,
    "input": [
        "GraphRAG combines knowledge graphs with retrieval.",
        "PPLX Decider performs structured decisions."
    ]
})

if embedding:
    items = embedding.get("data", [])
    print("Embedding objects:", len(items))
    for i, item in enumerate(items):
        vector = item.get("embedding", [])
        print(f"Input {i+1}: {len(vector)} dimensions")
