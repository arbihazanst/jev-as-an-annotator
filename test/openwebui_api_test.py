"""Test SGLang model and PPLX tool through your OpenWebUI account API.

Install dependency: pip install requests
Edit API_KEY, MODEL_ID, and PPLX_TOOL_ID below, then run:
    python openwebui_api_test.py

Do not commit this file or share it after inserting your real API key.
"""

import json
import requests

# ---- EDIT THESE VALUES (kept inside this file; no Linux environment variables) ----
BASE_URL = "https://llms.si-lab.org"
API_KEY = "API_KEY"
MODEL_ID = "openai/gpt-oss-120b"
PPLX_TOOL_ID = "server:pplxdeciderflash"
# -------------------------------------------------------------------------------

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def request_json(method, path, **kwargs):
    response = requests.request(
        method, f"{BASE_URL}{path}", headers=HEADERS, timeout=180, **kwargs
    )
    response.raise_for_status()
    return response.json()


def print_json(label, data):
    print(f"\n=== {label} ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    if API_KEY.startswith("PASTE_"):
        raise SystemExit("First paste your OpenWebUI API key into API_KEY in this file.")

    # List models visible to your user account. Copy the SGLang ID into MODEL_ID.
    models = request_json("GET", "/api/models")
    print_json("Available model IDs", [m.get("id") for m in models.get("data", [])])

    # Workspace Tools can appear here. Externally connected MCP/OpenAPI tool
    # servers may not be listed by this endpoint.
    try:
        tools = request_json("GET", "/api/v1/tools/")
        print_json("Workspace tools", [{"id": t.get("id"), "name": t.get("name")} for t in tools])
    except requests.HTTPError as exc:
        print(f"Workspace tool listing unavailable: {exc}")

    if MODEL_ID.startswith("PASTE_"):
        print("\nSet MODEL_ID to test SGLang chat. Skipping chat requests.")
        return

    # SGLang: directly invoke the model through OpenWebUI.
    response = request_json("POST", "/api/chat/completions", json={
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Explain RAG in one sentence."}],
        "stream": False,
    })
    print_json("SGLang answer", response)

    if PPLX_TOOL_ID.startswith("PASTE_"):
        print("\nSet PPLX_TOOL_ID if you also want to test PPLX through an OpenWebUI tool.")
        return

    # PPLX: request that OpenWebUI invoke the registered tool through the model.
    # This is not a direct call to PPLX's own API; actual tool use depends on
    # tool permissions and OpenWebUI's tool-execution settings.
    response = request_json("POST", "/api/chat/completions", json={
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": (
            "Use the PPLX decision tool to classify this request: "
            "The customer wants a refund for a damaged product."
        )}],
        "stream": False,
        "params": {"function_calling": "legacy"},
        "tool_ids": [PPLX_TOOL_ID],
    })
    print_json("Model response with PPLX tool enabled", response)


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as error:
        raise SystemExit(f"API request failed: {error}") from error
