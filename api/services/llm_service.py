"""Task 4: Implement analyze_journal_entry using any OpenAI-compatible API.

This project mandates the OpenAI Python SDK, which works with:
  - GitHub Models (default, free, no credit card required)
  - OpenAI proper
  - Azure OpenAI
  - Groq, Together, OpenRouter, Fireworks, DeepInfra
  - Ollama, LM Studio, vLLM (local)
  - Anthropic via their OpenAI-compat endpoint

Set OPENAI_API_KEY, and optionally OPENAI_BASE_URL and OPENAI_MODEL
in your .env file. Settings are loaded by ``api.config.Settings``.
"""

import json
from typing import Any

from openai import AsyncOpenAI

from api.config import get_settings


def _default_client() -> AsyncOpenAI:
    """Construct the real OpenAI client from application settings.

    Called lazily from ``analyze_journal_entry`` so tests can inject a
    ``MockAsyncOpenAI`` without ever triggering this code path.
    """
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


async def analyze_journal_entry(
    entry_id: str,
    entry_text: str,
    client: AsyncOpenAI | None = None,
) -> dict[str, Any]:
    if client is None:
        client = _default_client()

    settings = get_settings()

    messages: list[Any] = [
        {
            "role": "system",
            "content": (
                "Analyze the journal entry. Respond ONLY in JSON format with these keys: "
                "sentiment (positive/negative/neutral), summary (2 sentences), "
                "and topics (list of 2-4 strings)."
            ),
        },
        {"role": "user", "content": entry_text},
    ]

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    analysis = json.loads(content)

    return {
        "entry_id": entry_id,
        "sentiment": analysis.get("sentiment"),
        "summary": analysis.get("summary"),
        "topics": analysis.get("topics"),
    }


# Final Task 4 Push - Force Git Update
