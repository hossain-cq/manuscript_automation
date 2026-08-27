from __future__ import annotations

from functools import lru_cache
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from ..settings import get_settings


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Single place that builds the OpenAI-compatible client.

    Replaces the bare `OpenAI()` calls previously duplicated in each of the
    three prototype graph modules. Reads OPENAI_API_KEY / OPENAI_API_BASE from
    settings (.env) and fails loudly rather than silently hitting the public
    API with no key configured.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.openai_api_base:
        kwargs["base_url"] = settings.openai_api_base
        # The openai SDK retries 429s with backoff (respecting Retry-After)
        # using its own default max_retries=2, which isn't enough headroom
        # for Groq's free-tier per-minute cap: a node making several
        # sequential structured-output calls (e.g. run_audits' 3 assessor
        # roles) can trip the rolling TPM limit between calls even though
        # each individual request fits under it (confirmed: 429 "Used 2100,
        # Requested 7338... try again in 10.785s" on the 2nd of 3 calls).
        # OpenAI's own TPM limits are high enough that this isn't needed.
        kwargs["max_retries"] = 6
    return OpenAI(**kwargs)


def default_extra_body() -> dict:
    """Provider-conditional `extra_body` for structured-output calls.

    `{"reasoning": {"effort": "medium"}}` is an OpenAI-only extension for
    reasoning-model effort control (gpt-5 family). Sending it to any other
    OpenAI-compatible provider (Groq, a local vLLM/Ollama server, ...) fails
    with `property 'reasoning' is unsupported` — confirmed against Groq's
    `openai/gpt-oss-120b`. Only include it when OPENAI_API_BASE is unset,
    i.e. we're actually talking to OpenAI.
    """
    settings = get_settings()
    if settings.openai_api_base:
        return {}
    return {"reasoning": {"effort": "medium"}}


def default_max_completion_tokens(preferred: int) -> int:
    """Cap `max_completion_tokens` for non-OpenAI providers.

    Groq's free tier enforces an 8000 tokens-per-minute limit per model that
    covers prompt + completion together, so requesting 8000 completion
    tokens alone can exceed it on its own — confirmed: a real
    `LLMReadinessAssessment` call was rejected with "Limit 8000, Requested
    9542" against `openai/gpt-oss-120b`. 4000 was verified to fit comfortably
    for both `openai/gpt-oss-20b` and `openai/gpt-oss-120b`. OpenAI's own TPM
    limits are far higher, so the caller's preferred value is left alone
    there.
    """
    settings = get_settings()
    if settings.openai_api_base:
        return min(preferred, 4000)
    return preferred


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A JSON schema for `response_format: {"type": "json_schema", "strict": true}`
    that actually satisfies strict mode.

    Pydantic's `model_json_schema()` only lists fields *without* a default in
    `required` — a field like `abstain: bool = False` gets left out. Strict
    structured-output mode requires every property to be listed in
    `required` regardless of whether it has a default (confirmed against
    Groq: rejected `LLMReadinessAssessment`, which has `abstain: bool = False`,
    with "`required` is required to be supplied and to be an array including
    every key in properties"). Force `required` to equal `properties` at
    every level of the schema, including nested `$defs`, rather than auditing
    each Pydantic model by hand for defaulted fields.
    """
    schema = model.model_json_schema()

    def _fix(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"].keys())
            for value in node.values():
                _fix(value)
        elif isinstance(node, list):
            for item in node:
                _fix(item)

    _fix(schema)
    return schema
