from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .models import GenerationCandidate, Realization


PROMPT_VERSION = "v1"


class GlmClient(Protocol):
    def realize(self, candidates: Sequence[GenerationCandidate]) -> list[Realization]: ...


def request_hash(
    candidates: Sequence[GenerationCandidate],
    *,
    prompt_version: str = PROMPT_VERSION,
    model: str = "glm-4.7",
    temperature: float = 0.6,
    max_tokens: int = 4096,
) -> str:
    payload = {
        "prompt_version": prompt_version,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "records": [
            {
                "record_id": candidate.record_id,
                "source_sha256": candidate.provenance.source_sha256,
                "transformation": candidate.provenance.transformation,
                "event": candidate.structured_event,
            }
            for candidate in candidates
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_messages(candidates: Sequence[GenerationCandidate]) -> list[dict[str, str]]:
    hidden_fields = {"condition", "relevant", "perturbation"}
    records = [
        {
            "record_id": candidate.record_id,
            "grounded_event": {
                key: value
                for key, value in candidate.structured_event.items()
                if key not in hidden_fields
            },
        }
        for candidate in candidates
    ]
    system = (
        "You convert grounded structured events into concise natural-language observations for "
        "data construction. Preserve entity, field value, source, source authority, authentication, observed_at, and valid_from. Include every entity_terms and value_terms item verbatim. "
        "Do not invent facts. Do not mention UPDATE, IGNORE, HOLD, decision, reason_code, "
        "conflict_type, labels, oracle state, or training data. Return one JSON object with a "
        "records array. Each item must contain only record_id and observation_text."
    )
    user = json.dumps({"records": records}, ensure_ascii=False, sort_keys=True)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def parse_realizations(
    content: str,
    candidates: Sequence[GenerationCandidate],
    digest: str,
    *,
    usage: dict[str, int] | None = None,
) -> list[Realization]:
    raw = json.loads(_strip_json_fence(content))
    items = raw.get("records") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("GLM response must contain a records array")
    by_id: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("GLM record must be an object")
        record_id = item.get("record_id")
        observation_text = item.get("observation_text")
        if not isinstance(record_id, str) or not isinstance(observation_text, str):
            raise ValueError("GLM record requires string record_id and observation_text")
        if record_id in by_id:
            raise ValueError(f"duplicate GLM record_id: {record_id}")
        by_id[record_id] = observation_text.strip()
    expected = [candidate.record_id for candidate in candidates]
    if set(by_id) != set(expected):
        raise ValueError("GLM response record IDs do not match the request")
    return [
        Realization(
            record_id,
            by_id[record_id],
            digest,
            raw_response=content,
            usage=dict(usage or {}),
        )
        for record_id in expected
    ]


class FakeGlmClient:
    model = "fake-glm"
    prompt_version = PROMPT_VERSION

    def realize(self, candidates: Sequence[GenerationCandidate]) -> list[Realization]:
        digest = request_hash(candidates, model=self.model)
        results = []
        for candidate in candidates:
            event = candidate.structured_event
            text = (
                f"At {event['observed_at']}, {event['source']} reported "
                f"{event['field_id']} for {event['entity']} as {event['value']}. "
                f"The value is valid from {event['valid_from']}; source authority is "
                f"{event.get('source_authority', 'unspecified')}; authenticated is "
                f"{event.get('authenticated', False)}. Task context: {event['source_text']}"
            )
            results.append(Realization(candidate.record_id, text, digest))
        return results


class ZhipuGlmClient:
    model = "glm-4.7"
    prompt_version = PROMPT_VERSION

    def __init__(
        self,
        *,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        max_retries: int = 2,
        retry_base_seconds: float = 1.0,
    ) -> None:
        api_key = os.environ.get("ZAI_API_KEY")
        if not api_key:
            raise RuntimeError("ZAI_API_KEY is required for live GLM generation")
        try:
            from zai import ZhipuAiClient
        except ImportError as error:
            raise RuntimeError("zai-sdk is required for live GLM generation") from error
        self._client = ZhipuAiClient(api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds

    def realize(self, candidates: Sequence[GenerationCandidate]) -> list[Realization]:
        if not candidates:
            return []
        digest = request_hash(
            candidates,
            prompt_version=self.prompt_version,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=build_messages(candidates),
                    thinking={"type": "enabled"},
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content
                usage_object = getattr(response, "usage", None)
                usage = {
                    name: int(value)
                    for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if (value := getattr(usage_object, name, None)) is not None
                }
                return parse_realizations(content, candidates, digest, usage=usage)
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.retry_base_seconds * (2**attempt))
        error_name = type(last_error).__name__ if last_error is not None else "UnknownError"
        raise RuntimeError(
            f"GLM request failed after {self.max_retries + 1} attempts ({error_name})"
        ) from last_error


class JsonlResponseCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, list[Realization]] = {}
        if self.path.is_file():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    self._entries[raw["request_hash"]] = [
                        Realization(**item) for item in raw["realizations"]
                    ]

    def get(self, digest: str) -> list[Realization] | None:
        value = self._entries.get(digest)
        return list(value) if value is not None else None

    def put(self, digest: str, realizations: Sequence[Realization]) -> None:
        if digest in self._entries:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            "request_hash": digest,
            "realizations": [
                {
                    "record_id": item.record_id,
                    "observation_text": item.observation_text,
                    "request_hash": item.request_hash,
                    "raw_response": item.raw_response,
                    "usage": item.usage,
                }
                for item in realizations
            ],
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(raw, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        self._entries[digest] = list(realizations)


class CachedGlmClient:
    def __init__(self, inner: GlmClient, cache: JsonlResponseCache) -> None:
        self.inner = inner
        self.cache = cache

    def realize(self, candidates: Sequence[GenerationCandidate]) -> list[Realization]:
        model = getattr(self.inner, "model", "glm-4.7")
        prompt_version = getattr(self.inner, "prompt_version", PROMPT_VERSION)
        temperature = getattr(self.inner, "temperature", 0.6)
        max_tokens = getattr(self.inner, "max_tokens", 4096)
        digest = request_hash(
            candidates,
            prompt_version=prompt_version,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        cached = self.cache.get(digest)
        if cached is not None:
            return cached
        results = self.inner.realize(candidates)
        normalized = [
            Realization(
                item.record_id,
                item.observation_text,
                digest,
                item.raw_response,
                item.usage,
            )
            for item in results
        ]
        self.cache.put(digest, normalized)
        return normalized


