from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .glm import (
    CachedGlmClient,
    FakeGlmClient,
    GlmClient,
    JsonlResponseCache,
    PROMPT_VERSION,
    ZhipuGlmClient,
    request_hash,
)
from .models import GenerationCandidate, Realization


class CanonicalGlmClient:
    """Wraps a GLM client with canonical publication constraints."""

    model: str
    prompt_version: str

    def __init__(self, inner: GlmClient, *, allow_preview: bool = False) -> None:
        if not allow_preview and isinstance(inner, FakeGlmClient):
            raise RuntimeError(
                "Canonical publication requires a live GLM client, not FakeGlmClient"
            )
        self.inner = inner
        self.allow_preview = allow_preview
        self.model = getattr(inner, "model", "unknown")
        self.prompt_version = getattr(inner, "prompt_version", PROMPT_VERSION)

    def realize(self, candidates: Sequence[GenerationCandidate]) -> list[Realization]:
        realizations = self.inner.realize(candidates)
        total_tokens = sum(r.usage.get("total_tokens", 0) for r in realizations)
        if total_tokens == 0:
            raise RuntimeError("Canonical publication requires positive token usage")
        return realizations


def capture_generation_metadata(
    client: GlmClient,
    candidates: list[GenerationCandidate],
    realizations: list[Realization],
) -> list[dict[str, Any]]:
    """Build per-record generation metadata dicts for audit/export."""
    model = getattr(client, "model", "unknown")
    prompt_version = getattr(client, "prompt_version", PROMPT_VERSION)
    results: list[dict[str, Any]] = []
    for candidate, realization in zip(candidates, realizations):
        results.append(
            {
                "generator": model,
                "prompt_version": prompt_version,
                "source_task_id": candidate.provenance.source_record_id,
                "source_file_sha256": candidate.provenance.source_sha256,
                "request_hash": realization.request_hash,
                "token_usage": dict(realization.usage),
                "result_hash": hashlib.sha256(
                    realization.observation_text.encode("utf-8")
                ).hexdigest(),
                "record_id": candidate.record_id,
            }
        )
    return results


def create_v3_client(
    *,
    dry_run: bool = False,
    cache_path: str | Path | None = None,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    max_retries: int = 2,
    allow_preview: bool = False,
) -> GlmClient:
    """Construct the canonical Stage-1 SFT v3 generation client.

    In dry-run mode the client is FakeGlmClient with no constraints.
    Otherwise the pipeline is ZhipuGlmClient -> CachedGlmClient (if
    cache_path) -> CanonicalGlmClient.
    """
    if dry_run:
        return FakeGlmClient()

    inner: GlmClient = ZhipuGlmClient(
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )

    if cache_path is not None:
        inner = CachedGlmClient(inner, JsonlResponseCache(cache_path))

    inner = CanonicalGlmClient(inner, allow_preview=allow_preview)
    return inner
