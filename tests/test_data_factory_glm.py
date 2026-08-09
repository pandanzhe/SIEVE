import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sieve.core.types import Decision
from sieve.data_factory.glm import (
    CachedGlmClient,
    FakeGlmClient,
    JsonlResponseCache,
    ZhipuGlmClient,
    request_hash,
    build_messages,
)
from sieve.data_factory.models import (
    GroundedSourceRecord,
    ObservationType,
    Provenance,
    QuotaCell,
    ScenarioType,
)
from sieve.data_factory.rules import build_candidate


class CountingFake(FakeGlmClient):
    def __init__(self) -> None:
        self.calls = 0

    def realize(self, candidates):
        self.calls += 1
        return super().realize(candidates)


class GlmTests(unittest.TestCase):
    def setUp(self) -> None:
        source = GroundedSourceRecord(
            Provenance("WebArena", "commit", "3", "test.raw.json#3", "a" * 64, "Apache-2.0"),
            "shopping", ScenarioType.INFORMATION_TOOL, "shopping:3", "reference_answer",
            "unknown", "Product A", "shopping_website", 200, 200,
            "find the best seller", "Find the best seller",
        )
        self.candidate = build_candidate(
            source, QuotaCell(ObservationType.NEW_CONSISTENT, Decision.UPDATE, 1), sequence=1
        )
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_fake_client_realizes_without_network(self) -> None:
        result = FakeGlmClient().realize([self.candidate])
        self.assertEqual(result[0].record_id, self.candidate.record_id)
        self.assertIn("shopping:3", result[0].observation_text)
        self.assertIn("Product A", result[0].observation_text)

    def test_fake_observation_contains_only_runtime_evidence(self) -> None:
        text = FakeGlmClient().realize([self.candidate])[0].observation_text.casefold()

        self.assertNotIn("relevant", text)
        self.assertNotIn("condition", text)
        self.assertNotIn("perturbation", text)
        self.assertNotIn("decision", text)
    def test_real_client_requires_environment_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ZAI_API_KEY"):
                ZhipuGlmClient()

    def test_request_hash_is_stable(self) -> None:
        first = request_hash([self.candidate], prompt_version="v1")
        second = request_hash([self.candidate], prompt_version="v1")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_cache_prevents_duplicate_client_call(self) -> None:
        inner = CountingFake()
        cache = JsonlResponseCache(Path(self.temp_dir.name) / "cache.jsonl")
        client = CachedGlmClient(inner, cache)
        first = client.realize([self.candidate])
        second = client.realize([self.candidate])
        self.assertEqual(first, second)
        self.assertEqual(inner.calls, 1)

    def test_prompt_requires_verbatim_grounding_terms(self) -> None:
        messages = build_messages([self.candidate])
        self.assertIn("verbatim", messages[0]["content"])
        payload = messages[1]["content"]
        self.assertIn('"entity_terms": ["shopping:3"]', payload)
        self.assertIn('"value_terms": ["Product A"]', payload)


if __name__ == "__main__":
    unittest.main()
