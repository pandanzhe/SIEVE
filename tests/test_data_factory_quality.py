import unittest

from sieve.core.types import Decision
from sieve.data_factory.models import (
    GroundedSourceRecord,
    ObservationType,
    Provenance,
    QuotaCell,
    Realization,
    ScenarioType,
)
from sieve.data_factory.quality import validate_candidate
from sieve.data_factory.rules import build_candidate


class QualityTests(unittest.TestCase):
    def setUp(self) -> None:
        source = GroundedSourceRecord(
            Provenance("tau2-bench", "commit", "r1", "tasks.json#r1", "a" * 64, "MIT"),
            "retail",
            ScenarioType.TRANSACTION,
            "order-1",
            "status",
            "processing",
            "shipped",
            "official_order_api",
            200,
            200,
            "track the order",
            "The order record reports shipped.",
        )
        self.candidate = build_candidate(
            source,
            QuotaCell(ObservationType.NEW_CONSISTENT, Decision.UPDATE, 1),
            sequence=1,
        )

    def realization(self, text: str) -> Realization:
        return Realization(self.candidate.record_id, text, "b" * 64)

    def test_valid_record_executes_to_oracle(self) -> None:
        result = validate_candidate(
            self.candidate,
            self.realization("The official order API reports order-1 status as shipped."),
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason_codes, ())

    def test_label_leakage_is_rejected(self) -> None:
        result = validate_candidate(
            self.candidate,
            self.realization("decision=UPDATE; order-1 status is shipped."),
        )
        self.assertIn("label_leakage", result.reason_codes)

    def test_generation_only_metadata_is_rejected_from_observation_text(self) -> None:
        for leaked in (
            "relevant: true",
            "condition: applies_to_current_entity",
            "perturbation: stale",
        ):
            with self.subTest(leaked=leaked):
                result = validate_candidate(
                    self.candidate,
                    self.realization(f"order-1 status is shipped; {leaked}"),
                )
                self.assertIn("label_leakage", result.reason_codes)
    def test_missing_parent_for_counterfactual_is_rejected(self) -> None:
        broken = self.candidate.__class__(
            **{
                **self.candidate.__dict__,
                "provenance": self.candidate.provenance.__class__(
                    **{**self.candidate.provenance.__dict__, "parent_record_id": None}
                ),
            }
        )
        result = validate_candidate(
            broken,
            self.realization("The official order API reports order-1 status as shipped."),
        )
        self.assertIn("missing_parent_record_id", result.reason_codes)

    def test_semantic_value_change_is_rejected(self) -> None:
        result = validate_candidate(
            self.candidate,
            self.realization("The official order API reports order-1 status as cancelled."),
        )
        self.assertIn("missing_grounded_value", result.reason_codes)

    def test_atomic_terms_accept_natural_language_rendering_of_list(self) -> None:
        source = GroundedSourceRecord(
            Provenance("tau2-bench", "commit", "r2", "tasks.json#r2", "c" * 64, "MIT"),
            "retail",
            ScenarioType.TRANSACTION,
            "order-2",
            "item_ids",
            "unknown",
            [8926329222, 5312063289],
            "official_order_api",
            201,
            201,
            "modify the order",
            "Replace the two items.",
        )
        candidate = build_candidate(
            source,
            QuotaCell(ObservationType.NEW_CONSISTENT, Decision.UPDATE, 1),
            sequence=2,
        )
        result = validate_candidate(
            candidate,
            Realization(
                candidate.record_id,
                "For order-2, item IDs are 8926329222 and 5312063289.",
                "d" * 64,
            ),
        )
        self.assertTrue(result.accepted)



if __name__ == "__main__":
    unittest.main()
