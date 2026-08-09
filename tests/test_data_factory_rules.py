import unittest

from sieve.core.types import Decision
from sieve.data_factory.models import (
    GroundedSourceRecord,
    ObservationType,
    Provenance,
    QuotaCell,
    ScenarioType,
    VerificationAction,
)
from sieve.data_factory.rules import build_candidate


class RuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = GroundedSourceRecord(
            provenance=Provenance(
                "tau2-bench",
                "commit-abc",
                "retail-1",
                "tasks.json#retail-1",
                "a" * 64,
                "MIT",
            ),
            domain="retail",
            scenario_type=ScenarioType.TRANSACTION,
            entity="order-1",
            field_id="status",
            old_value="processing",
            new_value="shipped",
            source="official_order_api",
            observed_at=200,
            valid_from=200,
            goal="track order",
            source_text="The official order result reports shipped.",
        )

    def cell(
        self,
        observation_type: ObservationType,
        decision: Decision,
        verification_action: VerificationAction = VerificationAction.NO_VERIFY,
    ) -> QuotaCell:
        return QuotaCell(observation_type, decision, 1, verification_action)

    def test_authoritative_newer_value_updates(self) -> None:
        candidate = build_candidate(
            self.source,
            self.cell(ObservationType.NEW_CONSISTENT, Decision.UPDATE),
            sequence=1,
        )
        self.assertEqual(candidate.target.decision, Decision.UPDATE)
        self.assertEqual(candidate.target.affected_fields, ("status",))
        self.assertEqual(candidate.oracle_state.get("status").value, "shipped")
        self.assertEqual(candidate.context.observation.source_authority, "primary_record")
        self.assertTrue(candidate.context.observation.authenticated)

    def test_explicit_conflict_is_visible_without_internal_taxonomy(self) -> None:
        consistent = build_candidate(
            self.source,
            self.cell(ObservationType.NEW_CONSISTENT, Decision.UPDATE),
            sequence=6,
        )
        conflict = build_candidate(
            self.source,
            self.cell(ObservationType.EXPLICIT_CONFLICT, Decision.UPDATE),
            sequence=7,
        )
        self.assertNotEqual(
            consistent.context.observation.condition,
            conflict.context.observation.condition,
        )


    def test_wrong_entity_conflict_ignores_without_patch(self) -> None:
        candidate = build_candidate(
            self.source,
            self.cell(ObservationType.EXPLICIT_CONFLICT, Decision.IGNORE),
            sequence=2,
        )
        self.assertEqual(candidate.target.decision, Decision.IGNORE)
        self.assertEqual(candidate.target.patches, ())
        self.assertNotEqual(
            candidate.context.observation.entity,
            candidate.context.belief_state.get("status").entity,
        )

    def test_low_trust_high_risk_holds_for_verification(self) -> None:
        candidate = build_candidate(
            self.source,
            self.cell(
                ObservationType.TOOL_ERROR_OR_LOW_TRUST,
                Decision.HOLD,
                VerificationAction.VERIFY,
            ),
            sequence=3,
        )
        self.assertEqual(candidate.target.decision, Decision.HOLD)
        self.assertIsNotNone(candidate.target.verification)
        self.assertEqual(candidate.oracle_state.get("status").status.value, "pending_verification")
        self.assertEqual(candidate.context.observation.source_authority, "secondary")
        self.assertFalse(candidate.context.observation.authenticated)

    def test_unknown_prior_is_an_empty_slot_not_a_trusted_string(self) -> None:
        source = GroundedSourceRecord(
            **{**self.source.__dict__, "old_value": "unknown"}
        )

        candidate = build_candidate(
            source,
            self.cell(ObservationType.NEW_CONSISTENT, Decision.UPDATE),
            sequence=8,
        )

        slot = candidate.context.belief_state.get("status")
        self.assertIsNone(slot.value)
        self.assertEqual(slot.status.value, "empty")
        self.assertEqual(slot.source, "initial_state")
    def test_stale_update_is_newer_than_the_current_belief(self) -> None:
        candidate = build_candidate(
            self.source,
            self.cell(ObservationType.STALE, Decision.UPDATE),
            sequence=4,
        )
        current = candidate.context.belief_state.get("status")
        observation = candidate.context.observation
        self.assertLess(observation.valid_from, observation.observed_at)
        self.assertGreater(observation.valid_from, current.valid_from)

    def test_structured_value_uses_atomic_grounding_terms(self) -> None:
        source = GroundedSourceRecord(
            **{
                **self.source.__dict__,
                "new_value": [8926329222, 5312063289],
            }
        )
        candidate = build_candidate(
            source,
            self.cell(ObservationType.NEW_CONSISTENT, Decision.UPDATE),
            sequence=5,
        )
        self.assertEqual(
            candidate.structured_event["grounding"]["value_terms"],
            ["8926329222", "5312063289"],
        )
        self.assertEqual(
            candidate.structured_event["grounding"]["entity_terms"],
            ["order-1"],
        )



if __name__ == "__main__":
    unittest.main()
