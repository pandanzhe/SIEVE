import unittest
from collections import Counter

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
from sieve.data_factory.trajectories import (
    build_corpus_candidates,
    build_verification_successor,
)


class TrajectoryFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = GroundedSourceRecord(
            provenance=Provenance(
                "tau2-bench", "commit", "airline:17:passenger_name",
                "tasks.json#17", "a" * 64, "MIT",
            ),
            domain="airline",
            scenario_type=ScenarioType.TRANSACTION,
            entity="booking:R8K4P2",
            field_id="passenger_name",
            old_value="LI MING",
            new_value="MING LI",
            source="airline_booking_api",
            observed_at=1_785_800_600,
            valid_from=1_785_800_600,
            goal="Complete check-in for booking R8K4P2 after confirming identity.",
            source_text="The booking may require a passenger-name correction.",
        )

    def test_hold_successor_uses_executed_state_ledger_and_budget(self) -> None:
        hold = build_candidate(
            self.source,
            QuotaCell(
                ObservationType.TOOL_ERROR_OR_LOW_TRUST,
                Decision.HOLD,
                1,
                VerificationAction.VERIFY,
            ),
            sequence=1,
        )

        successor = build_verification_successor(hold, sequence=2)

        self.assertEqual(successor.group_id, hold.group_id)
        self.assertEqual(successor.context.belief_state, hold.oracle_state)
        self.assertEqual(len(successor.context.ledger.entries), 1)
        self.assertEqual(
            successor.context.budget.verification_remaining,
            hold.context.budget.verification_remaining - 1,
        )
        self.assertEqual(successor.target.decision, Decision.UPDATE)
        self.assertEqual(successor.target.patches[0].value, hold.context.observation.value)
        self.assertEqual(successor.context.observation.source_authority, "primary_record")
        self.assertTrue(successor.context.observation.authenticated)


    def test_corpus_schedule_has_exact_ratios_and_paired_holds(self) -> None:
        sources = []
        domains = ["retail", "airline", "telecom", "tool_api"]
        for index in range(16):
            domain = domains[index % len(domains)]
            sources.append(GroundedSourceRecord(
                provenance=Provenance(
                    "fixture", "commit", str(index), f"fixture.json#{index}",
                    f"{index:064x}", "MIT",
                ),
                domain=domain,
                scenario_type=ScenarioType.TRANSACTION,
                entity=f"{domain}:entity-{index}",
                field_id="status",
                old_value="old",
                new_value=f"new-{index}",
                source="service_api",
                observed_at=1_700_000_000 + index,
                valid_from=1_700_000_000 + index,
                goal=f"complete {domain} task {index}",
                source_text=f"grounded {domain} task {index}",
            ))

        records = build_corpus_candidates(sources, total=200, seed=7)

        self.assertEqual(len(records), 200)
        observation_times = [item.context.observation.observed_at for item in records]
        self.assertEqual(observation_times, sorted(observation_times))
        self.assertEqual(len(set(observation_times)), len(observation_times))
        self.assertEqual(
            Counter(item.target.decision.value for item in records),
            {"UPDATE": 140, "HOLD": 30, "IGNORE": 30},
        )
        by_group = {}
        for item in records:
            by_group.setdefault(item.group_id, []).append(item)
        paired = [items for items in by_group.values() if len(items) == 2]
        self.assertEqual(len(paired), 30)
        self.assertTrue(all(
            [item.target.decision for item in items] == [Decision.HOLD, Decision.UPDATE]
            for items in paired
        ))
if __name__ == "__main__":
    unittest.main()
