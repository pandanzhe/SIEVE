import unittest

from sieve.core.types import (
    BeliefSlot,
    BeliefState,
    Budget,
    Decision,
    EvidenceLedger,
    Observation,
    Patch,
    PatchOp,
    RevisionContext,
    RevisionOutput,
    RiskEnvelope,
    RiskLevel,
    SlotStatus,
)
from sieve.data.schema import SFTRecord
from sieve.training.hf_readiness import evaluate_generated_actions


def _record(decision: Decision, index: int) -> SFTRecord:
    field = f"field-{index}"
    context = RevisionContext(
        belief_state=BeliefState(
            2,
            [BeliefSlot(field, None, SlotStatus.EMPTY, "initial", 1, None, f"e:{index}")],
        ),
        observation=Observation(
            field, "value", "official", 2, 2, f"e:{index}", "test", True
        ),
        goal="test",
        risk=RiskEnvelope("test", (field,), RiskLevel.MEDIUM, True),
        budget=Budget(1, 1, 2),
        ledger=EvidenceLedger(2),
    )
    return SFTRecord(f"s-{index}", context, RevisionOutput(decision))


class HFReadinessTests(unittest.TestCase):
    def test_generation_metrics_penalize_invalid_json_and_false_updates(self) -> None:
        records = [_record(Decision.IGNORE, 0), _record(Decision.HOLD, 1)]
        completions = [
            '{"decision":"UPDATE","affected_fields":["field-0"],'
            '"patches":[{"op":"SET_VALUE","field_id":"field-0","value":"value"}],'
            '"verification":null}',
            "not-json",
        ]
        metrics = evaluate_generated_actions(completions, records)
        self.assertEqual(metrics["parse_rate"], 0.5)
        self.assertEqual(metrics["false_update_rate"], 0.5)
        self.assertEqual(metrics["full_action_exact_match"], 0.0)
        self.assertEqual(metrics["executable_rate"], 0.5)

    def test_patch_value_exact_match_is_reported_on_updates(self) -> None:
        record = _record(Decision.UPDATE, 2)
        field = record.context.observation.field_id
        record = SFTRecord(
            record.scenario_id,
            record.context,
            RevisionOutput(
                Decision.UPDATE,
                (field,),
                (Patch(PatchOp.SET_VALUE, field, "gold"),),
            ),
        )
        completion = (
            '{"decision":"UPDATE","affected_fields":["field-2"],'
            '"patches":[{"op":"SET_VALUE","field_id":"field-2",'
            '"value":"wrong"}],"verification":null}'
        )
        metrics = evaluate_generated_actions([completion], [record])
        self.assertEqual(metrics["patch_value_exact_match"], 0.0)


if __name__ == "__main__":
    unittest.main()
