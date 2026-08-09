import unittest

from sieve.core.executor import StateExecutor
from sieve.core.types import (
    BeliefSlot,
    BeliefState,
    Decision,
    EvidenceLedger,
    Observation,
    Patch,
    PatchOp,
    RevisionOutput,
    SlotStatus,
)


class ExecutorSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = StateExecutor()
        self.ledger = EvidenceLedger(2)

    def test_entity_mismatch_is_rejected_without_mutation(self) -> None:
        state = BeliefState(
            1,
            [BeliefSlot("status", "old", SlotStatus.TRUSTED, "api", 0, 0, "a")],
        )
        observation = Observation("status", "new", "api", 1, 1, "b", "current", True)
        output = RevisionOutput(
            Decision.UPDATE,
            ("status",),
            (Patch(PatchOp.SET_VALUE, "status", "new"),),
        )
        result = self.executor.apply(state, self.ledger, output, observation)
        self.assertFalse(result.executed)
        self.assertEqual(result.state.get("status").value, "old")

    def test_add_field_respects_fixed_capacity(self) -> None:
        state = BeliefState(
            1,
            [BeliefSlot("status", "old", SlotStatus.TRUSTED, "api", 0, 0, "a")],
        )
        observation = Observation("other", "value", "api", 1, 1, "a", "current", True)
        output = RevisionOutput(
            Decision.UPDATE,
            ("other",),
            (Patch(PatchOp.ADD_FIELD, "other", "value"),),
        )
        result = self.executor.apply(state, self.ledger, output, observation)
        self.assertFalse(result.executed)
        self.assertEqual(result.error, "no empty belief slot")

    def test_value_type_mismatch_is_rejected(self) -> None:
        state = BeliefState(
            1,
            [BeliefSlot("count", 1, SlotStatus.TRUSTED, "api", 0, 0, "a")],
        )
        observation = Observation("count", "two", "api", 1, 1, "a", "current", True)
        output = RevisionOutput(
            Decision.UPDATE,
            ("count",),
            (Patch(PatchOp.SET_VALUE, "count", "two"),),
        )
        result = self.executor.apply(state, self.ledger, output, observation)
        self.assertFalse(result.executed)
        self.assertEqual(result.state.get("count").value, 1)


if __name__ == "__main__":
    unittest.main()
