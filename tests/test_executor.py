import unittest

from sieve.core.executor import StateExecutor
from sieve.core.types import (
    BeliefSlot, BeliefState, Decision, EvidenceLedger, Observation,
    Patch, PatchOp, RevisionOutput, RiskEnvelope, RiskLevel, SlotStatus,
)


def base_state() -> BeliefState:
    return BeliefState(
        max_slots=3,
        slots=[BeliefSlot("payment_status", "unpaid", SlotStatus.TRUSTED, "api", 10, 10, "o1")],
    )


class ExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = StateExecutor()
        self.ledger = EvidenceLedger(capacity=2)
        self.obs = Observation("payment_status", "paid", "cache", 20, None, "o1", "current", True, "weak_source")

    def test_ignore_preserves_state(self) -> None:
        state = base_state()
        result = self.executor.apply(state, self.ledger, RevisionOutput(Decision.IGNORE), self.obs)
        self.assertEqual(result.state, state)
        self.assertIsNot(result.state, state)

    def test_hold_preserves_value_and_isolates_candidate(self) -> None:
        output = RevisionOutput(
            Decision.HOLD,
            ("payment_status",),
            (Patch(PatchOp.SET_STATUS, "payment_status", SlotStatus.PENDING.value),),
        )
        result = self.executor.apply(base_state(), self.ledger, output, self.obs)
        slot = result.state.get("payment_status")
        self.assertEqual(slot.value, "unpaid")
        self.assertEqual(slot.status, SlotStatus.PENDING)
        self.assertEqual(len(result.ledger.entries), 1)
        self.assertEqual(result.ledger.entries[0].observation.value, "paid")

    def test_update_cannot_touch_unaffected_field(self) -> None:
        output = RevisionOutput(
            Decision.UPDATE,
            ("shipping_address",),
            (Patch(PatchOp.SET_VALUE, "payment_status", "paid"),),
        )
        result = self.executor.apply(base_state(), self.ledger, output, self.obs)
        self.assertFalse(result.executed)
        self.assertEqual(result.costs["invalid_patch"], 1.0)
        self.assertEqual(result.costs["collateral_edit"], 1.0)
        self.assertEqual(result.state.get("payment_status").value, "unpaid")

    def test_ledger_capacity_is_fixed(self) -> None:
        ledger = EvidenceLedger(capacity=2)
        for index in range(3):
            ledger.append(self.obs, f"reason-{index}")
        self.assertEqual(len(ledger.entries), 2)
        self.assertEqual(ledger.entries[0].reason, "reason-1")

    def test_pending_dependency_blocks_irreversible_action(self) -> None:
        state = base_state()
        state.get("payment_status").status = SlotStatus.PENDING
        risk = RiskEnvelope("ship", ("payment_status",), RiskLevel.HIGH, False)
        self.assertTrue(self.executor.blocks_action(state, risk))


if __name__ == "__main__":
    unittest.main()
