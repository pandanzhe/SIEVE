import unittest

from sieve.core.transition import AgentEventLoop, EnvironmentAction
from sieve.core.types import (
    BeliefSlot, BeliefState, Budget, Decision, EvidenceLedger, Observation,
    RevisionOutput, SlotStatus,
)
from sieve.environments.toy import ToyOrderEnvironment


class TransitionTests(unittest.TestCase):
    def test_revision_and_environment_clocks_are_distinct(self) -> None:
        env = ToyOrderEnvironment(seed=3)
        loop = AgentEventLoop(
            env=env,
            belief_state=BeliefState(3, [BeliefSlot("payment_status", "unpaid", SlotStatus.TRUSTED, "api", 0, 0, "o1")]),
            ledger=EvidenceLedger(3),
            budget=Budget(2, 4, 8),
        )
        obs = Observation("payment_status", "paid", "api", 1, 1, "o1", "current", True)
        loop.revise(obs, RevisionOutput(Decision.IGNORE))
        loop.revise(obs, RevisionOutput(Decision.IGNORE))
        self.assertEqual(loop.revision_t, 2)
        self.assertEqual(loop.action_k, 0)
        returned = loop.act(EnvironmentAction("verify", "payment_status"))
        self.assertEqual(loop.action_k, 1)
        self.assertEqual(loop.revision_t, 2)
        self.assertEqual(returned[0].source, "official_payment_api")


    def test_failed_tool_action_does_not_partially_consume_budget(self) -> None:
        budget = Budget(verification_remaining=2, tool_remaining=0, steps_remaining=8)
        loop = AgentEventLoop(
            env=ToyOrderEnvironment(seed=3),
            belief_state=BeliefState(3),
            ledger=EvidenceLedger(3),
            budget=budget,
        )
        self.assertEqual(loop.act(EnvironmentAction("verify", "payment_status")), [])
        self.assertEqual(budget.verification_remaining, 2)
        self.assertEqual(loop.action_k, 0)


if __name__ == "__main__":
    unittest.main()
