import unittest

from sieve.core.types import BeliefSlot, BeliefState, SlotStatus
from sieve.training.lagrangian import LagrangeController
from sieve.training.rewards import belief_consistency, potential_difference


class RewardTests(unittest.TestCase):
    def test_potential_reward_measures_change_not_repeated_correctness(self) -> None:
        previous = BeliefState(2, [BeliefSlot("payment_status", "unpaid", SlotStatus.TRUSTED, "api", 0, 0, "o1")])
        current = BeliefState(2, [BeliefSlot("payment_status", "paid", SlotStatus.TRUSTED, "api", 1, 1, "o1")])
        oracle = {"payment_status": "paid"}
        self.assertEqual(belief_consistency(previous, oracle, ("payment_status",)), 0.0)
        self.assertEqual(belief_consistency(current, oracle, ("payment_status",)), 1.0)
        self.assertEqual(potential_difference(previous, current, oracle, ("payment_status",), 1.0), 1.0)
        self.assertEqual(potential_difference(current, current, oracle, ("payment_status",), 1.0), 0.0)

    def test_lagrange_multipliers_are_projected_non_negative(self) -> None:
        controller = LagrangeController({"false_update": 0.1}, learning_rate=0.5)
        controller.update({"false_update": 0.5})
        self.assertGreater(controller.multipliers["false_update"], 0.0)
        controller.update({"false_update": 0.0})
        self.assertGreaterEqual(controller.multipliers["false_update"], 0.0)


if __name__ == "__main__":
    unittest.main()
