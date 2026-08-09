import unittest

from sieve.training.constrained_grpo import (
    clipped_ratio,
    policy_gradient_coefficient,
)


class GRPOMathTests(unittest.TestCase):
    def test_reported_ratio_is_clipped(self) -> None:
        self.assertAlmostEqual(clipped_ratio(1.0, 0.0, 0.2), 1.2)
        self.assertAlmostEqual(clipped_ratio(-1.0, 0.0, 0.2), 0.8)

    def test_surrogate_gradient_stops_on_clipped_side(self) -> None:
        self.assertEqual(policy_gradient_coefficient(1.0, 0.0, 1.0, 0.2), 0.0)
        self.assertEqual(policy_gradient_coefficient(-1.0, 0.0, -1.0, 0.2), 0.0)
        self.assertGreater(policy_gradient_coefficient(0.1, 0.0, 1.0, 0.2), 0.0)
        self.assertLess(policy_gradient_coefficient(0.1, 0.0, -1.0, 0.2), 0.0)


if __name__ == "__main__":
    unittest.main()
