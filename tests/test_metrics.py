import unittest

from sieve.core.types import Decision
from sieve.evaluation.metrics import (
    affected_set_f1,
    bootstrap_mean_interval,
    contamination_area,
    decision_metrics,
    pass_at_k,
    recovery_latency,
)


class MetricTests(unittest.TestCase):
    def test_decision_metrics_include_false_and_missed_updates(self) -> None:
        truth = [Decision.UPDATE, Decision.HOLD, Decision.IGNORE, Decision.UPDATE]
        predicted = [Decision.UPDATE, Decision.UPDATE, Decision.IGNORE, Decision.HOLD]
        metrics = decision_metrics(truth, predicted)
        self.assertAlmostEqual(metrics["false_update_rate"], 0.5)
        self.assertAlmostEqual(metrics["missed_update_rate"], 0.5)
        self.assertGreater(metrics["macro_f1"], 0.0)

    def test_affected_set_f1(self) -> None:
        self.assertEqual(affected_set_f1({"a", "b"}, {"a", "c"}), 0.5)
        self.assertEqual(affected_set_f1(set(), set()), 1.0)

    def test_trajectory_diagnostics(self) -> None:
        contamination = [0.0, 1.0, 1.0, 0.0]
        self.assertAlmostEqual(contamination_area(contamination), 2.0)
        self.assertEqual(recovery_latency(contamination), 2)
        self.assertEqual(pass_at_k([[True, True], [True, False]], 2), 0.5)

    def test_bootstrap_interval_is_seeded(self) -> None:
        first = bootstrap_mean_interval([0.0, 1.0, 1.0], seed=4, samples=100)
        second = bootstrap_mean_interval([0.0, 1.0, 1.0], seed=4, samples=100)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
