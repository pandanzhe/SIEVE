import unittest

import numpy as np

from sieve.evaluation.hf_stage2 import totals_to_metrics


class HFStage2EvaluationTests(unittest.TestCase):
    def test_totals_are_converted_to_episode_and_action_rates(self) -> None:
        metrics = totals_to_metrics(
            np.array([2, 1, 1.5, 10, 9, 1, 1, 3, 2, 0], dtype=np.float64)
        )
        self.assertEqual(metrics["episodes"], 2.0)
        self.assertEqual(metrics["success_rate"], 0.5)
        self.assertEqual(metrics["mean_return"], 0.75)
        self.assertEqual(metrics["parse_rate"], 0.9)
        self.assertEqual(metrics["false_update_rate"], 0.1)
        self.assertEqual(metrics["verification_rate"], 0.3)
        self.assertEqual(metrics["stall_rate"], 0.2)


if __name__ == "__main__":
    unittest.main()
