import unittest

from sieve.training.hf_sft_metrics import compute_structured_metrics


class HFSFTMetricTests(unittest.TestCase):
    def test_metrics_respect_branch_masks_and_report_false_updates(self) -> None:
        metrics = compute_structured_metrics(
            decision_predictions=[0, 0, 2, 1],
            decision_targets=[0, 2, 2, 1],
            affected_predictions=[[1, 0], [1, 1], [1, 1], [0, 1]],
            affected_targets=[[1, 0], [0, 0], [0, 0], [0, 1]],
            affected_masks=[[1, 1], [0, 0], [0, 0], [1, 1]],
            verification_predictions=[0, 1, 0, 1],
            verification_targets=[-100, -100, -100, 1],
            patch_predictions=[
                [[1, 0], [0, 0]],
                [[1, 1], [1, 0]],
                [[1, 1], [1, 1]],
                [[0, 0], [0, 0]],
            ],
            patch_targets=[
                [[1, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            patch_masks=[[1, 1], [0, 0], [0, 0], [0, 0]],
        )

        self.assertAlmostEqual(metrics["decision_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["false_update_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["affected_micro_f1"], 1.0)
        self.assertAlmostEqual(metrics["verification_f1"], 1.0)
        self.assertAlmostEqual(metrics["patch_operation_micro_f1"], 1.0)

    def test_empty_optional_branches_are_well_defined(self) -> None:
        metrics = compute_structured_metrics(
            decision_predictions=[2],
            decision_targets=[2],
            affected_predictions=[[0]],
            affected_targets=[[0]],
            affected_masks=[[0]],
            verification_predictions=[0],
            verification_targets=[-100],
            patch_predictions=[[[0]]],
            patch_targets=[[[0]]],
            patch_masks=[[0]],
        )

        self.assertEqual(metrics["affected_micro_f1"], 0.0)
        self.assertEqual(metrics["verification_f1"], 0.0)
        self.assertEqual(metrics["patch_operation_micro_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
