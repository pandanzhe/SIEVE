import math
import unittest
from pathlib import Path

import numpy as np

from sieve.data.generate import generate_toy_trajectories
from sieve.policies.features import HashedFeaturizer
from sieve.policies.structured_policy import StructuredPolicy


TEST_ROOT = Path(__file__).resolve().parents[1] / "tmp" / "tests"


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = generate_toy_trajectories(3, seed=2, max_slots=4)

    def test_feature_dimension_is_fixed(self) -> None:
        encoder = HashedFeaturizer(64)
        shapes = {encoder.encode(record.context).shape for record in self.records}
        self.assertEqual(shapes, {(64,)})

    def test_action_masks_produce_valid_structures(self) -> None:
        policy = StructuredPolicy(64, 4, seed=1)
        for record in self.records:
            output = policy.predict(record.context, greedy=True)
            if output.decision.value == "IGNORE":
                self.assertFalse(output.affected_fields)
                self.assertFalse(output.patches)
            if record.context.budget.verification_remaining == 0:
                self.assertIsNone(output.verification)

    def test_conditional_log_probability_is_finite(self) -> None:
        policy = StructuredPolicy(64, 4, seed=1)
        value = policy.log_probability(self.records[1].context, self.records[1].target)
        self.assertTrue(math.isfinite(value))

    def test_checkpoint_round_trip(self) -> None:
        path = TEST_ROOT / "policy.npz"
        first = StructuredPolicy(64, 4, seed=1)
        first.save(path)
        second = StructuredPolicy.load(path)
        context = self.records[0].context
        np.testing.assert_allclose(first.decision_logits(context), second.decision_logits(context))


if __name__ == "__main__":
    unittest.main()
