import unittest

import numpy as np

from sieve.data.generate import generate_toy_trajectories
from sieve.policies.structured_policy import StructuredPolicy
from sieve.training.constrained_grpo import clipped_ratio, train_constrained_grpo
from sieve.training.sft import train_sft


class RLTests(unittest.TestCase):
    def test_probability_ratio_is_clipped(self) -> None:
        self.assertAlmostEqual(clipped_ratio(new_log_probability=1.0, old_log_probability=0.0, clip=0.2), 1.2)
        self.assertAlmostEqual(clipped_ratio(new_log_probability=-1.0, old_log_probability=0.0, clip=0.2), 0.8)

    def test_seeded_constrained_training_smoke_run(self) -> None:
        records = generate_toy_trajectories(30, seed=9, max_slots=4)
        policy = StructuredPolicy(96, 4, seed=9)
        train_sft(policy, records, 8, 0.06, seed=9)
        before = policy.w_decision.copy()
        history = train_constrained_grpo(
            policy,
            records,
            iterations=3,
            groups_per_iteration=3,
            group_size=3,
            learning_rate=0.01,
            seed=9,
            constraints={"false_update": 0.1, "unsafe_action": 0.05, "verification": 0.5, "stall": 0.3},
        )
        self.assertEqual(len(history), 3)
        self.assertTrue(all(np.isfinite(item["mean_return"]) for item in history))
        self.assertFalse(np.array_equal(before, policy.w_decision))


if __name__ == "__main__":
    unittest.main()
