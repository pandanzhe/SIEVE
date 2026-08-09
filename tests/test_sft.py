import unittest

from sieve.data.generate import generate_toy_trajectories
from sieve.policies.structured_policy import StructuredPolicy
from sieve.training.sft import evaluate_sft, train_sft


class SFTTests(unittest.TestCase):
    def test_training_lowers_loss_and_improves_accuracy(self) -> None:
        records = generate_toy_trajectories(80, seed=5, max_slots=4)
        policy = StructuredPolicy(128, 4, seed=5)
        before = evaluate_sft(policy, records)
        history = train_sft(
            policy,
            records,
            epochs=30,
            learning_rate=0.08,
            seed=5,
            class_weights=(1.0, 1.2, 1.0),
        )
        after = evaluate_sft(policy, records)
        self.assertLess(history[-1], history[0])
        self.assertLess(after["loss"], before["loss"])
        self.assertGreater(after["decision_accuracy"], 0.90)


if __name__ == "__main__":
    unittest.main()
