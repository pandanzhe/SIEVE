import unittest

from sieve.data.generate import generate_toy_trajectories
from sieve.environments.revision_env import ReplayRevisionEnvironment


class ReplayRevisionEnvironmentTests(unittest.TestCase):
    def test_events_follow_explicit_step_index_and_budget_is_snapshotted(self) -> None:
        records = generate_toy_trajectories(scenarios=1, seed=5, max_slots=4)
        environment = ReplayRevisionEnvironment(records)

        context = environment.reset("scenario-00000", seed=5)
        self.assertEqual(context.observation.perturbation, records[0].context.observation.perturbation)
        first_budget = context.budget.steps_remaining

        first = environment.step(records[0].target)
        self.assertFalse(first.terminated)
        self.assertEqual(context.budget.steps_remaining, first_budget)
        self.assertIsNotNone(first.next_context)
        self.assertEqual(first.next_context.observation.source, "cached_page")

        second = environment.step(records[1].target)
        self.assertFalse(second.terminated)
        self.assertIsNotNone(second.next_context)
        self.assertEqual(second.next_context.observation.source, "official_payment_api")

        final = environment.step(records[2].target)
        self.assertTrue(final.terminated)
        self.assertIsNone(final.next_context)
        self.assertTrue(final.info["success"])

    def test_reset_rejects_unknown_scenario(self) -> None:
        environment = ReplayRevisionEnvironment(
            generate_toy_trajectories(scenarios=1, seed=5, max_slots=4)
        )
        with self.assertRaises(KeyError):
            environment.reset("missing", seed=5)


if __name__ == "__main__":
    unittest.main()
