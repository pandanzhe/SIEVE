import json
import unittest
from dataclasses import fields

from sieve.core.types import Observation
from sieve.data.generate import generate_toy_trajectories, split_by_scenario
from sieve.data.schema import context_summary


class DataTests(unittest.TestCase):
    def test_observation_exposes_runtime_source_metadata(self) -> None:
        names = {item.name for item in fields(Observation)}

        self.assertIn("source_authority", names)
        self.assertIn("authenticated", names)

    def test_policy_context_is_json_without_generation_labels(self) -> None:
        record = generate_toy_trajectories(scenarios=1, seed=11, max_slots=4)[0]

        rendered = context_summary(record.context)

        self.assertTrue(rendered.startswith("{"))
        payload = json.loads(rendered)
        observation = payload["observation"]
        self.assertNotIn("condition", observation)
        self.assertNotIn("relevant", observation)
        self.assertNotIn("perturbation", observation)
    def test_paired_generation_contains_all_decisions(self) -> None:
        records = generate_toy_trajectories(scenarios=6, seed=11, max_slots=4)
        decisions = {record.target.decision.value for record in records}
        self.assertEqual(decisions, {"UPDATE", "HOLD", "IGNORE"})
        self.assertEqual(len(records), 18)

    def test_scenario_groups_do_not_cross_splits(self) -> None:
        records = generate_toy_trajectories(scenarios=20, seed=7, max_slots=4)
        splits = split_by_scenario(records, 0.7, 0.15, seed=7)
        groups = {name: {r.scenario_id for r in items} for name, items in splits.items()}
        self.assertFalse(groups["train"] & groups["dev"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["dev"] & groups["test"])


if __name__ == "__main__":
    unittest.main()
