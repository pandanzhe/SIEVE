import unittest

from sieve.data.generate import generate_toy_trajectories, split_by_scenario
from sieve.data.validation import validate_records, validate_split_isolation


class ValidationTests(unittest.TestCase):
    def test_generated_records_respect_label_contracts(self) -> None:
        records = generate_toy_trajectories(12, seed=3, max_slots=4)
        self.assertEqual(validate_records(records), [])

    def test_group_split_has_no_leakage(self) -> None:
        records = generate_toy_trajectories(12, seed=3, max_slots=4)
        splits = split_by_scenario(records, 0.7, 0.15, seed=3)
        self.assertEqual(validate_split_isolation(splits), [])


if __name__ == "__main__":
    unittest.main()
