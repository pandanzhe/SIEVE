import unittest

from sieve.data_factory.quotas import (
    count_by,
    full_quota_cells,
    preview_quota_cells,
)


class QuotaTests(unittest.TestCase):
    def test_preview_matrix_matches_approved_margins(self) -> None:
        cells = preview_quota_cells()
        self.assertEqual(sum(cell.count for cell in cells), 140)
        self.assertEqual(
            count_by(cells, "decision"),
            {"UPDATE": 56, "IGNORE": 49, "HOLD": 35},
        )
        self.assertEqual(
            count_by(cells, "observation_type"),
            {
                "new_consistent": 21,
                "explicit_conflict": 21,
                "implicit_conflict": 21,
                "stale": 21,
                "irrelevant": 21,
                "tool_error_or_low_trust": 21,
                "insufficient_or_ambiguous": 14,
            },
        )

    def test_full_quota_has_six_thousand_records(self) -> None:
        cells = full_quota_cells()
        self.assertEqual(sum(cell.count for cell in cells), 6000)
        self.assertEqual(
            count_by(cells, "observation_type"),
            {
                "new_consistent": 900,
                "explicit_conflict": 900,
                "implicit_conflict": 900,
                "stale": 900,
                "irrelevant": 900,
                "tool_error_or_low_trust": 900,
                "insufficient_or_ambiguous": 600,
            },
        )
        self.assertEqual(
            count_by(cells, "decision"),
            {"UPDATE": 4200, "IGNORE": 900, "HOLD": 900},
        )


if __name__ == "__main__":
    unittest.main()
