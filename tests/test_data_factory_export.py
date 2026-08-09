import json
import tempfile
import unittest
from pathlib import Path

from sieve.data.io import record_from_dict
from sieve.data_factory.export import audit_to_sft_dict, export_training_splits


def audit_record(record_id: str, group_id: str, text: str) -> dict:
    return {
        "record_id": record_id,
        "group_id": group_id,
        "provenance": {"source_dataset": "tau2-bench"},
        "taxonomy": {"observation_type": "new_consistent"},
        "observation_text": text,
        "state": {
            "belief_state": {
                "max_slots": 4,
                "slots": [{
                    "id": "status", "value": "processing", "status": "trusted",
                    "source": "prior", "observed_at": 10, "valid_from": 10,
                    "entity": "order-1"
                }]
            },
            "observation": {
                "field_id": "status", "value": "shipped", "source": "official_api",
                "observed_at": 20, "valid_from": 20, "entity": "order-1",
                "condition": "applies", "relevant": True,
                "perturbation": "new_consistent"
            },
            "goal": "track order",
            "risk": {
                "active_subgoal": "track order", "dependent_fields": ["status"],
                "risk": "medium", "reversible": True
            },
            "budget": {
                "verification_remaining": 2, "tool_remaining": 5,
                "steps_remaining": 8, "tokens_remaining": 4096
            },
            "ledger": {"capacity": 4, "entries": []}
        },
        "target": {
            "decision": "UPDATE", "affected_fields": ["status"],
            "patches": [{"op": "SET_VALUE", "field_id": "status", "value": "shipped"}],
            "verification": None, "reason_code": "CONSISTENT_EVIDENCE",
            "conflict_type": None, "verification_action": "NO_VERIFY"
        },
        "oracle": {"target_belief_state": {}},
        "generation": {"generator": "fake-glm"},
        "validation": {"accepted": True, "reason_codes": []}
    }


class ExportTests(unittest.TestCase):
    def test_training_view_removes_audit_fields_and_uses_realized_text(self) -> None:
        raw = audit_record("r1", "group-a", "Official API says order-1 is shipped.")
        exported = audit_to_sft_dict(raw, step_index=0)

        self.assertEqual(set(exported), {"scenario_id", "step_index", "context", "target"})
        self.assertEqual(exported["context"]["observation"]["value"], raw["observation_text"])
        observation = exported["context"]["observation"]
        self.assertNotIn("condition", observation)
        self.assertNotIn("relevant", observation)
        self.assertNotIn("perturbation", observation)
        self.assertNotIn("reason_code", exported["target"])
        loaded = record_from_dict(exported)
        self.assertEqual(loaded.target.decision.value, "UPDATE")

    def test_export_keeps_each_group_in_one_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "accepted.jsonl"
            records = [
                audit_record("r1", "group-a", "order-1 shipped from source a"),
                audit_record("r2", "group-a", "order-1 remains shipped from source a"),
                audit_record("r3", "group-b", "order-1 shipped from source b"),
                audit_record("r4", "group-c", "order-1 shipped from source c"),
            ]
            source.write_text(
                "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
            )

            counts = export_training_splits(source, root / "training", seed=42)

            self.assertEqual(sum(counts.values()), 4)
            locations: dict[str, set[str]] = {}
            for split in ("train", "dev", "test"):
                path = root / "training" / f"{split}.jsonl"
                for line in path.read_text(encoding="utf-8").splitlines():
                    item = json.loads(line)
                    record_from_dict(item)
                    locations.setdefault(item["scenario_id"], set()).add(split)
            self.assertTrue(all(len(splits) == 1 for splits in locations.values()))


if __name__ == "__main__":
    unittest.main()
