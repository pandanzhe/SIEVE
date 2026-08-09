import json
import tempfile
import unittest
from pathlib import Path

from sieve.data_factory.normalize import (
    normalize_agentbench,
    normalize_tau2,
    normalize_toolbench,
    normalize_webarena,
)


class NormalizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_tau2_task_becomes_grounded_record(self) -> None:
        path = self.root / "tasks.json"
        path.write_text(json.dumps([{
            "id": "7",
            "user_scenario": {"instructions": {
                "domain": "retail",
                "reason_for_call": "Track order W7",
                "known_info": "Order W7 belongs to Ana"
            }},
            "evaluation_criteria": {"actions": [{
                "name": "get_order_details",
                "arguments": {"order_id": "W7", "status": "shipped"}
            }]}
        }]), encoding="utf-8")
        record = normalize_tau2(path, "commit-tau", "MIT")[0]
        self.assertEqual(record.provenance.source_dataset, "tau2-bench")
        self.assertEqual(record.entity, "W7")
        self.assertEqual(record.new_value, "shipped")

    def test_toolbench_observation_becomes_tool_record(self) -> None:
        path = self.root / "answer.json"
        path.write_text(json.dumps({"tree": {"tree": {
            "node_type": "Action", "description": "weather_api",
            "children": [{"node_type": "Action Input", "observation":
                '{"error":"","response":{"city":"Paris","temp":18}}'}]
        }}}), encoding="utf-8")
        record = normalize_toolbench([path], "commit-tool", "Apache-2.0")[0]
        self.assertEqual(record.provenance.source_dataset, "ToolBench")
        self.assertEqual(record.source, "weather_api")
        self.assertIn("Paris", str(record.new_value))

    def test_agentbench_table_becomes_database_record(self) -> None:
        path = self.root / "dev.jsonl"
        path.write_text(json.dumps({
            "description": "What is the current status?", "label": ["active"],
            "table": {"table_name": "accounts", "table_info": {
                "columns": [{"name": "status", "type": "TEXT"}],
                "rows": [["pending"]]
            }}
        }) + "\n", encoding="utf-8")
        record = normalize_agentbench(path, "commit-agent", "Apache-2.0")[0]
        self.assertEqual(record.entity, "accounts")
        self.assertEqual(record.old_value, "pending")
        self.assertEqual(record.new_value, "active")

    def test_webarena_task_becomes_web_record(self) -> None:
        path = self.root / "test.raw.json"
        path.write_text(json.dumps([{
            "task_id": 3, "sites": ["shopping"],
            "intent": "Find the best seller",
            "eval": {"reference_answers": {"exact_match": "Product A"}}
        }]), encoding="utf-8")
        record = normalize_webarena(path, "commit-web", "MIT")[0]
        self.assertEqual(record.provenance.source_dataset, "WebArena")
        self.assertEqual(record.new_value, "Product A")
        self.assertEqual(record.field_id, "reference_answer")


    def test_webarena_order_uses_business_entity_and_field(self) -> None:
        path = self.root / "test.raw.json"
        path.write_text(json.dumps([{
            "task_id": 362,
            "sites": ["shopping"],
            "intent": "Show me the billing address for order number 00178.",
            "instantiation_dict": {
                "info": "billing address",
                "order_number": "00178",
            },
            "eval": {"reference_answers": {
                "must_include": ["101 S San Mateo Dr"]
            }},
        }]), encoding="utf-8")

        record = normalize_webarena(path, "commit-web", "Apache-2.0")[0]

        self.assertEqual(record.entity, "order:00178")
        self.assertEqual(record.field_id, "billing_address")
        self.assertEqual(record.metadata["instantiation_dict"]["order_number"], "00178")
if __name__ == "__main__":
    unittest.main()
