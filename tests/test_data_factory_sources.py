import json
import tempfile
import unittest
from pathlib import Path

from sieve.data_factory.sources import load_source_manifest


class SourceAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        source = root / "tau2.jsonl"
        source.write_text(
            json.dumps(
                {
                    "id": "retail-001",
                    "domain": "retail",
                    "scenario_type": "transaction",
                    "entity": "order-001",
                    "field_id": "status",
                    "old_value": "processing",
                    "new_value": "shipped",
                    "source": "official_order_api",
                    "observed_at": 200,
                    "valid_from": 200,
                    "goal": "track order",
                    "source_text": "The order status is shipped.",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        self.manifest_path = root / "manifest.json"
        self.manifest_path.write_text(
            json.dumps(
                {
                    "sources": [
                        {
                            "dataset": "tau2-bench",
                            "version": "commit-abc",
                            "license": "MIT",
                            "path": "tau2.jsonl",
                            "format": "jsonl",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.root = root

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_jsonl_adapter_preserves_traceable_identity(self) -> None:
        records = load_source_manifest(self.manifest_path)
        record = records[0]
        self.assertEqual(record.provenance.source_dataset, "tau2-bench")
        self.assertEqual(record.provenance.source_record_id, "retail-001")
        self.assertEqual(len(record.provenance.source_sha256), 64)
        self.assertEqual(record.scenario_type.value, "transaction")

    def test_missing_license_is_rejected(self) -> None:
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        del raw["sources"][0]["license"]
        broken = self.root / "missing-license.json"
        broken.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "license"):
            load_source_manifest(broken)


if __name__ == "__main__":
    unittest.main()
