import json
import tempfile
import unittest
from pathlib import Path

from sieve.data_factory.publish import publish_canonical_sft
from sieve.policies.hf_data import SIEVE_SYSTEM_PROMPT


class PublishTests(unittest.TestCase):
    def test_publish_copies_only_source_train_and_dev_with_prompt_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "accepted.jsonl"
            splits = root / "training"
            output = root / "sft"
            quality_report = root / "quality_report.json"
            splits.mkdir()
            source.write_text('{"record_id":"source-1"}\n', encoding="utf-8")
            quality_report.write_text('{"accepted":1}', encoding="utf-8")
            row = {
                "scenario_id": "episode-1",
                "step_index": 0,
                "context": {},
                "target": {},
            }
            for name in ("train", "dev", "test"):
                (splits / f"{name}.jsonl").write_text(
                    json.dumps(row) + "\n", encoding="utf-8"
                )

            manifest = publish_canonical_sft(
                source,
                splits,
                output,
                prompt_version="stage1-system-v2",
                quality_report_path=quality_report,
            )

            self.assertTrue((output / "source" / "records.jsonl").is_file())
            self.assertTrue((output / "clean" / "train.jsonl").is_file())
            self.assertTrue((output / "clean" / "dev.jsonl").is_file())
            self.assertFalse((output / "clean" / "test.jsonl").exists())
            self.assertTrue((output / "quality_report.json").is_file())
            self.assertEqual(manifest["system_prompt"]["text"], SIEVE_SYSTEM_PROMPT)
            self.assertEqual(manifest["train"]["records"], 1)
            self.assertEqual(manifest["validation"]["records"], 1)


if __name__ == "__main__":
    unittest.main()
