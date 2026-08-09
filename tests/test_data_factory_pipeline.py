import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from sieve.data_factory.glm import FakeGlmClient
from sieve.data_factory.pipeline import PipelineConfig, run_pipeline


class CountingFake(FakeGlmClient):
    def __init__(self) -> None:
        self.calls = 0

    def realize(self, candidates):
        self.calls += 1
        return super().realize(candidates)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        entries = []
        datasets = {
            "tau2-bench": "MIT",
            "ToolBench": "Apache-2.0",
            "AgentBench": "Apache-2.0",
            "WebArena": "Apache-2.0",
        }
        for dataset, license_name in datasets.items():
            filename = f"{dataset}.jsonl"
            with (self.root / filename).open("w", encoding="utf-8") as handle:
                for index in range(180):
                    handle.write(json.dumps({
                        "id": f"{dataset}-{index}", "domain": "retail",
                        "scenario_type": "transaction", "entity": f"order-{dataset}-{index}",
                        "field_id": "status", "old_value": "processing",
                        "new_value": f"shipped-{index}", "source": "official_api",
                        "observed_at": 200 + index, "valid_from": 200 + index,
                        "goal": "track order", "source_text": f"Grounded source {dataset} {index}"
                    }) + "\n")
            entries.append({
                "dataset": dataset, "version": "commit", "license": license_name,
                "path": filename, "format": "jsonl"
            })
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"sources": entries}), encoding="utf-8")
        self.output = self.root / "output"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_dry_run_writes_exact_preview_files(self) -> None:
        result = run_pipeline(PipelineConfig(
            self.manifest, self.output, total=140, batch_size=5, seed=7
        ), FakeGlmClient())
        self.assertEqual(result.accepted, 140)
        with (self.output / "accepted.jsonl").open(encoding="utf-8") as handle:
            self.assertEqual(sum(1 for _ in handle), 140)
        self.assertTrue((self.output / "manifest.json").exists())
        report = json.loads((self.output / "quality_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["source_counts"], {
            "tau2-bench": 70, "ToolBench": 35, "AgentBench": 21, "WebArena": 14
        })

    def test_resume_does_not_repeat_completed_requests(self) -> None:
        client = CountingFake()
        config = PipelineConfig(self.manifest, self.output, total=12, batch_size=4, seed=3)
        run_pipeline(config, client)
        first_calls = client.calls
        result = run_pipeline(config, client)
        self.assertEqual(result.accepted, 12)
        self.assertEqual(client.calls, first_calls)

    def test_distinct_transformations_may_share_a_grounded_group(self) -> None:
        for dataset in ("tau2-bench", "ToolBench", "AgentBench", "WebArena"):
            source_path = self.root / f"{dataset}.jsonl"
            rows = source_path.read_text(encoding="utf-8").splitlines()[:24]
            source_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        result = run_pipeline(
            PipelineConfig(self.manifest, self.output, total=140, batch_size=5, seed=11),
            FakeGlmClient(),
        )

        self.assertEqual(result.accepted, 140)
        records = [
            json.loads(line)
            for line in (self.output / "accepted.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        derivations = {
            (record["group_id"], record["provenance"]["transformation"])
            for record in records
        }
        self.assertEqual(len(derivations), 140)
        self.assertLess(len({record["group_id"] for record in records}), 140)
    def test_full_run_requires_exact_confirmation(self) -> None:
        with self.assertRaisesRegex(ValueError, "confirm_full"):
            run_pipeline(PipelineConfig(
                self.manifest, self.output, total=6000, confirm_full=None
            ), FakeGlmClient())

    def test_trajectory_mode_preserves_hold_successor_pairs(self) -> None:
        domains = ("retail", "airline", "database")
        entries = []
        for domain in domains:
            filename = f"trajectory-{domain}.jsonl"
            with (self.root / filename).open("w", encoding="utf-8") as handle:
                for index in range(4):
                    handle.write(json.dumps({
                        "id": f"{domain}-{index}", "domain": domain,
                        "scenario_type": "transaction", "entity": f"{domain}:{index}",
                        "field_id": "status", "old_value": "pending",
                        "new_value": f"ready-{index}", "source": "official_api",
                        "observed_at": 1000 + index, "valid_from": 1000 + index,
                        "goal": f"complete {domain} task {index}",
                        "source_text": f"Grounded {domain} record {index}"
                    }) + "\n")
            entries.append({
                "dataset": domain, "version": "commit", "license": "Apache-2.0",
                "path": filename, "format": "jsonl"
            })
        manifest = self.root / "trajectory-manifest.json"
        manifest.write_text(json.dumps({"sources": entries}), encoding="utf-8")
        output = self.root / "trajectory-output"

        result = run_pipeline(PipelineConfig(
            manifest, output, total=20, batch_size=5, seed=9,
            trajectory_mode=True,
        ), FakeGlmClient())

        self.assertEqual(result.accepted, 20)
        records = [json.loads(line) for line in (output / "accepted.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()]
        decisions = Counter(item["target"]["decision"] for item in records)
        self.assertEqual(decisions, {"UPDATE": 14, "HOLD": 3, "IGNORE": 3})
        groups = Counter(item["group_id"] for item in records)
        paired = [group for group, count in groups.items() if count == 2]
        self.assertEqual(len(paired), 3)
        for group in paired:
            pair = [item for item in records if item["group_id"] == group]
            self.assertEqual(
                [item["target"]["decision"] for item in pair],
                ["HOLD", "UPDATE"],
            )


if __name__ == "__main__":
    unittest.main()




