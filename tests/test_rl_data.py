import json
import tempfile
import unittest
from pathlib import Path

from sieve.core.types import (
    Decision,
    Patch,
    PatchOp,
    RevisionOutput,
    SlotStatus,
    VerificationRequest,
)
from sieve.environments.scenario_revision_env import ScenarioRevisionEnvironment
from sieve.rl_data.build import SCENARIO_TEMPLATES, build_rl_corpus, split_base_task
from sieve.rl_data.io import read_scenarios


def _audit_row(index: int, macro: str) -> dict:
    if macro == "commerce":
        dataset, domain = "WebArena", "shopping"
    elif macro == "service":
        dataset, domain = "tau2-bench", "telecom"
    else:
        dataset, domain = "AgentBench", "database"
    entity = f"{domain}:{index}"
    field_id = "reference_answer"
    value = f"value-{macro}-{index}"
    return {
        "record_id": f"record-{macro}-{index}",
        "group_id": f"group-{macro}-{index}",
        "provenance": {
            "source_dataset": dataset,
            "source_version": "test-version",
            "source_record_id": str(index),
            "parent_record_id": f"parent-{index}",
            "source_uri": f"local#{index}",
            "source_sha256": f"sha-{index}",
            "license": "MIT",
            "transformation": "new_consistent:update",
        },
        "taxonomy": {"domain": domain},
        "state": {
            "goal": f"Resolve task {index}",
            "observation": {
                "entity": entity,
                "field_id": field_id,
                "value": value,
                "source": "official_api",
                "observed_at": 1_750_000_000 + index,
                "valid_from": 1_750_000_000 + index,
                "source_authority": "primary_record",
                "authenticated": True,
            },
            "risk": {"risk": "medium", "reversible": True},
        },
        "observation_text": (
            f"Authenticated official_api reports {field_id} for {entity} as {value}."
        ),
        "target": {
            "decision": "UPDATE",
            "affected_fields": [field_id],
            "patches": [
                {"op": "SET_VALUE", "field_id": field_id, "value": value}
            ],
            "verification": None,
        },
        "validation": {"accepted": True, "reason_codes": []},
    }


class RLDataTests(unittest.TestCase):
    def test_split_is_stable_and_uses_base_task(self) -> None:
        first = split_base_task("tau2-bench:task-17", seed=42)
        second = split_base_task("tau2-bench:task-17", seed=42)
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "dev", "test"})

    def test_builder_creates_task_disjoint_action_conditioned_scenarios(self) -> None:
        rows = [
            _audit_row(index, macro)
            for macro in ("commerce", "service", "workflow")
            for index in range(180)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            report = build_rl_corpus(
                source,
                root / "rl",
                seed=42,
                split_counts={"train": 20, "dev": 10, "test": 10},
            )

            self.assertEqual(report["split_counts"], {"train": 20, "dev": 10, "test": 10})
            split_tasks = {}
            for split, count in report["split_counts"].items():
                scenarios = read_scenarios(root / "rl" / "scenarios" / f"{split}.jsonl")
                self.assertEqual(len(scenarios), count)
                self.assertTrue(all(len(item.initial_state.slots) >= 3 for item in scenarios))
                self.assertTrue(all(len(item.risk.dependent_fields) == 2 for item in scenarios))
                templates = {
                    str(item.provenance.get("scenario_template"))
                    for item in scenarios
                }
                self.assertTrue(templates <= set(SCENARIO_TEMPLATES))
                self.assertTrue(set(SCENARIO_TEMPLATES) <= templates)
                self.assertTrue(all(len(item.events) >= 3 for item in scenarios))
                self.assertTrue(
                    all(
                        any(event.expected_decision == "HOLD" for event in item.events)
                        for item in scenarios
                    )
                )
                self.assertTrue(
                    all(
                        all(
                            split_base_task(donor, seed=42) == split
                            for donor in item.provenance["distractor_base_task_ids"]
                        )
                        for item in scenarios
                    )
                )
                self.assertTrue(
                    all(
                        all(
                            event.verification_observation is None
                            or event.verification_tool
                            == f"verify_{event.observation.field_id}"
                            for event in item.events
                        )
                        for item in scenarios
                    )
                )
                raw = json.loads(
                    (root / "rl" / "scenarios" / f"{split}.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()[0]
                )
                self.assertNotIn("target", raw)
                self.assertIn("environment_private", raw)
                split_tasks[split] = {item.base_task_id for item in scenarios}

            self.assertFalse(split_tasks["train"] & split_tasks["dev"])
            self.assertFalse(split_tasks["train"] & split_tasks["test"])
            self.assertFalse(split_tasks["dev"] & split_tasks["test"])
            self.assertTrue((root / "rl" / "manifest.json").is_file())
            self.assertTrue((root / "rl" / "quality_report.json").is_file())
            quality = json.loads(
                (root / "rl" / "quality_report.json").read_text(encoding="utf-8")
            )
            self.assertGreater(quality["derived_secondary_scenarios"], 0)
            self.assertEqual(
                set(quality["scenario_template_counts"]["train"]),
                set(SCENARIO_TEMPLATES),
            )

    def test_structured_values_support_closed_loop_success(self) -> None:
        rows = [
            _audit_row(index, macro)
            for macro in ("commerce", "service", "workflow")
            for index in range(180)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            build_rl_corpus(
                source,
                root / "rl",
                seed=42,
                split_counts={"train": 10, "dev": 10, "test": 10},
            )
            scenario = read_scenarios(root / "rl" / "scenarios" / "dev.jsonl")[0]
            environment = ScenarioRevisionEnvironment([scenario])
            environment.reset(scenario.scenario_id, seed=42)

            for _step in range(8):
                observation = environment._current_observation()
                event = environment._base_event()
                if environment._verification_observation is not None:
                    output = RevisionOutput(
                        Decision.UPDATE,
                        (observation.field_id,),
                        (
                            Patch(
                                PatchOp.SET_VALUE,
                                observation.field_id,
                                observation.value,
                            ),
                        ),
                        None,
                    )
                elif event.expected_decision == "HOLD":
                    output = RevisionOutput(
                        Decision.HOLD,
                        (observation.field_id,),
                        (
                            Patch(
                                PatchOp.SET_STATUS,
                                observation.field_id,
                                SlotStatus.PENDING.value,
                            ),
                        ),
                        VerificationRequest(event.verification_tool, observation.field_id),
                    )
                elif event.expected_decision == "UPDATE":
                    output = RevisionOutput(
                        Decision.UPDATE,
                        (observation.field_id,),
                        (
                            Patch(
                                PatchOp.SET_VALUE,
                                observation.field_id,
                                observation.value,
                            ),
                        ),
                        None,
                    )
                else:
                    output = RevisionOutput(Decision.IGNORE)
                transition = environment.step(output)
                if transition.terminated:
                    self.assertTrue(transition.info["success"])
                    break
            else:
                self.fail("ideal Stage-2 policy did not terminate")

    def test_builder_groups_fields_and_uses_related_distractors(self) -> None:
        rows = []
        for macro in ("commerce", "service", "workflow"):
            for index in range(180):
                for field_id in ("primary_field", "secondary_field"):
                    row = _audit_row(index, macro)
                    dataset = row["provenance"]["source_dataset"]
                    domain = row["taxonomy"]["domain"]
                    row["record_id"] = f"record-{macro}-{index}-{field_id}"
                    row["provenance"]["parent_record_id"] = (
                        f"{dataset}:{domain}:{index}:{field_id}"
                    )
                    row["provenance"]["source_record_id"] = (
                        f"{domain}:{index}:{field_id}"
                    )
                    row["provenance"]["source_uri"] = (
                        f"C:/private/cache/{macro}/{index}#{field_id}"
                    )
                    row["state"]["observation"]["field_id"] = field_id
                    value = f"{macro}-{index}-{field_id}"
                    row["state"]["observation"]["value"] = value
                    row["target"]["affected_fields"] = [field_id]
                    row["target"]["patches"] = [
                        {"op": "SET_VALUE", "field_id": field_id, "value": value}
                    ]
                    rows.append(row)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            build_rl_corpus(
                source,
                root / "rl",
                seed=42,
                split_counts={"train": 10, "dev": 10, "test": 10},
            )
            scenarios = read_scenarios(root / "rl" / "scenarios" / "train.jsonl")
            for scenario in scenarios:
                self.assertEqual(len(scenario.risk.dependent_fields), 2)
                self.assertIn(
                    scenario.provenance["scenario_template"], SCENARIO_TEMPLATES
                )
                self.assertFalse(scenario.base_task_id.startswith("WebArena:WebArena:"))
                self.assertNotRegex(str(scenario.provenance["source_uri"]), r"^[A-Za-z]:")


if __name__ == "__main__":
    unittest.main()
