import json
import tempfile
import unittest
from pathlib import Path

from sieve.core.types import (
    Decision,
    Patch,
    PatchOp,
    RevisionOutput,
    VerificationRequest,
)
from sieve.data.schema import context_summary
from sieve.environments.scenario_revision_env import ScenarioRevisionEnvironment
from sieve.rl_data.build import build_rl_corpus
from sieve.rl_data.io import read_scenarios

from tests.test_rl_data import _audit_row


class ScenarioRevisionEnvironmentTests(unittest.TestCase):
    def _scenario(self, root: Path):
        rows = [
            _audit_row(index, macro)
            for macro in ("commerce", "service", "workflow")
            for index in range(120)
        ]
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
        return read_scenarios(root / "rl" / "scenarios" / "train.jsonl")[0]

    def test_hold_with_verification_enters_conditional_micro_transition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scenario = self._scenario(Path(directory))
            environment = ScenarioRevisionEnvironment([scenario])
            context = environment.reset(scenario.scenario_id, seed=7)
            field_id = context.observation.field_id
            verification_tool = scenario.events[0].verification_tool

            step = environment.step(
                RevisionOutput(
                    decision=Decision.HOLD,
                    affected_fields=(field_id,),
                    verification=VerificationRequest(
                        tool=verification_tool, field_id=field_id
                    ),
                )
            )

            self.assertFalse(step.terminated)
            self.assertEqual(step.info["transition"], "verification")
            self.assertEqual(step.next_context.observation.source, "official_verification_api")
            self.assertEqual(step.next_context.budget.verification_remaining, 1)
            self.assertEqual(step.costs["verification"], 1.0)
            self.assertNotIn("oracle_state", step.info)
            self.assertNotIn("source_ambiguity", context_summary(step.next_context))

    def test_verification_tool_is_visible_and_distinct_from_evidence_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scenario = self._scenario(Path(directory))
            event = scenario.events[0]
            policy_slot = scenario.initial_state.get("__verification_policy")

            self.assertIsNotNone(event.verification_tool)
            self.assertNotEqual(
                event.verification_tool, event.verification_observation.source
            )
            self.assertEqual(
                policy_slot.value["tool_by_field"][event.observation.field_id],
                event.verification_tool,
            )

    def test_wrong_update_changes_state_and_incurs_constraint_cost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scenario = self._scenario(Path(directory))
            environment = ScenarioRevisionEnvironment([scenario])
            context = environment.reset(scenario.scenario_id, seed=7)
            field_id = context.observation.field_id

            step = environment.step(
                RevisionOutput(
                    decision=Decision.UPDATE,
                    affected_fields=(field_id,),
                    patches=(Patch(PatchOp.SET_VALUE, field_id, "contaminated"),),
                )
            )

            self.assertEqual(step.costs["false_update"], 1.0)
            self.assertEqual(step.next_context.belief_state.get(field_id).value, "contaminated")
            self.assertEqual(step.next_context.observation.condition, "outside_active_entity")

    def test_invalid_format_cost_is_recorded_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scenario = self._scenario(Path(directory))
            environment = ScenarioRevisionEnvironment([scenario])
            environment.reset(scenario.scenario_id, seed=7)
            step = environment.step(
                RevisionOutput(decision=Decision.IGNORE), invalid_format=True
            )
            self.assertEqual(step.costs["invalid_format"], 1.0)

    def test_completion_tokens_decrement_visible_token_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scenario = self._scenario(Path(directory))
            environment = ScenarioRevisionEnvironment([scenario])
            context = environment.reset(scenario.scenario_id, seed=7)

            step = environment.step(
                RevisionOutput(decision=Decision.IGNORE), token_cost=17
            )

            self.assertEqual(
                step.next_context.budget.tokens_remaining,
                context.budget.tokens_remaining - 17,
            )

    def test_unknown_verification_tool_cannot_open_private_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scenario = self._scenario(Path(directory))
            environment = ScenarioRevisionEnvironment([scenario])
            context = environment.reset(scenario.scenario_id, seed=7)
            field_id = context.observation.field_id
            step = environment.step(
                RevisionOutput(
                    decision=Decision.HOLD,
                    affected_fields=(field_id,),
                    verification=VerificationRequest(
                        tool="totally_fake_tool", field_id=field_id
                    ),
                )
            )
            self.assertEqual(step.info["transition"], "next_event")
            self.assertEqual(step.costs["verification"], 0.0)
            self.assertEqual(step.costs["invalid_patch"], 1.0)
            self.assertEqual(step.next_context.budget.verification_remaining, 2)
            self.assertNotEqual(
                step.next_context.observation.source, "official_verification_api"
            )

    def test_wrong_field_cannot_open_verification_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scenario = self._scenario(Path(directory))
            environment = ScenarioRevisionEnvironment([scenario])
            context = environment.reset(scenario.scenario_id, seed=7)
            other_field = next(
                field
                for field in scenario.risk.dependent_fields
                if field != context.observation.field_id
            )
            event = scenario.events[0]
            step = environment.step(
                RevisionOutput(
                    decision=Decision.HOLD,
                    affected_fields=(other_field,),
                    verification=VerificationRequest(
                        tool=event.verification_tool, field_id=other_field
                    ),
                )
            )
            self.assertEqual(step.info["transition"], "next_event")
            self.assertEqual(step.costs["verification"], 0.0)
            self.assertEqual(step.costs["invalid_patch"], 1.0)
            self.assertEqual(step.next_context.budget.verification_remaining, 2)


if __name__ == "__main__":
    unittest.main()
