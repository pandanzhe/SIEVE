import unittest

from sieve.core.types import Decision
from sieve.policies.hf_revision_io import (
    assess_stage1_readiness,
    parse_revision_output,
    render_context_prompt,
    safe_parse_revision_output,
)
from sieve.rl_data.schema import RLScenario


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "\n".join(f"{item['role']}:{item['content']}" for item in messages)


class HFRevisionIOTests(unittest.TestCase):
    def test_parser_accepts_exact_json_and_enforces_decision_gates(self) -> None:
        output = parse_revision_output(
            '{"decision":"UPDATE","affected_fields":["address"],'
            '"patches":[{"op":"SET_VALUE","field_id":"address","value":"Seoul"}],'
            '"verification":null}'
        )
        self.assertEqual(output.decision, Decision.UPDATE)
        self.assertEqual(output.patches[0].value, "Seoul")
        with self.assertRaises(ValueError):
            parse_revision_output(
                '{"decision":"IGNORE","affected_fields":["address"],'
                '"patches":[],"verification":null}'
            )
        with self.assertRaises(ValueError):
            parse_revision_output(
                '{"decision":"IGNORE","affected_fields":[],"patches":[],'
                '"verification":null} trailing prose'
            )

    def test_safe_parser_returns_non_mutating_fallback(self) -> None:
        output, valid, error = safe_parse_revision_output("not json")
        self.assertEqual(output.decision, Decision.IGNORE)
        self.assertFalse(valid)
        self.assertIsNotNone(error)

    def test_parser_rejects_non_rfc_numeric_constants(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-standard JSON constant"):
            parse_revision_output(
                '{"decision":"UPDATE","affected_fields":["score"],'
                '"patches":[{"op":"SET_VALUE","field_id":"score","value":NaN}],'
                '"verification":null}'
            )

    def test_parser_rejects_duplicate_object_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            parse_revision_output(
                '{"decision":"IGNORE","decision":"UPDATE",'
                '"affected_fields":[],"patches":[],"verification":null}'
            )

    def test_prompt_contains_only_policy_visible_context(self) -> None:
        scenario = RLScenario.from_dict(
            {
                "scenario_id": "s1",
                "base_task_id": "b1",
                "variant_id": 0,
                "split": "train",
                "domain": "airline",
                "macro_domain": "service",
                "provenance": {},
                "initial_context": {
                    "belief_state": {
                        "max_slots": 3,
                        "slots": [
                            {
                                "id": "seat",
                                "value": None,
                                "status": "empty",
                                "source": "initial",
                                "observed_at": 1,
                                "valid_from": None,
                                "entity": "booking:1",
                            }
                        ],
                    },
                    "goal": "Change the seat",
                    "risk": {
                        "active_subgoal": "Change the seat",
                        "dependent_fields": ["seat"],
                        "risk": "medium",
                        "reversible": True,
                    },
                    "budget": {
                        "verification_remaining": 1,
                        "tool_remaining": 1,
                        "steps_remaining": 4,
                        "tokens_remaining": 100,
                    },
                    "ledger": {"capacity": 2, "entries": []},
                },
                "environment_private": {
                    "oracle_state": {"seat": "12A"},
                    "events": [
                        {
                            "event_id": "e1",
                            "kind": "authoritative",
                            "expected_decision": "UPDATE",
                            "verification_observation": None,
                            "observation": {
                                "field_id": "seat",
                                "value": "official says 12A",
                                "source": "official",
                                "observed_at": 2,
                                "valid_from": 2,
                                "entity": "booking:1",
                                "condition": "authoritative",
                                "relevant": True,
                                "perturbation": "clean",
                                "source_authority": "primary_record",
                                "authenticated": True,
                            },
                        }
                    ],
                },
            }
        )
        from sieve.environments.scenario_revision_env import ScenarioRevisionEnvironment

        environment = ScenarioRevisionEnvironment([scenario])
        context = environment.reset("s1", seed=1)
        prompt = render_context_prompt(_Tokenizer(), context)
        self.assertNotIn("oracle_state", prompt)
        self.assertNotIn("expected_decision", prompt)
        self.assertIn("official says 12A", prompt)

    def test_readiness_gate_requires_all_generation_metrics(self) -> None:
        ready, failures = assess_stage1_readiness(
            {
                "parse_rate": 0.99,
                "executable_rate": 0.98,
                "decision_macro_f1": 0.93,
                "full_action_exact_match": 0.88,
                "patch_value_exact_match": 0.90,
                "false_update_rate": 0.02,
                "group_reward_variance": 0.01,
                "closed_loop_success_rate": 0.70,
                "closed_loop_parse_rate": 0.98,
            }
        )
        self.assertTrue(ready)
        self.assertEqual(failures, {})
        ready, failures = assess_stage1_readiness(
            {
                "parse_rate": 0.90,
                "executable_rate": 0.98,
                "decision_macro_f1": 0.93,
                "full_action_exact_match": 0.88,
                "patch_value_exact_match": 0.90,
                "false_update_rate": 0.02,
                "group_reward_variance": 0.01,
                "closed_loop_success_rate": 0.70,
                "closed_loop_parse_rate": 0.98,
            }
        )
        self.assertFalse(ready)
        self.assertIn("parse_rate", failures)


if __name__ == "__main__":
    unittest.main()
