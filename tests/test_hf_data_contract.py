import json
import unittest

from sieve.core.types import Decision, Patch, PatchOp, RevisionOutput, VerificationRequest
from sieve.data.generate import generate_toy_trajectories
from sieve.data.schema import SFTRecord
from sieve.policies.hf_data import (
    DECISION_INDEX,
    PATCH_INDEX,
    build_structured_label_plan,
    render_revision_prompt,
)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}
        self.messages: list[dict[str, str]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return f"SYSTEM:{messages[0]['content']}\nUSER:{messages[1]['content']}\nASSISTANT:"


class HFDataContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = generate_toy_trajectories(3, seed=11, max_slots=4)[0]

    def test_prompt_explicitly_disables_qwen_thinking(self) -> None:
        tokenizer = RecordingTokenizer()

        prompt = render_revision_prompt(tokenizer, self.base, enable_thinking=False)

        self.assertIn("ASSISTANT:", prompt)
        self.assertFalse(tokenizer.kwargs["enable_thinking"])
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])
        self.assertFalse(tokenizer.kwargs["tokenize"])

    def test_prompt_places_fixed_protocol_before_json_context(self) -> None:
        tokenizer = RecordingTokenizer()

        render_revision_prompt(tokenizer, self.base, enable_thinking=False)

        self.assertEqual([item["role"] for item in tokenizer.messages], ["system", "user"])
        system = tokenizer.messages[0]["content"]
        self.assertIn("UPDATE", system)
        self.assertIn("HOLD", system)
        self.assertIn("IGNORE", system)
        self.assertIn("Return valid JSON only", system)
        context = json.loads(tokenizer.messages[1]["content"])
        self.assertIn("belief_state", context)
        self.assertIn("observation", context)
        self.assertNotIn("condition", context["observation"])
        self.assertNotIn("relevant", context["observation"])
        self.assertNotIn("perturbation", context["observation"])
    def test_update_uses_per_slot_multi_label_patch_targets(self) -> None:
        field_id = self.base.context.observation.field_id
        target = RevisionOutput(
            decision=Decision.UPDATE,
            affected_fields=(field_id,),
            patches=(
                Patch(PatchOp.SET_VALUE, field_id, "new-value"),
                Patch(PatchOp.SET_PROVENANCE, field_id, "trusted-tool"),
            ),
        )
        record = SFTRecord("multi-patch", self.base.context, target)

        labels = build_structured_label_plan(record, max_slots=4)
        slot = labels.slot_field_ids.index(field_id)

        self.assertEqual(labels.decision_label, DECISION_INDEX[Decision.UPDATE])
        self.assertEqual(labels.affected_labels[slot], 1.0)
        self.assertEqual(labels.patch_operation_mask[slot], 1.0)
        self.assertEqual(
            labels.patch_operation_labels[slot][PATCH_INDEX[PatchOp.SET_VALUE]], 1.0
        )
        self.assertEqual(
            labels.patch_operation_labels[slot][PATCH_INDEX[PatchOp.SET_PROVENANCE]], 1.0
        )
        self.assertEqual(labels.verification_label, -100)

    def test_hold_supervises_verification_without_patch_operations(self) -> None:
        field_id = self.base.context.observation.field_id
        target = RevisionOutput(
            decision=Decision.HOLD,
            affected_fields=(field_id,),
            verification=VerificationRequest("lookup", field_id),
        )
        record = SFTRecord("hold", self.base.context, target)

        labels = build_structured_label_plan(record, max_slots=4)

        self.assertEqual(labels.verification_label, 1)
        self.assertFalse(any(labels.patch_operation_mask))
        self.assertEqual(sum(sum(row) for row in labels.patch_operation_labels), 0.0)

    def test_ignore_masks_affected_and_patch_branches(self) -> None:
        record = SFTRecord(
            "ignore",
            self.base.context,
            RevisionOutput(decision=Decision.IGNORE),
        )

        labels = build_structured_label_plan(record, max_slots=4)

        self.assertFalse(any(labels.affected_mask))
        self.assertFalse(any(labels.patch_operation_mask))
        self.assertEqual(labels.verification_label, -100)


if __name__ == "__main__":
    unittest.main()
