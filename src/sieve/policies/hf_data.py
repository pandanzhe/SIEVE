from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..core.types import Decision, PatchOp
from ..data.schema import SFTRecord, context_summary


DECISION_INDEX = {Decision.UPDATE: 0, Decision.HOLD: 1, Decision.IGNORE: 2}
PATCH_INDEX = {operation: index for index, operation in enumerate(PatchOp)}

SIEVE_SYSTEM_PROMPT = """You are the observation evaluation policy of a multi-step agent.

Given the current executable belief state, task goal, new observation, risk constraints,
evidence ledger, and remaining budget, produce exactly one structured cognitive action.

Decision definitions:
- UPDATE: the observation applies to the current task and has sufficient support to modify the executable belief state.
- HOLD: the observation may affect the current task, but current evidence is insufficient for safe commitment.
- IGNORE: the observation has no admissible effect on the current task state and does not require verification.

Constraints:
1. UPDATE must contain local executable patches.
2. HOLD must preserve the current executable value. It may mark an affected field as pending and request verification.
3. IGNORE must contain no affected fields, patches, or verification request.
4. Do not modify fields outside affected_fields.
5. Return valid JSON only, without explanations."""


@dataclass(frozen=True)
class StructuredLabelPlan:
    decision_label: int
    slot_field_ids: tuple[str, ...]
    slot_mask: tuple[float, ...]
    affected_labels: tuple[float, ...]
    affected_mask: tuple[float, ...]
    verification_label: int
    patch_operation_labels: tuple[tuple[float, ...], ...]
    patch_operation_mask: tuple[float, ...]


def _slot_field_ids(record: SFTRecord, max_slots: int) -> tuple[str, ...]:
    fields = [slot.id for slot in record.context.belief_state.slots[:max_slots]]
    observation_field = record.context.observation.field_id
    if observation_field not in fields and len(fields) < max_slots:
        fields.append(observation_field)
    return tuple(fields)


def build_structured_label_plan(record: SFTRecord, max_slots: int) -> StructuredLabelPlan:
    """Build labels from a target-independent, deployment-visible slot registry."""
    if max_slots <= 0:
        raise ValueError("max_slots must be positive")
    fields = _slot_field_ids(record, max_slots)
    field_to_slot = {field_id: index for index, field_id in enumerate(fields)}
    slot_mask = [1.0 if index < len(fields) else 0.0 for index in range(max_slots)]
    affected = [0.0] * max_slots
    patch_operations = [[0.0] * len(PATCH_INDEX) for _ in range(max_slots)]

    for field_id in record.target.affected_fields:
        if field_id not in field_to_slot:
            raise ValueError(f"affected field cannot be represented in slot registry: {field_id}")
        affected[field_to_slot[field_id]] = 1.0
    for patch in record.target.patches:
        if patch.field_id not in field_to_slot:
            raise ValueError(
                f"patch field cannot be represented in slot registry: {patch.field_id}"
            )
        patch_operations[field_to_slot[patch.field_id]][PATCH_INDEX[patch.op]] = 1.0

    decision = record.target.decision
    affected_mask = slot_mask if decision is not Decision.IGNORE else [0.0] * max_slots
    patch_mask = slot_mask if decision is Decision.UPDATE else [0.0] * max_slots
    verification_label = (
        int(record.target.verification is not None) if decision is Decision.HOLD else -100
    )
    return StructuredLabelPlan(
        decision_label=DECISION_INDEX[decision],
        slot_field_ids=fields,
        slot_mask=tuple(slot_mask),
        affected_labels=tuple(affected),
        affected_mask=tuple(affected_mask),
        verification_label=verification_label,
        patch_operation_labels=tuple(tuple(row) for row in patch_operations),
        patch_operation_mask=tuple(patch_mask),
    )


def render_revision_prompt(
    tokenizer: Any,
    record: SFTRecord,
    *,
    enable_thinking: bool = False,
) -> str:
    messages = [
        {"role": "system", "content": SIEVE_SYSTEM_PROMPT},
        {"role": "user", "content": context_summary(record.context)},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking")
            return tokenizer.apply_chat_template(messages, **kwargs)
    return (
        f"<|sieve_system|>\n{SIEVE_SYSTEM_PROMPT}\n<|sieve_state|>\n"
        f"{context_summary(record.context)}\n<|sieve_action|>\n"
    )


class HFRevisionCollator:
    """Build causal-LM and gated structured labels with lazy torch imports."""

    def __init__(
        self,
        tokenizer: Any,
        max_slots: int,
        max_length: int = 2048,
        max_completion_length: int = 512,
        enable_thinking: bool = False,
    ) -> None:
        if max_slots <= 0:
            raise ValueError("max_slots must be positive")
        if max_length < 32:
            raise ValueError("max_length must be at least 32")
        if not 1 <= max_completion_length < max_length:
            raise ValueError("max_completion_length must be within the sequence length")
        self.tokenizer = tokenizer
        self.max_slots = max_slots
        self.max_length = max_length
        self.max_completion_length = max_completion_length
        self.enable_thinking = enable_thinking

    @staticmethod
    def _target_json(record: SFTRecord) -> str:
        target = record.target
        payload = {
            "decision": target.decision.value,
            "affected_fields": list(target.affected_fields),
            "patches": [
                {"op": patch.op.value, "field_id": patch.field_id, "value": patch.value}
                for patch in target.patches
            ],
            "verification": (
                {"tool": target.verification.tool, "field_id": target.verification.field_id}
                if target.verification
                else None
            ),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def __call__(self, records: Sequence[SFTRecord]) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("HF collation requires the 'hf' optional dependencies") from exc
        if not records:
            raise ValueError("HF collation requires at least one record")
        if self.tokenizer.eos_token is None:
            raise ValueError("tokenizer must define an eos_token")

        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        prompt_last_indices: list[int] = []
        plans: list[StructuredLabelPlan] = []
        for record in records:
            prompt = render_revision_prompt(
                self.tokenizer,
                record,
                enable_thinking=self.enable_thinking,
            )
            completion = self._target_json(record) + self.tokenizer.eos_token
            completion_ids = self.tokenizer(
                completion,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_completion_length,
            )["input_ids"]
            prompt_ids = self.tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length - len(completion_ids),
            )["input_ids"]
            if not prompt_ids:
                raise ValueError("tokenized revision prompt must not be empty")
            input_rows.append(prompt_ids + completion_ids)
            label_rows.append([-100] * len(prompt_ids) + completion_ids)
            prompt_last_indices.append(len(prompt_ids) - 1)
            plans.append(build_structured_label_plan(record, self.max_slots))

        padded = self.tokenizer.pad(
            [{"input_ids": row} for row in input_rows],
            padding=True,
            return_tensors="pt",
        )
        labels = torch.full_like(padded["input_ids"], -100)
        pool_indices: list[int] = []
        for row_index, row_labels in enumerate(label_rows):
            left_offset = (
                labels.size(1) - len(row_labels)
                if self.tokenizer.padding_side == "left"
                else 0
            )
            labels[row_index, left_offset : left_offset + len(row_labels)] = torch.tensor(
                row_labels, dtype=torch.long
            )
            pool_indices.append(prompt_last_indices[row_index] + left_offset)

        return {
            **padded,
            "lm_labels": labels,
            "pool_indices": torch.tensor(pool_indices, dtype=torch.long),
            "decision_labels": torch.tensor(
                [plan.decision_label for plan in plans], dtype=torch.long
            ),
            "slot_mask": torch.tensor([plan.slot_mask for plan in plans], dtype=torch.bool),
            "affected_labels": torch.tensor(
                [plan.affected_labels for plan in plans], dtype=torch.float32
            ),
            "affected_mask": torch.tensor(
                [plan.affected_mask for plan in plans], dtype=torch.bool
            ),
            "verification_labels": torch.tensor(
                [plan.verification_label for plan in plans], dtype=torch.long
            ),
            "patch_operation_labels": torch.tensor(
                [plan.patch_operation_labels for plan in plans], dtype=torch.float32
            ),
            "patch_operation_mask": torch.tensor(
                [plan.patch_operation_mask for plan in plans], dtype=torch.bool
            ),
        }

