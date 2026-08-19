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


def _identify_value_byte_spans(completion_text: str) -> list[tuple[int, int]]:
    """Return (start, end) byte offsets of patch value content in the JSON.

    Scans for ``"value":`` keys inside the ``"patches"`` array and marks the
    byte range of the value content (string, number, null, bool, array, object).
    """
    import re as _re

    spans: list[tuple[int, int]] = []
    encoded = completion_text.encode("utf-8")

    # Find "patches" array start
    patches_match = _re.search(r'"patches"\s*:\s*\[', completion_text)
    if not patches_match:
        return spans
    patches_start = patches_match.end()

    # Within patches, find each "value": occurrence
    pos = patches_start
    while pos < len(completion_text):
        value_match = _re.search(r'"value"\s*:\s*', completion_text[pos:])
        if not value_match:
            break
        value_content_start = pos + value_match.end()

        # Determine the extent of the value (string, number, null, bool, object, array)
        ch = completion_text[value_content_start : value_content_start + 1]
        if ch == '"':
            # String: find closing quote (handle escapes)
            end = value_content_start + 1
            while end < len(completion_text):
                if completion_text[end] == "\\" and end + 1 < len(completion_text):
                    end += 2
                    continue
                if completion_text[end] == '"':
                    end += 1
                    break
                end += 1
            # UTF-8 byte spans for the string content (inside quotes)
            byte_start = len(completion_text[:value_content_start + 1].encode("utf-8"))
            byte_end = len(completion_text[:end - 1].encode("utf-8"))
            spans.append((byte_start, byte_end))
            pos = end
        elif ch in ("n", "t", "f"):
            # null, true, false
            for keyword in ("null", "true", "false"):
                if completion_text[value_content_start:].startswith(keyword):
                    byte_start = len(completion_text[:value_content_start].encode("utf-8"))
                    byte_end = len(completion_text[:value_content_start + len(keyword)].encode("utf-8"))
                    spans.append((byte_start, byte_end))
                    pos = value_content_start + len(keyword)
                    break
            else:
                pos = value_content_start + 1
        elif ch in ("-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
            # Number
            num_match = _re.match(r'[-\d.eE+]+', completion_text[value_content_start:])
            if num_match:
                end = value_content_start + num_match.end()
                byte_start = len(completion_text[:value_content_start].encode("utf-8"))
                byte_end = len(completion_text[:end].encode("utf-8"))
                spans.append((byte_start, byte_end))
                pos = end
            else:
                pos = value_content_start + 1
        else:
            pos = value_content_start + 1

    return spans


def _byte_offset_to_token_index(
    token_offsets: list[int], byte_offset: int
) -> int:
    """Map a byte offset to the token index that contains it."""
    for idx, offset in enumerate(token_offsets):
        if offset <= byte_offset < (token_offsets[idx + 1] if idx + 1 < len(token_offsets) else offset + 1):
            return idx
    return -1


def _compute_token_span_labels(
    completion_ids: list[int],
    completion_text: str,
    tokenizer: Any,
) -> tuple[list[int], list[int]]:
    """Return (structure_labels, value_labels) for a single completion.

    Both lists have the same length as completion_ids. Structure tokens get
    their actual ID; value tokens get their actual ID. All other positions
    are -100 (ignored in loss). Together structure ∪ value = all tokens.
    """
    value_spans = _identify_value_byte_spans(completion_text)

    # Compute byte offset of each token
    token_offsets: list[int] = []
    byte_pos = 0
    for token_id in completion_ids:
        token_offsets.append(byte_pos)
        try:
            token_str = tokenizer.decode([token_id], add_special_tokens=False)
        except Exception:
            token_str = ""
        byte_pos += len(token_str.encode("utf-8"))

    # Determine which tokens are "value" tokens
    n = len(completion_ids)
    is_value = [False] * n
    for span_start, span_end in value_spans:
        for idx in range(n):
            tok_start = token_offsets[idx]
            tok_end = token_offsets[idx + 1] if idx + 1 < n else tok_start + 1
            # Token overlaps with value span
            if tok_start < span_end and tok_end > span_start:
                is_value[idx] = True

    structure_labels = []
    value_labels = []
    for idx, token_id in enumerate(completion_ids):
        if is_value[idx]:
            structure_labels.append(-100)
            value_labels.append(token_id)
        else:
            structure_labels.append(token_id)
            value_labels.append(-100)

    return structure_labels, value_labels


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
        structure_label_rows: list[list[int]] = []
        value_label_rows: list[list[int]] = []
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

            # Compute token-span labels for structure/value objectives
            completion_text = self._target_json(record)
            struct_labels, val_labels = _compute_token_span_labels(
                completion_ids, completion_text, self.tokenizer
            )

            input_rows.append(prompt_ids + completion_ids)
            label_rows.append([-100] * len(prompt_ids) + completion_ids)
            # Prompt tokens are -100 in structure/value labels too
            structure_label_rows.append([-100] * len(prompt_ids) + struct_labels)
            value_label_rows.append([-100] * len(prompt_ids) + val_labels)
            prompt_last_indices.append(len(prompt_ids) - 1)
            plans.append(build_structured_label_plan(record, self.max_slots))

        padded = self.tokenizer.pad(
            [{"input_ids": row} for row in input_rows],
            padding=True,
            return_tensors="pt",
        )
        labels = torch.full_like(padded["input_ids"], -100)
        structure_labels_tensor = torch.full_like(padded["input_ids"], -100)
        value_labels_tensor = torch.full_like(padded["input_ids"], -100)
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
            structure_labels_tensor[row_index, left_offset : left_offset + len(row_labels)] = torch.tensor(
                structure_label_rows[row_index], dtype=torch.long
            )
            value_labels_tensor[row_index, left_offset : left_offset + len(row_labels)] = torch.tensor(
                value_label_rows[row_index], dtype=torch.long
            )
            pool_indices.append(prompt_last_indices[row_index] + left_offset)

        return {
            **padded,
            "lm_labels": labels,
            "structure_labels": structure_labels_tensor,
            "value_labels": value_labels_tensor,
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

