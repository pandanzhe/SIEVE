from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..core.types import (
    Decision,
    Patch,
    PatchOp,
    RevisionContext,
    RevisionOutput,
    SlotStatus,
    VerificationRequest,
)
from ..data.schema import context_summary
from .hf_data import SIEVE_SYSTEM_PROMPT

_TOP_LEVEL_KEYS = {"decision", "affected_fields", "patches", "verification"}
_PATCH_KEYS = {"op", "field_id", "value"}
_VERIFICATION_KEYS = {"tool", "field_id"}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is not allowed: {key}")
        result[key] = value
    return result


def render_context_prompt(
    tokenizer: Any,
    context: RevisionContext,
    *,
    enable_thinking: bool = False,
) -> str:
    """Render the deployment-visible context without private scenario labels."""
    messages = [
        {"role": "system", "content": SIEVE_SYSTEM_PROMPT},
        {"role": "user", "content": context_summary(context)},
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
        f"{context_summary(context)}\n<|sieve_action|>\n"
    )


def revision_output_payload(output: RevisionOutput) -> dict[str, Any]:
    return {
        "decision": output.decision.value,
        "affected_fields": list(output.affected_fields),
        "patches": [
            {"op": patch.op.value, "field_id": patch.field_id, "value": patch.value}
            for patch in output.patches
        ],
        "verification": (
            None
            if output.verification is None
            else {
                "tool": output.verification.tool,
                "field_id": output.verification.field_id,
            }
        ),
    }


def revision_output_json(output: RevisionOutput) -> str:
    return json.dumps(
        revision_output_payload(output),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _string_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(value)


def parse_revision_output(text: str) -> RevisionOutput:
    """Parse one exact JSON action and reject prose, unknown keys, and gated misuse."""
    try:
        raw = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"output is not one exact JSON object: {error.msg}") from error
    if not isinstance(raw, dict):
        raise ValueError("output must be a JSON object")
    if set(raw) != _TOP_LEVEL_KEYS:
        missing = sorted(_TOP_LEVEL_KEYS - set(raw))
        extra = sorted(set(raw) - _TOP_LEVEL_KEYS)
        raise ValueError(f"invalid top-level keys; missing={missing}, extra={extra}")

    try:
        decision = Decision(raw["decision"])
    except (TypeError, ValueError) as error:
        raise ValueError("decision must be UPDATE, HOLD, or IGNORE") from error
    affected_fields = _string_list(raw["affected_fields"], "affected_fields")

    raw_patches = raw["patches"]
    if not isinstance(raw_patches, list):
        raise ValueError("patches must be a list")
    patches: list[Patch] = []
    for index, item in enumerate(raw_patches):
        if not isinstance(item, dict) or set(item) != _PATCH_KEYS:
            raise ValueError(f"patches[{index}] must contain exactly op, field_id, value")
        if not isinstance(item["field_id"], str):
            raise ValueError(f"patches[{index}].field_id must be a string")
        try:
            operation = PatchOp(item["op"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"patches[{index}].op is invalid") from error
        patches.append(Patch(operation, item["field_id"], item["value"]))

    raw_verification = raw["verification"]
    verification = None
    if raw_verification is not None:
        if (
            not isinstance(raw_verification, dict)
            or set(raw_verification) != _VERIFICATION_KEYS
            or not isinstance(raw_verification["tool"], str)
            or not isinstance(raw_verification["field_id"], str)
        ):
            raise ValueError("verification must be null or contain string tool and field_id")
        verification = VerificationRequest(
            tool=raw_verification["tool"], field_id=raw_verification["field_id"]
        )

    patch_fields = [patch.field_id for patch in patches]
    if len(patch_fields) != len(set(patch_fields)):
        raise ValueError("patch fields must not contain duplicates")
    if decision is Decision.IGNORE:
        if affected_fields or patches or verification is not None:
            raise ValueError("IGNORE must not carry fields, patches, or verification")
    elif decision is Decision.UPDATE:
        if not affected_fields or not patches:
            raise ValueError("UPDATE requires affected fields and executable patches")
        if verification is not None:
            raise ValueError("UPDATE may not request verification")
        if set(affected_fields) != set(patch_fields):
            raise ValueError("UPDATE affected fields must equal patch fields")
    else:
        if not affected_fields:
            raise ValueError("HOLD requires at least one affected field")
        if verification is not None and verification.field_id not in affected_fields:
            raise ValueError("verification target must be an affected field")
        for patch in patches:
            if (
                patch.field_id not in affected_fields
                or patch.op is not PatchOp.SET_STATUS
                or patch.value != SlotStatus.PENDING.value
            ):
                raise ValueError("HOLD patches may only mark affected fields pending")

    return RevisionOutput(decision, affected_fields, tuple(patches), verification)


def safe_parse_revision_output(text: str) -> tuple[RevisionOutput, bool, str | None]:
    try:
        return parse_revision_output(text), True, None
    except ValueError as error:
        return RevisionOutput(decision=Decision.IGNORE), False, str(error)


DEFAULT_READINESS_THRESHOLDS: dict[str, tuple[str, float]] = {
    "parse_rate": ("min", 0.98),
    "executable_rate": ("min", 0.97),
    "decision_macro_f1": ("min", 0.90),
    "full_action_exact_match": ("min", 0.85),
    "patch_value_exact_match": ("min", 0.85),
    "false_update_rate": ("max", 0.03),
    "group_reward_variance": ("min_exclusive", 0.0),
    "closed_loop_success_rate": ("min", 0.60),
    "closed_loop_parse_rate": ("min", 0.95),
}


def assess_stage1_readiness(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, tuple[str, float]] | None = None,
) -> tuple[bool, dict[str, str]]:
    """Apply the pre-RL gate to free-generation metrics, not teacher-forced loss."""
    rules = dict(thresholds or DEFAULT_READINESS_THRESHOLDS)
    failures: dict[str, str] = {}
    for name, (operator, threshold) in rules.items():
        if name not in metrics:
            failures[name] = "missing"
            continue
        value = float(metrics[name])
        passed = {
            "min": value >= threshold,
            "min_exclusive": value > threshold,
            "max": value <= threshold,
        }.get(operator)
        if passed is None:
            raise ValueError(f"unknown readiness operator: {operator}")
        if not passed:
            failures[name] = f"observed={value:.6g}, required {operator} {threshold:.6g}"
    return not failures, failures
