from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict

from ..core.executor import StateExecutor
from ..core.types import Decision
from .models import GenerationCandidate, Realization, ValidationResult, VerificationAction


_LABEL_LEAKAGE = re.compile(
    r"(?:"
    r"\b(?:decision|reason_code|conflict_type)\s*[:=]\s*"
    r"(?:UPDATE|IGNORE|HOLD|[A-Z_]{4,})\b"
    r"|\b(?:relevant|condition|perturbation)\s*[:=]"
    r")",
    re.IGNORECASE,
)


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _grounded_value_present(value: object, normalized_text: str) -> bool:
    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{", "(")):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(stripped)
                    break
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    continue
    if isinstance(parsed, dict):
        return all(_grounded_value_present(item, normalized_text) for item in parsed.values())
    if isinstance(parsed, (list, tuple, set)):
        return all(_grounded_value_present(item, normalized_text) for item in parsed)
    return str(parsed).casefold() in normalized_text


def _belief_signature(state: object) -> str:
    return json.dumps(asdict(state), ensure_ascii=False, sort_keys=True, default=str)


def validate_candidate(
    candidate: GenerationCandidate,
    realization: Realization,
    *,
    seen_texts: set[str] | None = None,
) -> ValidationResult:
    issues: list[str] = []
    text = realization.observation_text.strip()
    normalized = _normalized_text(text)

    if realization.record_id != candidate.record_id:
        issues.append("record_id_mismatch")
    if not text:
        issues.append("empty_observation_text")
    if _LABEL_LEAKAGE.search(text):
        issues.append("label_leakage")

    provenance = candidate.provenance
    required_provenance = (
        provenance.source_dataset,
        provenance.source_version,
        provenance.source_record_id,
        provenance.source_uri,
        provenance.license,
    )
    if not all(required_provenance):
        issues.append("incomplete_provenance")
    if len(provenance.source_sha256) != 64:
        issues.append("invalid_source_hash")
    else:
        try:
            bytes.fromhex(provenance.source_sha256)
        except ValueError:
            issues.append("invalid_source_hash")
    if provenance.transformation != "identity" and not provenance.parent_record_id:
        issues.append("missing_parent_record_id")

    event = candidate.structured_event
    grounding = event.get("grounding", {})
    entity_terms = grounding.get("entity_terms") or [event.get("entity", "")]
    value_terms = grounding.get("value_terms") or [event.get("value", "")]
    if any(
        str(term).strip() and str(term).casefold() not in normalized
        for term in entity_terms
    ):
        issues.append("missing_grounded_entity")
    if any(
        term not in ("", None) and not _grounded_value_present(term, normalized)
        for term in value_terms
    ):
        issues.append("missing_grounded_value")

    target = candidate.target
    if target.decision is Decision.IGNORE:
        if target.affected_fields or target.patches or target.verification is not None:
            issues.append("invalid_ignore_branch")
    elif target.decision is Decision.UPDATE:
        if not target.affected_fields or not target.patches or target.verification is not None:
            issues.append("invalid_update_branch")
    elif target.decision is Decision.HOLD:
        if not target.affected_fields or not target.patches:
            issues.append("invalid_hold_branch")
        if (
            candidate.verification_action is VerificationAction.VERIFY
            and target.verification is None
        ):
            issues.append("missing_verification_request")
        if (
            candidate.verification_action is VerificationAction.DEFER
            and target.verification is not None
        ):
            issues.append("defer_has_verification_request")

    execution = StateExecutor().apply(
        candidate.context.belief_state,
        candidate.context.ledger,
        target,
        candidate.context.observation,
    )
    if not execution.executed:
        issues.append("patch_rejected")
    elif _belief_signature(execution.state) != _belief_signature(candidate.oracle_state):
        issues.append("oracle_state_mismatch")

    if seen_texts is not None:
        if normalized in seen_texts:
            issues.append("exact_duplicate")
        elif not issues:
            seen_texts.add(normalized)

    return ValidationResult(not issues, tuple(dict.fromkeys(issues)))


