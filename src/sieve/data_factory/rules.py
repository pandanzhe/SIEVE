from __future__ import annotations

from dataclasses import replace
import ast
import json
import re
from typing import Any

from ..core.executor import StateExecutor
from ..core.types import (
    BeliefSlot,
    BeliefState,
    Budget,
    Decision,
    EvidenceLedger,
    Observation,
    Patch,
    PatchOp,
    RevisionContext,
    RevisionOutput,
    RiskEnvelope,
    RiskLevel,
    SlotStatus,
    VerificationRequest,
)
from .models import (
    Difficulty,
    GenerationCandidate,
    GroundedSourceRecord,
    ObservationType,
    Provenance,
    QuotaCell,
    VerificationAction,
)


_REASON_CODES = {
    ObservationType.NEW_CONSISTENT: "CONSISTENT_EVIDENCE",
    ObservationType.EXPLICIT_CONFLICT: "EXPLICIT_CONFLICT",
    ObservationType.IMPLICIT_CONFLICT: "IMPLICIT_CONFLICT",
    ObservationType.STALE: "TEMPORAL_RELATION",
    ObservationType.IRRELEVANT: "GOAL_RELEVANCE",
    ObservationType.TOOL_ERROR_OR_LOW_TRUST: "SOURCE_RELIABILITY",
    ObservationType.INSUFFICIENT_OR_AMBIGUOUS: "INSUFFICIENT_EVIDENCE",
    ObservationType.VERIFICATION_CONFIRMED: "VERIFIED_EVIDENCE",
}

_CONFLICT_TYPES = {
    ObservationType.EXPLICIT_CONFLICT: "explicit",
    ObservationType.IMPLICIT_CONFLICT: "implicit",
    ObservationType.STALE: "temporal",
    ObservationType.IRRELEVANT: "relevance",
    ObservationType.TOOL_ERROR_OR_LOW_TRUST: "source",
    ObservationType.INSUFFICIENT_OR_AMBIGUOUS: "ambiguity",
}


_GROUNDING_KEY_NAMES = {
    "error", "response", "id", "name", "phone", "email", "website", "codepostal",
    "value", "status", "data", "result",
}


def _atomic_grounding_terms(value: Any, *, limit: int = 4) -> list[str]:
    terms: list[str] = []

    def append(item: Any, depth: int = 0) -> None:
        if len(terms) >= limit or depth > 4 or item is None:
            return
        if isinstance(item, str):
            stripped = item.strip()
            if not stripped:
                return
            if stripped.startswith(("[", "{", "(")):
                for parser in (json.loads, ast.literal_eval):
                    try:
                        parsed = parser(stripped)
                    except (ValueError, SyntaxError, json.JSONDecodeError):
                        continue
                    append(parsed, depth + 1)
                    return
            if len(stripped) > 120:
                quoted = re.findall(r'["\']([^"\']{2,80})["\']', stripped)
                for token in quoted:
                    if token.casefold() not in _GROUNDING_KEY_NAMES:
                        append(token, depth + 1)
                return
            if stripped not in terms:
                terms.append(stripped)
            return
        if isinstance(item, dict):
            for nested in item.values():
                append(nested, depth + 1)
            return
        if isinstance(item, (list, tuple, set)):
            for nested in item:
                append(nested, depth + 1)
            return
        rendered = str(item)
        if rendered not in terms:
            terms.append(rendered)

    append(value)
    return terms


def _typed_pair(old_value: Any, new_value: Any) -> tuple[Any, Any]:
    if type(old_value) is type(new_value):
        return old_value, new_value
    return str(old_value), str(new_value)


def authoritative_source(domain: str, fallback: str) -> str:
    normalized = domain.casefold()
    if normalized in {"shopping", "shopping_admin", "retail"}:
        return "order_service_api"
    if normalized == "airline":
        return "airline_booking_api"
    if normalized == "telecom":
        return "telecom_account_api"
    if normalized == "database":
        return "database_query"
    if normalized in {"tool_api", "workflow"}:
        return "workflow_service_api"
    return fallback


def secondary_source(domain: str) -> str:
    normalized = domain.casefold()
    if normalized in {"shopping", "shopping_admin", "retail"}:
        return "forwarded_customer_message"
    if normalized == "airline":
        return "travel_agency_forward"
    if normalized == "telecom":
        return "unverified_support_message"
    return "cached_tool_result"


def _unknown_value(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().casefold() in {"", "unknown", "n/a", "none"}
    )

def _derived_provenance(
    source: GroundedSourceRecord, observation_type: ObservationType, decision: Decision
) -> Provenance:
    base = source.provenance
    return Provenance(
        base.source_dataset,
        base.source_version,
        base.source_record_id,
        base.source_uri,
        base.source_sha256,
        base.license,
        parent_record_id=f"{base.source_dataset}:{base.source_record_id}",
        transformation=f"{observation_type.value}:{decision.value.lower()}",
    )


def build_candidate(
    source: GroundedSourceRecord,
    quota: QuotaCell,
    *,
    sequence: int,
    distractor: GroundedSourceRecord | None = None,
) -> GenerationCandidate:
    if quota.count <= 0:
        raise ValueError("quota count must be positive")
    if _unknown_value(source.old_value):
        old_value, new_value = None, source.new_value
    else:
        old_value, new_value = _typed_pair(source.old_value, source.new_value)
    observation_time = max(10, source.observed_at) + sequence
    current_valid = max(0, (source.valid_from or observation_time) - 10)
    current_observed = max(current_valid, observation_time - 10)
    state = BeliefState(
        8,
        [
            BeliefSlot(
                source.field_id,
                old_value,
                SlotStatus.EMPTY if old_value is None else SlotStatus.TRUSTED,
                "initial_state" if old_value is None else "prior_verified_state",
                current_observed,
                current_valid,
                source.entity,
            )
        ],
    )

    obs_value = new_value
    obs_source = source.source
    obs_entity = source.entity
    obs_field = source.field_id
    valid_from: int | None = observation_time
    relevant = True
    condition = "applies_to_current_entity"
    source_authority = "primary_record"
    authenticated = True

    if quota.observation_type is ObservationType.NEW_CONSISTENT:
        if quota.decision is Decision.IGNORE:
            obs_value = old_value
            condition = "duplicate_of_current_state"
        elif quota.decision is Decision.HOLD:
            obs_source = secondary_source(source.domain)
    elif quota.observation_type is ObservationType.EXPLICIT_CONFLICT:
        condition = "explicit_value_conflict_with_current_state"
        if quota.decision is Decision.IGNORE:
            if distractor is not None:
                obs_entity = distractor.entity
                obs_field = distractor.field_id
                obs_value = distractor.new_value
                obs_source = authoritative_source(distractor.domain, distractor.source)
            else:
                obs_entity = f"other-{source.entity}"
            condition = "explicitly_different_entity"
        elif quota.decision is Decision.HOLD:
            obs_source = secondary_source(source.domain)
    elif quota.observation_type is ObservationType.IMPLICIT_CONFLICT:
        condition = "conflict_inferred_from_consequence"
        if quota.decision is Decision.IGNORE:
            obs_source = "uncorroborated_indirect_report"
        elif quota.decision is Decision.HOLD:
            obs_source = "indirect_report"
    elif quota.observation_type is ObservationType.STALE:
        if quota.decision is Decision.UPDATE:
            valid_from = current_valid + 5
        elif quota.decision is Decision.IGNORE:
            valid_from = max(0, current_valid - 5)
        else:
            valid_from = None
            obs_source = "undated_cache"
        observation_time = max(observation_time, (valid_from or 0) + 5)
    elif quota.observation_type is ObservationType.IRRELEVANT:
        condition = "outside_active_subgoal"
        if quota.decision is Decision.IGNORE:
            relevant = False
        elif quota.decision is Decision.HOLD:
            obs_source = "unverified_auxiliary_source"
        else:
            condition = "maintained_long_term_field_outside_active_subgoal"
    elif quota.observation_type is ObservationType.TOOL_ERROR_OR_LOW_TRUST:
        condition = "tool_reliability_requires_assessment"
        if quota.decision is Decision.UPDATE:
            obs_source = "official_recovered_tool"
        elif quota.decision is Decision.IGNORE:
            obs_source = "tool_error_response"
        else:
            obs_source = secondary_source(source.domain)
    elif quota.observation_type is ObservationType.INSUFFICIENT_OR_AMBIGUOUS:
        valid_from = None
        condition = "partial_or_ambiguous_evidence"
        if quota.decision is Decision.UPDATE:
            obs_source = "official_corroborated_partial"
        elif quota.decision is Decision.IGNORE:
            relevant = False
        else:
            obs_source = "ambiguous_report"

    if quota.decision is Decision.UPDATE:
        obs_source = authoritative_source(source.domain, obs_source)
        source_authority = "primary_record"
        authenticated = True
    elif quota.decision is Decision.HOLD:
        source_authority = "secondary"
        authenticated = False
    elif obs_entity != source.entity:
        source_authority = "primary_record"
        authenticated = True
    else:
        source_authority = "untrusted_or_inapplicable"
        authenticated = False
    observation = Observation(
        obs_field,
        obs_value,
        obs_source,
        observation_time,
        valid_from,
        obs_entity,
        condition,
        relevant,
        quota.observation_type.value,
        source_authority=source_authority,
        authenticated=authenticated,
    )
    risk = RiskEnvelope(
        source.goal,
        (source.field_id,),
        RiskLevel.HIGH if quota.decision is Decision.HOLD else RiskLevel.MEDIUM,
        quota.decision is not Decision.HOLD,
    )
    context = RevisionContext(
        state,
        observation,
        source.goal,
        risk,
        Budget(2, 5, 8, 4096),
        EvidenceLedger(4),
    )

    if quota.decision is Decision.UPDATE:
        target = RevisionOutput(
            Decision.UPDATE,
            (source.field_id,),
            (Patch(PatchOp.SET_VALUE, source.field_id, obs_value),),
        )
    elif quota.decision is Decision.IGNORE:
        target = RevisionOutput(Decision.IGNORE)
    else:
        verification = None
        if quota.verification_action is VerificationAction.VERIFY:
            verification = VerificationRequest(
                f"verify_{source.field_id}", source.field_id
            )
        target = RevisionOutput(
            Decision.HOLD,
            (source.field_id,),
            (
                Patch(
                    PatchOp.SET_STATUS,
                    source.field_id,
                    SlotStatus.PENDING.value,
                ),
            ),
            verification,
        )

    execution = StateExecutor().apply(
        context.belief_state, context.ledger, target, observation
    )
    if not execution.executed:
        raise ValueError(f"rule produced invalid target: {execution.error}")

    difficulty = Difficulty.HARD if quota.observation_type in {
        ObservationType.IMPLICIT_CONFLICT,
        ObservationType.INSUFFICIENT_OR_AMBIGUOUS,
    } else Difficulty.MEDIUM
    record_id = f"sieve-{sequence:06d}"
    group_id = f"{source.provenance.source_dataset}:{source.provenance.source_record_id}"
    return GenerationCandidate(
        record_id,
        group_id,
        _derived_provenance(source, quota.observation_type, quota.decision),
        quota.observation_type,
        source.scenario_type,
        source.domain,
        difficulty,
        context,
        target,
        execution.state,
        _REASON_CODES[quota.observation_type],
        _CONFLICT_TYPES.get(quota.observation_type),
        quota.verification_action,
        {
            "source_text": source.source_text,
            "entity": obs_entity,
            "field_id": obs_field,
            "value": obs_value,
            "source": obs_source,
            "observed_at": observation_time,
            "valid_from": valid_from,
            "condition": condition,
            "relevant": relevant,
            "source_authority": source_authority,
            "authenticated": authenticated,
            "source_dataset": source.provenance.source_dataset,
            "distractor_source_record_id": (
                distractor.provenance.source_record_id if distractor is not None else None
            ),
            "grounding": {
                "entity_terms": [str(obs_entity)],
                "value_terms": _atomic_grounding_terms(obs_value),
            },
        },
    )


