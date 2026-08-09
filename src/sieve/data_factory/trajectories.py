from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace
from ..core.executor import StateExecutor
from ..core.types import (
    Budget,
    Decision,
    Observation,
    Patch,
    PatchOp,
    RevisionContext,
    RevisionOutput,
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
from .rules import _atomic_grounding_terms, authoritative_source, build_candidate


_COMMERCE_DOMAINS = {"retail", "shopping", "shopping_admin"}
_SERVICE_DOMAINS = {"airline", "telecom"}


def macro_domain(source: GroundedSourceRecord) -> str:
    domain = source.domain.casefold()
    if domain in _COMMERCE_DOMAINS:
        return "commerce"
    if domain in _SERVICE_DOMAINS:
        return "service"
    return "workflow"


def _macro_targets(total: int) -> dict[str, int]:
    if total <= 0 or total % 20:
        raise ValueError("trajectory corpus total must be a positive multiple of 20")
    return {
        "commerce": total * 50 // 100,
        "service": total * 30 // 100,
        "workflow": total * 20 // 100,
    }


def _allocate_counts(targets: dict[str, int], count: int) -> dict[str, int]:
    """Allocate an exact global count while preserving domain proportions."""
    total = sum(targets.values())
    raw = {name: target * count / total for name, target in targets.items()}
    allocated = {name: int(value) for name, value in raw.items()}
    remainder = count - sum(allocated.values())
    order = sorted(
        targets,
        key=lambda name: (raw[name] - allocated[name], targets[name], name),
        reverse=True,
    )
    for name in order[:remainder]:
        allocated[name] += 1
    return allocated


def build_corpus_candidates(
    sources: list[GroundedSourceRecord],
    *,
    total: int = 6000,
    seed: int = 42,
) -> list[GenerationCandidate]:
    """Create domain-stratified singleton and HOLD-to-UPDATE SFT decision points."""
    targets = _macro_targets(total)
    pools: dict[str, list[GroundedSourceRecord]] = {
        name: [source for source in sources if macro_domain(source) == name]
        for name in targets
    }
    missing = [name for name, records in pools.items() if not records]
    if missing:
        raise ValueError(f"source data is missing macro domains: {missing}")

    rng = random.Random(seed)
    for records in pools.values():
        rng.shuffle(records)
    indexes: Counter[str] = Counter()
    episode = 0
    sequence = 0
    source_event = 0
    result: list[GenerationCandidate] = []

    def next_source(name: str) -> GroundedSourceRecord:
        nonlocal source_event
        records = pools[name]
        index = indexes[name]
        indexes[name] += 1
        source = records[index % len(records)]
        source_event += 1
        observed_at = 1_750_000_000 + source_event * 600
        validity_lag = max(0, source.observed_at - source.valid_from) if source.valid_from is not None else None
        return replace(
            source,
            observed_at=observed_at,
            valid_from=(observed_at - validity_lag if validity_lag is not None else None),
        )

    def next_distractor(name: str, source: GroundedSourceRecord) -> GroundedSourceRecord:
        records = pools[name]
        same_field = [
            item for item in records
            if item.entity != source.entity and item.field_id == source.field_id
        ]
        candidates = same_field or [item for item in records if item.entity != source.entity]
        if not candidates:
            raise ValueError(f"macro domain {name} requires at least two distinct entities")
        return candidates[indexes[name] % len(candidates)]

    def with_group(candidate: GenerationCandidate, name: str) -> GenerationCandidate:
        nonlocal episode
        episode += 1
        group = (
            f"{name}:{candidate.provenance.source_dataset}:"
            f"{candidate.provenance.source_record_id}:episode-{episode:06d}"
        )
        return replace(candidate, group_id=group)

    hold_counts = _allocate_counts(targets, total * 15 // 100)
    ignore_counts = _allocate_counts(targets, total * 15 // 100)
    for name, macro_total in targets.items():
        hold_count = hold_counts[name]
        ignore_count = ignore_counts[name]
        update_count = macro_total - hold_count - ignore_count
        standalone_updates = update_count - hold_count

        for _ in range(standalone_updates):
            source = next_source(name)
            sequence += 1
            candidate = build_candidate(
                source,
                QuotaCell(ObservationType.NEW_CONSISTENT, Decision.UPDATE, 1),
                sequence=sequence,
            )
            result.append(with_group(candidate, name))

        for _ in range(hold_count):
            source = next_source(name)
            sequence += 1
            hold = build_candidate(
                source,
                QuotaCell(
                    ObservationType.TOOL_ERROR_OR_LOW_TRUST,
                    Decision.HOLD,
                    1,
                    VerificationAction.VERIFY,
                ),
                sequence=sequence,
            )
            hold = with_group(hold, name)
            sequence += 1
            successor = build_verification_successor(hold, sequence=sequence)
            result.extend((hold, successor))

        for _ in range(ignore_count):
            source = next_source(name)
            distractor = next_distractor(name, source)
            sequence += 1
            candidate = build_candidate(
                source,
                QuotaCell(ObservationType.EXPLICIT_CONFLICT, Decision.IGNORE, 1),
                sequence=sequence,
                distractor=distractor,
            )
            result.append(with_group(candidate, name))

    if len(result) != total:
        raise RuntimeError(f"trajectory schedule produced {len(result)} records, expected {total}")
    return result

def build_verification_successor(
    hold: GenerationCandidate,
    *,
    sequence: int,
) -> GenerationCandidate:
    """Build the authoritative next decision after executing a HOLD action."""
    if hold.target.decision is not Decision.HOLD:
        raise ValueError("verification successor requires a HOLD candidate")
    if hold.target.verification is None:
        raise ValueError("verification successor requires a verification request")

    executed_hold = StateExecutor().apply(
        hold.context.belief_state,
        hold.context.ledger,
        hold.target,
        hold.context.observation,
    )
    if not executed_hold.executed:
        raise ValueError(f"HOLD transition failed: {executed_hold.error}")

    prior = hold.context.observation
    observed_at = prior.observed_at + 300
    source = authoritative_source(hold.domain, prior.source)
    evidence = Observation(
        field_id=prior.field_id,
        value=prior.value,
        source=source,
        observed_at=observed_at,
        valid_from=max(prior.valid_from or observed_at, observed_at - 60),
        entity=prior.entity,
        condition="verified_by_primary_record",
        relevant=True,
        perturbation=ObservationType.VERIFICATION_CONFIRMED.value,
        source_authority="primary_record",
        authenticated=True,
    )
    previous_budget = hold.context.budget
    context = RevisionContext(
        executed_hold.state,
        evidence,
        hold.context.goal,
        hold.context.risk,
        Budget(
            verification_remaining=max(0, previous_budget.verification_remaining - 1),
            tool_remaining=max(0, previous_budget.tool_remaining - 1),
            steps_remaining=max(0, previous_budget.steps_remaining - 1),
            tokens_remaining=max(0, previous_budget.tokens_remaining - 256),
        ),
        executed_hold.ledger,
    )
    target = RevisionOutput(
        Decision.UPDATE,
        (evidence.field_id,),
        (Patch(PatchOp.SET_VALUE, evidence.field_id, evidence.value),),
    )
    execution = StateExecutor().apply(
        context.belief_state,
        context.ledger,
        target,
        evidence,
    )
    if not execution.executed:
        raise ValueError(f"verification UPDATE failed: {execution.error}")

    base = hold.provenance
    provenance = Provenance(
        base.source_dataset,
        base.source_version,
        base.source_record_id,
        base.source_uri,
        base.source_sha256,
        base.license,
        parent_record_id=hold.record_id,
        transformation="verification_confirmed:update",
    )
    return GenerationCandidate(
        record_id=f"sieve-{sequence:06d}",
        group_id=hold.group_id,
        provenance=provenance,
        observation_type=ObservationType.VERIFICATION_CONFIRMED,
        scenario_type=hold.scenario_type,
        domain=hold.domain,
        difficulty=Difficulty.MEDIUM,
        context=context,
        target=target,
        oracle_state=execution.state,
        reason_code="VERIFIED_EVIDENCE",
        conflict_type=None,
        verification_action=VerificationAction.NO_VERIFY,
        structured_event={
            "source_text": f"Authoritative verification for {hold.context.goal}",
            "entity": evidence.entity,
            "field_id": evidence.field_id,
            "value": evidence.value,
            "source": evidence.source,
            "source_authority": evidence.source_authority,
            "authenticated": evidence.authenticated,
            "observed_at": evidence.observed_at,
            "valid_from": evidence.valid_from,
            "source_dataset": provenance.source_dataset,
            "grounding": {
                "entity_terms": [str(evidence.entity)],
                "value_terms": _atomic_grounding_terms(evidence.value),
            },
        },
    )
