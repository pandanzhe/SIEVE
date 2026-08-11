from __future__ import annotations

from .types import (
    BeliefSlot,
    BeliefState,
    Decision,
    EvidenceLedger,
    ExecutorResult,
    Observation,
    Patch,
    PatchOp,
    RevisionOutput,
    RiskEnvelope,
    RiskLevel,
    SlotStatus,
)


class StateExecutor:
    """Apply typed local patches while enforcing non-learned state invariants."""

    def apply(
        self,
        state: BeliefState,
        ledger: EvidenceLedger,
        output: RevisionOutput,
        observation: Observation | None = None,
    ) -> ExecutorResult:
        next_state = state.clone()
        next_ledger = ledger.clone()
        costs = {"invalid_patch": 0.0, "collateral_edit": 0.0}
        error = self._validate(next_state, output, observation)
        if error:
            costs["invalid_patch"] = 1.0
            if "outside affected fields" in error:
                costs["collateral_edit"] = 1.0
            return ExecutorResult(next_state, next_ledger, False, costs, error)

        if output.decision is Decision.IGNORE:
            return ExecutorResult(next_state, next_ledger, True, costs)

        if output.decision is Decision.HOLD:
            for patch in output.patches:
                slot = next_state.get(patch.field_id)
                if slot is not None:
                    slot.status = SlotStatus.PENDING
            if observation is not None:
                # Never expose the generator-only perturbation label through the
                # policy-visible evidence ledger.
                next_ledger.append(observation, "held_for_verification")
            return ExecutorResult(next_state, next_ledger, True, costs)

        for patch in output.patches:
            self._apply_patch(next_state, patch, observation)
        return ExecutorResult(next_state, next_ledger, True, costs)

    def _validate(
        self,
        state: BeliefState,
        output: RevisionOutput,
        observation: Observation | None,
    ) -> str | None:
        affected = set(output.affected_fields)
        if len(affected) != len(output.affected_fields):
            return "affected fields must be unique"
        if output.decision is Decision.IGNORE:
            if affected or output.patches or output.verification is not None:
                return "IGNORE must not carry fields, patches, or verification"
            return None

        if not affected:
            return "non-IGNORE decisions require affected fields"
        if output.decision is Decision.HOLD:
            if (
                output.verification is not None
                and output.verification.field_id not in affected
            ):
                return "verification target is outside affected fields"
            for patch in output.patches:
                if patch.field_id not in affected:
                    return "HOLD patch is outside affected fields"
                if patch.op is not PatchOp.SET_STATUS or patch.value != SlotStatus.PENDING.value:
                    return "HOLD may only set pending status"
                if state.get(patch.field_id) is None:
                    return "HOLD patch target does not exist"
            return None

        if output.verification is not None:
            return "UPDATE may not request verification"
        if not output.patches:
            return "UPDATE requires at least one patch"
        patch_fields = [patch.field_id for patch in output.patches]
        if len(patch_fields) != len(set(patch_fields)):
            return "UPDATE patch fields must be unique"
        for patch in output.patches:
            error = self._validate_update_patch(state, affected, patch, observation)
            if error:
                return error
        return None

    @staticmethod
    def _validate_update_patch(
        state: BeliefState,
        affected: set[str],
        patch: Patch,
        observation: Observation | None,
    ) -> str | None:
        if patch.field_id not in affected:
            return "UPDATE patch is outside affected fields"
        current = state.get(patch.field_id)
        if patch.op is PatchOp.ADD_FIELD:
            if current is not None:
                return "ADD_FIELD target already exists"
            if len(state.slots) >= state.max_slots:
                return "no empty belief slot"
        elif current is None:
            return "patch target does not exist"

        if observation is not None and patch.op in {PatchOp.ADD_FIELD, PatchOp.SET_VALUE}:
            if patch.field_id != observation.field_id:
                return "observation may only supply the matching field"
            if current is not None and current.entity != observation.entity:
                return "observation entity does not match the target slot"

        if patch.op is PatchOp.SET_VALUE and current is not None:
            if current.value is not None and type(patch.value) is not type(current.value):
                return "SET_VALUE type does not match the target slot"
        elif patch.op is PatchOp.SET_STATUS:
            try:
                SlotStatus(patch.value)
            except (TypeError, ValueError):
                return "SET_STATUS value is invalid"
        elif patch.op is PatchOp.SET_PROVENANCE and not isinstance(patch.value, str):
            return "SET_PROVENANCE requires a string"
        elif patch.op is PatchOp.SET_VALIDITY and not (
            patch.value is None or isinstance(patch.value, int)
        ):
            return "SET_VALIDITY requires an integer or null"
        return None

    @staticmethod
    def _apply_patch(
        state: BeliefState,
        patch: Patch,
        observation: Observation | None,
    ) -> None:
        if patch.op is PatchOp.ADD_FIELD:
            source = observation.source if observation else "unknown"
            observed_at = observation.observed_at if observation else 0
            valid_from = observation.valid_from if observation else None
            entity = observation.entity if observation else "unknown"
            state.slots.append(
                BeliefSlot(
                    patch.field_id,
                    patch.value,
                    SlotStatus.TRUSTED,
                    source,
                    observed_at,
                    valid_from,
                    entity,
                )
            )
            return

        slot = state.get(patch.field_id)
        if slot is None:
            raise RuntimeError("validated patch target disappeared")
        if patch.op is PatchOp.SET_VALUE:
            slot.value = patch.value
            slot.status = SlotStatus.TRUSTED
            if observation:
                slot.source = observation.source
                slot.observed_at = observation.observed_at
                slot.valid_from = observation.valid_from
                slot.entity = observation.entity
        elif patch.op is PatchOp.SET_STATUS:
            slot.status = SlotStatus(patch.value)
        elif patch.op is PatchOp.SET_PROVENANCE:
            slot.source = str(patch.value)
        elif patch.op is PatchOp.SET_VALIDITY:
            slot.valid_from = int(patch.value) if patch.value is not None else None

    @staticmethod
    def blocks_action(state: BeliefState, risk: RiskEnvelope) -> bool:
        if risk.risk is not RiskLevel.HIGH and risk.reversible:
            return False
        return any(
            (slot := state.get(field_id)) is None or slot.status is SlotStatus.PENDING
            for field_id in risk.dependent_fields
        )
