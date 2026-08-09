from __future__ import annotations

import hashlib
import re

import numpy as np

from ..core.types import RevisionContext
from ..data.schema import context_summary


TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+|[^\s]", re.UNICODE)


class HashedFeaturizer:
    def __init__(self, dimension: int) -> None:
        if dimension < 16:
            raise ValueError("feature dimension must be at least 16")
        self.dimension = dimension

    def encode(self, context: RevisionContext) -> np.ndarray:
        """Encode stable, deployment-visible semantics without high-cardinality values."""
        vector = np.zeros(self.dimension, dtype=np.float64)
        observation = context.observation
        stable_text = " ".join([
            context.goal,
            context.risk.active_subgoal,
            context.risk.risk.value,
            observation.field_id,
            observation.source,
            observation.source_authority or "unknown_authority",
            *(context.risk.dependent_fields),
            *(
                f"{slot.id} {slot.status.value} {slot.source}"
                for slot in context.belief_state.slots
            ),
        ])
        for token in TOKEN_RE.findall(stable_text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little")
            index = raw % self.dimension
            sign = 1.0 if (raw >> 8) & 1 else -1.0
            vector[index] += sign

        matching_slot = context.belief_state.get(observation.field_id)
        vector[0] += 1.0
        vector[1] += context.budget.verification_remaining / max(
            1, context.budget.tool_remaining
        )
        vector[2] += float(matching_slot is not None)
        vector[3] += 4.0 * float(
            matching_slot is not None and matching_slot.entity == observation.entity
        )
        vector[4] += float(
            observation.valid_from is not None
            and observation.valid_from >= observation.observed_at
        )
        vector[5] += float(
            observation.source.startswith("official_")
            or observation.source_authority == "primary_record"
        )
        vector[6] += 4.0 * float(observation.valid_from is None)
        vector[7] += float(observation.authenticated is True)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
