from __future__ import annotations

from typing import Protocol

from ..core.types import RevisionContext, RevisionOutput


class RevisionPolicy(Protocol):
    def predict(self, context: RevisionContext, greedy: bool = True) -> RevisionOutput: ...

    def log_probability(self, context: RevisionContext, output: RevisionOutput) -> float: ...
