from __future__ import annotations

from typing import Protocol

from ..core.transition import EnvironmentAction
from ..core.types import Observation


class AgentEnvironment(Protocol):
    def step(self, action: EnvironmentAction) -> list[Observation]: ...

    def oracle_state(self) -> dict[str, object]: ...
  