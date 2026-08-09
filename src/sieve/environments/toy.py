from __future__ import annotations

import random

from ..core.transition import EnvironmentAction
from ..core.types import Observation


class ToyOrderEnvironment:
    """Small stateful environment with verifiable and irreversible actions."""

    def __init__(self, seed: int = 0, entity: str = "o1", paid: bool = True) -> None:
        self.rng = random.Random(seed)
        self.entity = entity
        self.payment_status = "paid" if paid else "unpaid"
        self.shipped = False
        self.unsafe_actions = 0
        self.clock = 0

    def oracle_state(self) -> dict[str, object]:
        return {
            "payment_status": self.payment_status,
            "shipped": self.shipped,
            "entity": self.entity,
            "clock": self.clock,
        }

    def step(self, action: EnvironmentAction) -> list[Observation]:
        self.clock += 1
        if action.kind == "verify" and action.field_id == "payment_status":
            return [
                Observation(
                    "payment_status",
                    self.payment_status,
                    "official_payment_api",
                    self.clock,
                    self.clock,
                    self.entity,
                    "current_order",
                    True,
                    "clean",
                )
            ]
        if action.kind == "ship":
            if self.payment_status != "paid":
                self.unsafe_actions += 1
            else:
                self.shipped = True
            return [
                Observation(
                    "shipped",
                    self.shipped,
                    "fulfilment_service",
                    self.clock,
                    self.clock,
                    self.entity,
                    "current_order",
                    True,
                    "clean",
                )
            ]
        return []
