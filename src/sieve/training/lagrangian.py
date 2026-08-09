from __future__ import annotations

from collections.abc import Mapping


class LagrangeController:
    def __init__(
        self,
        limits: Mapping[str, float],
        learning_rate: float,
        initial: Mapping[str, float] | None = None,
    ) -> None:
        if not limits or any(float(value) < 0 for value in limits.values()):
            raise ValueError("limits must contain non-negative values")
        if learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        self.limits = {key: float(value) for key, value in limits.items()}
        self.learning_rate = float(learning_rate)
        self.multipliers = {key: float((initial or {}).get(key, 0.0)) for key in self.limits}

    def penalty(self, costs: Mapping[str, float]) -> float:
        return sum(self.multipliers[key] * float(costs.get(key, 0.0)) for key in self.limits)

    def update(self, observed: Mapping[str, float]) -> dict[str, float]:
        for key, limit in self.limits.items():
            value = self.multipliers[key] + self.learning_rate * (float(observed.get(key, 0.0)) - limit)
            self.multipliers[key] = max(0.0, value)
        return dict(self.multipliers)
