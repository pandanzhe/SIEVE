from __future__ import annotations

from dataclasses import replace

from ..core.types import Observation


def perturb(observation: Observation, kind: str) -> Observation:
    if kind == "clean":
        return observation
    if kind == "stale":
        return replace(observation, source="archive", valid_from=0, perturbation=kind)
    if kind == "wrong_entity":
        return replace(observation, entity=f"other-{observation.entity}", perturbation=kind)
    if kind == "weak_source":
        return replace(observation, source="cached_page", valid_from=None, perturbation=kind)
    if kind == "conflict":
        return replace(observation, source="user_message", valid_from=None, perturbation=kind)
    if kind == "irrelevant":
        return replace(observation, field_id="weather", relevant=False, perturbation=kind)
    if kind == "delayed_correction":
        return replace(observation, source="delayed_webhook", valid_from=None, perturbation=kind)
    if kind in {"duplicate", "out_of_order", "partial"}:
        return replace(observation, perturbation=kind)
    raise ValueError(f"unknown perturbation: {kind}")
