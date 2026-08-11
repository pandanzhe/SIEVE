from __future__ import annotations

from typing import Any

import numpy as np


def cosine_iteration_multiplier(
    iteration: int,
    total_iterations: int,
    warmup_ratio: float,
) -> float:
    """Deterministic LR multiplier indexed by rollout iteration, not DDP batches."""
    if total_iterations <= 0 or not 0 <= iteration < total_iterations:
        raise ValueError("iteration must be inside a positive training horizon")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    warmup = round(total_iterations * warmup_ratio)
    step = iteration + 1
    if warmup > 0 and step <= warmup:
        return step / warmup
    decay_steps = max(1, total_iterations - warmup)
    progress = (step - warmup) / decay_steps
    return float(0.5 * (1.0 + np.cos(np.pi * progress)))


def ddp_token_loss_scale(
    *,
    global_token_count: int,
    world_size: int,
    gradient_accumulation_steps: int,
) -> float:
    """Scale token sums so Accelerate accumulation plus DDP yields a global mean."""
    if global_token_count <= 0:
        raise ValueError("global_token_count must be positive")
    if world_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("world size and accumulation steps must be positive")
    return world_size * gradient_accumulation_steps / global_token_count


def grouped_advantages(
    scores: Any,
    group_ids: Any,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Normalize penalized trajectory scores independently within each GRPO group."""
    values = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(group_ids)
    if values.ndim != 1 or groups.ndim != 1 or len(values) != len(groups):
        raise ValueError("scores and group_ids must be equal-length vectors")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    result = np.zeros_like(values)
    for group_id in np.unique(groups):
        mask = groups == group_id
        group = values[mask]
        if len(group) < 2:
            continue
        result[mask] = (group - group.mean()) / (group.std() + epsilon)
    return result


def clipped_surrogate_loss(
    new_log_probabilities: Any,
    old_log_probabilities: Any,
    advantages: Any,
    *,
    clip_ratio: float,
) -> float:
    new = np.asarray(new_log_probabilities, dtype=np.float64)
    old = np.asarray(old_log_probabilities, dtype=np.float64)
    advantage = np.asarray(advantages, dtype=np.float64)
    if new.shape != old.shape or new.shape != advantage.shape:
        raise ValueError("log probabilities and advantages must share a shape")
    if not 0.0 <= clip_ratio < 1.0:
        raise ValueError("clip_ratio must be in [0, 1)")
    ratio = np.exp(np.clip(new - old, -20.0, 20.0))
    clipped = np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    return -float(np.mean(np.minimum(ratio * advantage, clipped * advantage)))


def sampled_kl(policy_log_probabilities: Any, reference_log_probabilities: Any) -> float:
    """Non-negative sampled KL estimator used by GRPO implementations."""
    policy = np.asarray(policy_log_probabilities, dtype=np.float64)
    reference = np.asarray(reference_log_probabilities, dtype=np.float64)
    if policy.shape != reference.shape:
        raise ValueError("policy and reference log probabilities must share a shape")
    log_ratio = reference - policy
    return float(np.mean(np.exp(np.clip(log_ratio, -20.0, 20.0)) - log_ratio - 1.0))
