from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..core.types import RevisionContext, RevisionOutput
from ..data.schema import SFTRecord
from ..environments.revision_env import ReplayRevisionEnvironment, RevisionRLEnvironment
from ..policies.structured_policy import StructuredPolicy, softmax
from .lagrangian import LagrangeController


@dataclass(frozen=True)
class RevisionExperience:
    context: RevisionContext
    output: RevisionOutput
    old_log_probability: float


@dataclass(frozen=True)
class Trajectory:
    experiences: tuple[RevisionExperience, ...]
    task_return: float
    costs: dict[str, float]


def clipped_ratio(new_log_probability: float, old_log_probability: float, clip: float) -> float:
    ratio = float(np.exp(np.clip(new_log_probability - old_log_probability, -20.0, 20.0)))
    return float(np.clip(ratio, 1.0 - clip, 1.0 + clip))


def collect_trajectory(
    policy: StructuredPolicy,
    environment: RevisionRLEnvironment,
    scenario_id: str,
    seed: int,
) -> Trajectory:
    context = environment.reset(scenario_id, seed)
    experiences: list[RevisionExperience] = []
    total_return = 0.0
    total_costs: dict[str, float] = {}

    while True:
        output, _trace = policy.sample_with_trace(context, greedy=False)
        old_log_probability = policy.log_probability(context, output)
        experiences.append(RevisionExperience(context, output, old_log_probability))
        step = environment.step(output)
        total_return += step.reward
        for name, value in step.costs.items():
            total_costs[name] = total_costs.get(name, 0.0) + value
        if step.terminated:
            break
        if step.next_context is None:
            raise RuntimeError("non-terminal environment step must return next_context")
        context = step.next_context

    event_count = max(1, len(experiences))
    normalized_costs = {key: value / event_count for key, value in total_costs.items()}
    return Trajectory(tuple(experiences), total_return, normalized_costs)


def policy_gradient_coefficient(
    new_log_probability: float,
    old_log_probability: float,
    advantage: float,
    clip: float,
) -> float:
    """Derivative multiplier for the clipped surrogate with respect to log-probability."""
    ratio = float(np.exp(np.clip(new_log_probability - old_log_probability, -20.0, 20.0)))
    if (advantage >= 0.0 and ratio > 1.0 + clip) or (
        advantage < 0.0 and ratio < 1.0 - clip
    ):
        return 0.0
    return advantage * ratio


def group_relative_advantages(
    trajectories: Sequence[Trajectory],
    controller: LagrangeController,
    epsilon: float = 1e-6,
) -> np.ndarray:
    scores = np.asarray(
        [trajectory.task_return - controller.penalty(trajectory.costs) for trajectory in trajectories],
        dtype=np.float64,
    )
    if len(scores) < 2:
        return np.zeros_like(scores)
    return (scores - scores.mean()) / (scores.std() + epsilon)


def _update_factorized_policy(
    policy: StructuredPolicy,
    reference: StructuredPolicy,
    experience: RevisionExperience,
    advantage: float,
    learning_rate: float,
    clip: float,
    kl_coefficient: float,
) -> None:
    context, output = experience.context, experience.output
    coefficient = policy_gradient_coefficient(
        policy.log_probability(context, output),
        experience.old_log_probability,
        advantage,
        clip,
    )
    x = policy.featurizer.encode(context)
    indices = policy.output_indices(context, output)

    def update_head(
        weights: np.ndarray,
        bias: np.ndarray,
        probabilities: np.ndarray,
        reference_probabilities: np.ndarray,
        target_index: int,
    ) -> None:
        score_gradient = -probabilities
        score_gradient[target_index] += 1.0
        kl_gradient = probabilities - reference_probabilities
        gradient = coefficient * score_gradient - kl_coefficient * kl_gradient
        weights += learning_rate * np.outer(gradient, x)
        bias += learning_rate * gradient

    update_head(
        policy.w_decision,
        policy.b_decision,
        softmax(policy.decision_logits(context)),
        softmax(reference.decision_logits(context)),
        indices["decision"],
    )
    if "field" in indices:
        update_head(
            policy.w_field,
            policy.b_field,
            softmax(policy.field_logits(context)),
            softmax(reference.field_logits(context)),
            indices["field"],
        )
    if "verify" in indices:
        update_head(
            policy.w_verify,
            policy.b_verify,
            softmax(policy.verify_logits(context)),
            softmax(reference.verify_logits(context)),
            indices["verify"],
        )


def optimize_constrained_grpo(
    policy: StructuredPolicy,
    environment: RevisionRLEnvironment,
    iterations: int,
    groups_per_iteration: int,
    group_size: int,
    learning_rate: float,
    seed: int,
    constraints: Mapping[str, float],
    clip_ratio: float = 0.2,
    kl_coefficient: float = 0.02,
    multiplier_learning_rate: float = 0.08,
) -> list[dict[str, float]]:
    if group_size < 2:
        raise ValueError("GRPO group_size must be at least two")
    if iterations <= 0 or groups_per_iteration <= 0:
        raise ValueError("iterations and groups_per_iteration must be positive")
    if learning_rate <= 0 or multiplier_learning_rate < 0:
        raise ValueError("policy learning rate must be positive and multiplier rate non-negative")
    if not 0.0 <= clip_ratio < 1.0:
        raise ValueError("clip_ratio must be in [0, 1)")
    if kl_coefficient < 0:
        raise ValueError("kl_coefficient must be non-negative")
    if not constraints or any(limit < 0 for limit in constraints.values()):
        raise ValueError("constraints must contain non-negative limits")
    scenario_ids = environment.scenario_ids
    if not scenario_ids:
        raise ValueError("RL environment must expose at least one scenario")

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    reference = policy.clone()
    controller = LagrangeController(constraints, multiplier_learning_rate)
    history: list[dict[str, float]] = []

    for iteration in range(iterations):
        iteration_trajectories: list[Trajectory] = []
        for group_index in range(groups_per_iteration):
            scenario_id = scenario_ids[int(rng.integers(0, len(scenario_ids)))]
            group = [
                collect_trajectory(
                    policy,
                    environment,
                    scenario_id,
                    seed
                    + iteration * groups_per_iteration * group_size
                    + group_index * group_size
                    + sample_index,
                )
                for sample_index in range(group_size)
            ]
            advantages = group_relative_advantages(group, controller)
            for trajectory, advantage in zip(group, advantages, strict=True):
                step_learning_rate = learning_rate / max(1, len(trajectory.experiences))
                for experience in trajectory.experiences:
                    _update_factorized_policy(
                        policy,
                        reference,
                        experience,
                        float(advantage),
                        step_learning_rate,
                        clip_ratio,
                        kl_coefficient,
                    )
            iteration_trajectories.extend(group)

        mean_costs = {
            key: float(np.mean([trajectory.costs.get(key, 0.0) for trajectory in iteration_trajectories]))
            for key in constraints
        }
        controller.update(mean_costs)
        row = {
            "mean_return": float(np.mean([t.task_return for t in iteration_trajectories])),
            "groups": float(groups_per_iteration),
            "group_size": float(group_size),
        }
        row.update({f"cost_{key}": value for key, value in mean_costs.items()})
        row.update({f"lambda_{key}": value for key, value in controller.multipliers.items()})
        history.append(row)
    return history


def train_constrained_grpo(
    policy: StructuredPolicy,
    records: Sequence[SFTRecord],
    iterations: int,
    groups_per_iteration: int,
    group_size: int,
    learning_rate: float,
    seed: int,
    constraints: Mapping[str, float],
    clip_ratio: float = 0.2,
    kl_coefficient: float = 0.02,
    gamma: float = 0.97,
    multiplier_learning_rate: float = 0.08,
) -> list[dict[str, float]]:
    """Train against replay data; use optimize_constrained_grpo for live environments."""
    environment = ReplayRevisionEnvironment(records, gamma=gamma)
    return optimize_constrained_grpo(
        policy=policy,
        environment=environment,
        iterations=iterations,
        groups_per_iteration=groups_per_iteration,
        group_size=group_size,
        learning_rate=learning_rate,
        seed=seed,
        constraints=constraints,
        clip_ratio=clip_ratio,
        kl_coefficient=kl_coefficient,
        multiplier_learning_rate=multiplier_learning_rate,
    )
