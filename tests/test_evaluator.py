import unittest

from sieve.core.types import (
    Decision,
    Patch,
    PatchOp,
    RevisionOutput,
    SlotStatus,
    VerificationRequest,
)
from sieve.data.generate import generate_toy_trajectories
from sieve.evaluation.evaluator import evaluate_policy


class VisibleRulePolicy:
    def predict(self, context, greedy: bool = True) -> RevisionOutput:
        del greedy
        observation = context.observation
        current = context.belief_state.get(observation.field_id)
        entity_matches = current is not None and current.entity == observation.entity
        fresh = (
            observation.valid_from is not None
            and observation.valid_from >= observation.observed_at
        )
        if not observation.relevant or not entity_matches or (
            observation.valid_from is not None and not fresh
        ):
            return RevisionOutput(Decision.IGNORE)
        if not fresh:
            return RevisionOutput(
                Decision.HOLD,
                (observation.field_id,),
                (
                    Patch(
                        PatchOp.SET_STATUS,
                        observation.field_id,
                        SlotStatus.PENDING.value,
                    ),
                ),
                VerificationRequest("official_payment_api", observation.field_id),
            )
        return RevisionOutput(
            Decision.UPDATE,
            (observation.field_id,),
            (Patch(PatchOp.SET_VALUE, observation.field_id, observation.value),),
        )


class EvaluatorTests(unittest.TestCase):
    def test_step_and_trajectory_metrics_share_environment_semantics(self) -> None:
        records = generate_toy_trajectories(scenarios=4, seed=7, max_slots=4)
        metrics = evaluate_policy(VisibleRulePolicy(), records, seed=7)
        self.assertEqual(metrics["decision_accuracy"], 1.0)
        self.assertEqual(metrics["patch_validity"], 1.0)
        self.assertEqual(metrics["task_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
