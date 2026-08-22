import unittest

from sieve.policies.hf_lora_policy import (
    _ddp_zero_loss_anchors,
    _enable_non_reentrant_gradient_checkpointing,
)


class _CheckpointingBackbone:
    def __init__(self) -> None:
        self.checkpointing_kwargs = None
        self.input_grads_enabled = False

    def gradient_checkpointing_enable(self, **kwargs: object) -> None:
        self.checkpointing_kwargs = kwargs

    def enable_input_require_grads(self) -> None:
        self.input_grads_enabled = True


class _FakeLogits:
    def __init__(self, value: float) -> None:
        self.value = value

    def sum(self) -> "_FakeLogits":
        return self

    def __mul__(self, value: float) -> float:
        return self.value * value


class HFLoraPolicyTests(unittest.TestCase):
    def test_gradient_checkpointing_is_non_reentrant_for_ddp(self) -> None:
        backbone = _CheckpointingBackbone()

        _enable_non_reentrant_gradient_checkpointing(backbone)

        self.assertEqual(
            backbone.checkpointing_kwargs,
            {"gradient_checkpointing_kwargs": {"use_reentrant": False}},
        )
        self.assertTrue(backbone.input_grads_enabled)

    def test_all_structured_heads_anchor_the_ddp_graph_without_changing_loss(self) -> None:
        result = {
            "decision_logits": _FakeLogits(1.0),
            "structure_loss": 2.0,
            "value_loss": 3.0,
        }

        anchors = _ddp_zero_loss_anchors(result)

        self.assertEqual(anchors, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
