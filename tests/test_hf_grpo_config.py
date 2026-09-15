import tempfile
import unittest
from pathlib import Path

import numpy as np

from sieve.training.hf_grpo_config import parse_hf_grpo_config
from sieve.training.hf_grpo_math import (
    clipped_surrogate_loss,
    cosine_iteration_multiplier,
    ddp_token_loss_scale,
    grouped_advantages,
    sampled_kl,
)


def _config() -> dict:
    return {
        "backend": "hf_constrained_grpo",
        "seed": 42,
        "expected_split_counts": {"train": 1600, "dev": 200, "test": 300},
        "paths": {
            "stage1_dev_file": "data/sft/clean/dev.jsonl",
            "readiness_report": "outputs/stage1-qwen25-3b/readiness.json",
            "train_file": "data/rl/scenarios/train.jsonl",
            "dev_file": "data/rl/scenarios/dev.jsonl",
            "test_file": "data/rl/scenarios/test.jsonl",
            "model_path": "model",
            "sft_checkpoint": "outputs/stage1-qwen25-3b/best",
            "output_dir": "outputs/stage2-qwen25-3b",
        },
        "model": {
            "dtype": "bfloat16",
            "enable_thinking": False,
            "max_prompt_length": 1536,
            "max_completion_length": 384,
            "gradient_checkpointing": True,
            "attention_implementation": "sdpa",
        },
        "rollout": {
            "group_size": 4,
            "groups_per_iteration": 2,
            "max_episode_steps": 8,
            "temperature": 1.0,
            "top_p": 1.0,
        },
        "train": {
            "iterations": 20,
            "update_epochs": 1,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-6,
            "weight_decay": 0.0,
            "warmup_ratio": 0.03,
            "clip_ratio": 0.2,
            "kl_coefficient": 0.02,
            "gamma": 0.97,
            "multiplier_learning_rate": 0.08,
            "max_grad_norm": 1.0,
            "eval_steps": 10,
            "save_steps": 10,
            "save_total_limit": 2,
            "resume_from": None,
        },
        "hardware": {"num_processes": 2},
        "constraints": {
            "false_update": 0.03,
            "unsafe_action": 0.01,
            "verification": 1.00,
            "stall": 0.10,
            "invalid_format": 0.02,
            "invalid_patch": 0.01,
            "collateral_edit": 0.0,
            "budget_violation": 0.0,
        },
    }


class HFGRPOConfigTests(unittest.TestCase):
    def test_config_resolves_all_paths_inside_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = parse_hf_grpo_config(_config(), Path(directory))
            self.assertEqual(config.rollout.group_size, 4)
            self.assertEqual(config.hardware.num_processes, 2)
            self.assertEqual(config.model.dtype, "bfloat16")
            self.assertTrue(config.paths.train_file.is_absolute())
            self.assertEqual(config.constraints["invalid_format"], 0.02)
            self.assertIsNone(config.train.eval_scenario_limit)
            self.assertEqual(
                config.expected_split_counts,
                {"train": 1600, "dev": 200, "test": 300},
            )

    def test_config_rejects_single_sample_groups(self) -> None:
        raw = _config()
        raw["rollout"]["group_size"] = 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                parse_hf_grpo_config(raw, Path(directory))

    def test_config_accepts_exploratory_sampling(self) -> None:
        raw = _config()
        raw["rollout"]["temperature"] = 1.2
        raw["rollout"]["top_p"] = 0.95
        with tempfile.TemporaryDirectory() as directory:
            config = parse_hf_grpo_config(raw, Path(directory))
        self.assertEqual(config.rollout.temperature, 1.2)
        self.assertEqual(config.rollout.top_p, 0.95)

    def test_config_accepts_eval_scenario_limit(self) -> None:
        raw = _config()
        raw["train"]["eval_scenario_limit"] = 20
        with tempfile.TemporaryDirectory() as directory:
            config = parse_hf_grpo_config(raw, Path(directory))
        self.assertEqual(config.train.eval_scenario_limit, 20)

    def test_config_accepts_gigpo_algorithm(self) -> None:
        raw = _config()
        raw["algorithm"] = {
            "name": "gigpo",
            "trajectory_advantage_weight": 0.4,
            "step_advantage_weight": 0.6,
        }
        with tempfile.TemporaryDirectory() as directory:
            config = parse_hf_grpo_config(raw, Path(directory))
        self.assertEqual(config.algorithm.name, "gigpo")
        self.assertEqual(config.algorithm.trajectory_advantage_weight, 0.4)
        self.assertEqual(config.algorithm.step_advantage_weight, 0.6)

    def test_config_rejects_unknown_algorithm(self) -> None:
        raw = _config()
        raw["algorithm"] = {"name": "tree_grpo"}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                parse_hf_grpo_config(raw, Path(directory))

    def test_group_advantages_are_normalized_within_each_group(self) -> None:
        advantages = grouped_advantages(
            np.array([1.0, 2.0, 3.0, 10.0, 10.0]),
            np.array([0, 0, 0, 1, 1]),
        )
        self.assertAlmostEqual(float(advantages[:3].mean()), 0.0, places=6)
        self.assertTrue(np.allclose(advantages[3:], 0.0))

    def test_clipped_objective_and_sampled_kl_are_finite(self) -> None:
        old = np.log(np.array([0.5, 0.5]))
        new = np.log(np.array([0.8, 0.2]))
        advantage = np.array([1.0, -1.0])
        loss = clipped_surrogate_loss(new, old, advantage, clip_ratio=0.2)
        self.assertAlmostEqual(loss, -0.2, places=6)
        divergence = sampled_kl(new, old)
        self.assertGreaterEqual(divergence, 0.0)
        self.assertAlmostEqual(
            ddp_token_loss_scale(
                global_token_count=80,
                world_size=2,
                gradient_accumulation_steps=4,
            ),
            0.1,
        )
        self.assertAlmostEqual(cosine_iteration_multiplier(0, 200, 0.03), 1 / 6)
        self.assertAlmostEqual(cosine_iteration_multiplier(199, 200, 0.03), 0.0)


if __name__ == "__main__":
    unittest.main()
