import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class Stage1CLIContractTests(unittest.TestCase):
    def test_launch_defaults_to_one_gpu_and_supports_two_gpu_override(self) -> None:
        script = (ROOT / "scripts" / "run_sft.sh").read_text(encoding="utf-8")

        self.assertIn("SIEVE_NUM_GPUS:-1", script)
        self.assertIn("accelerate.commands.launch", script)
        self.assertIn("--num-processes", script)
        self.assertIn("common.sh", script)

    def test_config_uses_canonical_data_and_local_model(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs" / "sft_qwen3_4b.yaml").read_text(encoding="utf-8-sig")
        )

        self.assertEqual(config["model"]["path"], "model")
        self.assertEqual(config["paths"]["train_file"], "data/sft/clean/train.jsonl")
        self.assertEqual(config["paths"]["dev_file"], "data/sft/clean/dev.jsonl")
        self.assertNotIn("test_file", config["paths"])

    def test_training_environment_is_version_pinned(self) -> None:
        requirements = (ROOT / "requirements" / "train.txt").read_text(encoding="utf-8")

        for package in ("torch", "transformers", "peft", "accelerate"):
            self.assertRegex(requirements, rf"(?m)^{package}==")


if __name__ == "__main__":
    unittest.main()
