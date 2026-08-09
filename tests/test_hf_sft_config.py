import unittest
from pathlib import Path

from sieve.config import load_config
from sieve.training.hf_sft_config import parse_hf_sft_config


ROOT = Path(__file__).resolve().parents[1]


class HFSFTConfigTests(unittest.TestCase):
    def test_qwen_config_defaults_to_one_gpu_bf16_lora(self) -> None:
        raw = load_config(ROOT / "configs" / "sft_qwen3_4b.yaml")

        config = parse_hf_sft_config(raw, ROOT)

        self.assertEqual(config.model.path, ROOT / "model")
        self.assertEqual(config.model.dtype, "bfloat16")
        self.assertFalse(config.model.enable_thinking)
        self.assertEqual(config.lora.rank, 16)
        self.assertEqual(config.hardware.num_processes, 1)
        self.assertEqual(config.train.effective_batch_size, 32)
        self.assertTrue(config.paths.train_file.is_relative_to(ROOT))
        self.assertTrue(config.paths.output_dir.is_relative_to(ROOT))

    def test_two_gpus_are_optional_not_required(self) -> None:
        raw = load_config(ROOT / "configs" / "sft_qwen3_4b.yaml")
        raw["hardware"]["num_processes"] = 2

        config = parse_hf_sft_config(raw, ROOT)

        self.assertEqual(config.hardware.num_processes, 2)

    def test_rejects_invalid_precision(self) -> None:
        raw = load_config(ROOT / "configs" / "sft_qwen3_4b.yaml")
        raw["model"]["dtype"] = "float16"

        with self.assertRaisesRegex(ValueError, "dtype"):
            parse_hf_sft_config(raw, ROOT)

    def test_rejects_non_integral_gradient_accumulation(self) -> None:
        raw = load_config(ROOT / "configs" / "sft_qwen3_4b.yaml")
        raw["train"]["effective_batch_size"] = 31

        with self.assertRaisesRegex(ValueError, "effective_batch_size"):
            parse_hf_sft_config(raw, ROOT)


if __name__ == "__main__":
    unittest.main()


