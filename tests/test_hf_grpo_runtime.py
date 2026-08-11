import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sieve.rl_data.build import build_rl_corpus
from sieve.training.hf_grpo import (
    _disable_policy_dropout,
    _model_asset_fingerprint,
    _sampling_generation_args,
    _validate_accelerator_profile,
    audit_hf_grpo_inputs,
)
from sieve.training.hf_grpo_config import parse_hf_grpo_config

from tests.test_hf_grpo_config import _config
from tests.test_rl_data import _audit_row


class HFGRPORuntimeTests(unittest.TestCase):
    def test_model_asset_fingerprint_changes_with_tokenizer_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer_config.json").write_text(
                '{"version":1}', encoding="utf-8"
            )
            (model / "model.safetensors").write_bytes(b"weights")
            first = _model_asset_fingerprint(model)
            (model / "tokenizer_config.json").write_text(
                '{"version":2}', encoding="utf-8"
            )
            second = _model_asset_fingerprint(model)
            self.assertIsNotNone(first)
            self.assertNotEqual(first, second)

    def test_audit_rejects_verification_tool_contract_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            rows = [
                _audit_row(index, macro)
                for macro in ("commerce", "service", "workflow")
                for index in range(150)
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            build_rl_corpus(
                source,
                root / "data" / "rl",
                seed=42,
                split_counts={"train": 20, "dev": 10, "test": 10},
            )
            train_file = root / "data" / "rl" / "scenarios" / "train.jsonl"
            raw_lines = train_file.read_text(encoding="utf-8").splitlines()
            first = json.loads(raw_lines[0])
            first["environment_private"]["events"][0]["verification_tool"] = "fake"
            raw_lines[0] = json.dumps(first, ensure_ascii=False, sort_keys=True)
            train_file.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
            manifest_path = root / "data" / "rl" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["split_sha256"]["train"] = hashlib.sha256(
                train_file.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            stage1_dev = root / "data" / "sft" / "clean" / "dev.jsonl"
            stage1_dev.parent.mkdir(parents=True, exist_ok=True)
            stage1_dev.write_text("{}\n", encoding="utf-8")

            raw_config = _config()
            audit = audit_hf_grpo_inputs(parse_hf_grpo_config(raw_config, root))

            self.assertEqual(audit["event_contract_errors"], 1)
            self.assertEqual(audit["split_count_mismatches"], 3)
            self.assertFalse(audit["ready_for_stage2"])

    def test_training_profile_rejects_cpu_before_model_loading(self) -> None:
        class Device:
            type = "cpu"

        class FakeAccelerator:
            num_processes = 2
            device = Device()

        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            _validate_accelerator_profile(FakeAccelerator(), expected_processes=2)

    def test_grpo_forwards_disable_dropout_without_forcing_eval_mode(self) -> None:
        class Dropout:
            def __init__(self) -> None:
                self.p = 0.05

        class FakePolicy:
            training = True

            def __init__(self) -> None:
                self.dropout = Dropout()

            def modules(self):
                return (self, self.dropout)

        policy = FakePolicy()
        self.assertEqual(_disable_policy_dropout(policy), 1)
        self.assertEqual(policy.dropout.p, 0.0)
        self.assertTrue(policy.training)

    def test_sampling_args_neutralize_inherited_logit_warpers(self) -> None:
        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 1

        config = parse_hf_grpo_config(_config(), Path.cwd())
        args = _sampling_generation_args(config, FakeTokenizer())
        self.assertTrue(args["do_sample"])
        self.assertEqual(args["temperature"], 1.0)
        self.assertEqual(args["top_p"], 1.0)
        self.assertEqual(args["top_k"], 0)
        self.assertEqual(args["typical_p"], 1.0)
        self.assertEqual(args["repetition_penalty"], 1.0)
        self.assertEqual(args["no_repeat_ngram_size"], 0)
        self.assertIsNone(args["forced_eos_token_id"])

    def test_offline_audit_validates_splits_without_loading_torch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            rows = [
                _audit_row(index, macro)
                for macro in ("commerce", "service", "workflow")
                for index in range(150)
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            build_rl_corpus(
                source,
                root / "data" / "rl",
                seed=42,
                split_counts={"train": 20, "dev": 10, "test": 10},
            )
            stage1_dev = root / "data" / "sft" / "clean" / "dev.jsonl"
            stage1_dev.parent.mkdir(parents=True, exist_ok=True)
            stage1_dev.write_text("{}\n", encoding="utf-8")
            raw_config = _config()
            raw_config["expected_split_counts"] = {
                "train": 20,
                "dev": 10,
                "test": 10,
            }
            config = parse_hf_grpo_config(raw_config, root)
            audit = audit_hf_grpo_inputs(config)

            self.assertEqual(audit["split_counts"], {"train": 20, "dev": 10, "test": 10})
            self.assertEqual(audit["base_task_overlap"], 0)
            self.assertEqual(audit["duplicate_scenario_ids"], 0)
            self.assertEqual(audit["distractor_split_violations"], 0)
            self.assertEqual(audit["missing_distractor_provenance"], 0)
            self.assertEqual(audit["split_count_mismatches"], 0)
            self.assertTrue(audit["manifest_matches_files"])
            self.assertFalse(audit["model_ready"])
            self.assertFalse(audit["sft_adapter_ready"])
            self.assertFalse(audit["readiness_report_present"])
            self.assertFalse(audit["readiness_report_matches_artifacts"])
            self.assertFalse(audit["stage1_readiness_gate_passed"])
            self.assertFalse(audit["ready_for_stage2"])
            self.assertIn("torch", audit["dependencies"])


if __name__ == "__main__":
    unittest.main()
