import unittest
from pathlib import Path

from sieve.config import load_config
from sieve.training.hf_sft import (
    audit_sft_inputs,
    checkpoint_score,
    dependency_report,
)
from sieve.training.hf_sft_config import parse_hf_sft_config


ROOT = Path(__file__).resolve().parents[1]


class HFSFTRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        raw = load_config(ROOT / "configs" / "sft_qwen3_4b.yaml")
        cls.config = parse_hf_sft_config(raw, ROOT)

    def test_offline_audit_uses_only_clean_train_and_dev(self) -> None:
        audit = audit_sft_inputs(self.config)

        self.assertEqual(audit["source_records"], 6000)
        self.assertEqual(audit["train_records"], 4789)
        self.assertEqual(audit["dev_records"], 619)
        self.assertEqual(audit["scenario_overlap"], 0)
        self.assertEqual(audit["label_contract_errors"], 0)
        self.assertFalse(audit["model_ready"])

    def test_dependency_report_does_not_import_optional_packages(self) -> None:
        report = dependency_report()

        self.assertEqual(
            set(report), {"torch", "transformers", "peft", "accelerate"}
        )
        self.assertTrue(all("installed" in item for item in report.values()))

    def test_checkpoint_score_penalizes_false_updates(self) -> None:
        safe = {
            "decision_macro_f1": 0.8,
            "affected_micro_f1": 0.7,
            "verification_f1": 0.6,
            "patch_operation_micro_f1": 0.5,
            "false_update_rate": 0.05,
        }
        unsafe = {**safe, "decision_macro_f1": 0.85, "false_update_rate": 0.30}

        self.assertGreater(checkpoint_score(safe), checkpoint_score(unsafe))


if __name__ == "__main__":
    unittest.main()


