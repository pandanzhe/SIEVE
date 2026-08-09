import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PortabilityTests(unittest.TestCase):
    def test_source_and_configs_have_no_windows_absolute_paths(self) -> None:
        candidates = list((ROOT / "src").rglob("*.py")) + list((ROOT / "configs").glob("*.yaml"))
        offenders = []
        for path in candidates:
            if re.search(r"[A-Za-z]:\\", path.read_text(encoding="utf-8-sig")):
                offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [])

    def test_grounded_data_script_uses_shared_root_and_all_pipeline_stages(self) -> None:
        script = (ROOT / "scripts" / "run_grounded_data.sh").read_text(encoding="utf-8")
        self.assertIn("common.sh", script)
        self.assertIn("prepare_grounded_sources", script)
        self.assertIn("generate_grounded", script)
        self.assertIn("export_grounded", script)
        self.assertIn("publish_stage1_data", script)
        config = (ROOT / "configs" / "grounded_full_dry_run.json").read_text(encoding="utf-8")
        self.assertNotRegex(config, r"[A-Za-z]:[\\/]")
    def test_source_preparation_reads_only_persistent_raw_data(self) -> None:
        source = (
            ROOT / "src" / "sieve" / "cli" / "prepare_grounded_sources.py"
        ).read_text(encoding="utf-8")

        self.assertIn('root / "data" / "raw"', source)
        self.assertNotIn('root / "data" / "source_cache" / "repos"', source)

    def test_grounded_intermediates_default_to_disposable_tmp_paths(self) -> None:
        script = (ROOT / "scripts" / "run_grounded_data.sh").read_text(encoding="utf-8")
        full_config = (
            ROOT / "configs" / "grounded_full_dry_run.json"
        ).read_text(encoding="utf-8")
        preview_config = (ROOT / "configs" / "grounded_preview.json").read_text(
            encoding="utf-8"
        )

        self.assertIn("tmp/data_factory", script)
        self.assertIn("data/sft", script)
        self.assertNotIn("data/source_cache", script)
        self.assertNotIn("data/generated", script)
        self.assertIn("tmp/data_factory", full_config)
        self.assertIn("tmp/data_factory", preview_config)

    def test_shell_scripts_share_the_root_bootstrap(self) -> None:
        scripts = list((ROOT / "scripts").glob("*.sh"))
        self.assertTrue(scripts)
        common = (ROOT / "scripts" / "common.sh").read_text(encoding="utf-8")
        self.assertIn('BASH_SOURCE[0]', common)
        self.assertIn('ROOT_DIR=', common)
        for script in scripts:
            if script.name != "common.sh":
                self.assertIn("common.sh", script.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

