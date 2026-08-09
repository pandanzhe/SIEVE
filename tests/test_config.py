import unittest
from pathlib import Path

from sieve.config import load_config, resolve_repo_path


TEST_ROOT = Path(__file__).resolve().parents[1] / "tmp" / "tests"


class ConfigTests(unittest.TestCase):
    def test_relative_path_resolves_inside_explicit_root(self) -> None:
        root = TEST_ROOT.resolve()
        self.assertEqual(resolve_repo_path(root, "outputs/run"), root / "outputs/run")

    def test_path_escape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_repo_path(TEST_ROOT, "../outside")

    def test_yaml_overrides_are_merged(self) -> None:
        config = TEST_ROOT / "config.yaml"
        config.write_text("seed: 7\ntrain:\n  epochs: 2\n", encoding="utf-8")
        loaded = load_config(config, overrides={"train": {"epochs": 4}})
        self.assertEqual(loaded["seed"], 7)
        self.assertEqual(loaded["train"]["epochs"], 4)


    def test_base_config_is_relative_to_declaring_file(self) -> None:
        config_dir = TEST_ROOT / "nested"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "base.yaml").write_text("seed: 13\n", encoding="utf-8")
        child = config_dir / "child.yaml"
        child.write_text("base: base.yaml\ntrain:\n  epochs: 2\n", encoding="utf-8")
        loaded = load_config(child)
        self.assertEqual(loaded["seed"], 13)


if __name__ == "__main__":
    unittest.main()
